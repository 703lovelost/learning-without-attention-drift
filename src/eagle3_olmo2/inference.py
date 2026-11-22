"""Run target-only and speculative generation with the pinned SGLang runtime."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

_local_opener = build_opener(ProxyHandler({}))


def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _local_opener.open(request, timeout=timeout) as response:
        return json.loads(response.read())


@contextmanager
def sglang_server(args, mode, settings):
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.target_model,
        "--tokenizer-path",
        args.tokenizer_path or args.target_model,
        "--dtype",
        settings["dtype"],
        "--random-seed",
        str(settings["seed"]),
        "--context-length",
        str(settings["context_length"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--mem-fraction-static",
        str(settings["mem_fraction_static"]),
        "--max-running-requests",
        "1",
        "--attention-backend",
        settings["attention_backend"],
        "--disable-cuda-graph",
    ]
    if mode == "speculative":
        command.extend(
            (
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                args.draft_model,
                "--speculative-num-steps",
                str(settings["speculative_steps"]),
                "--speculative-eagle-topk",
                str(settings["speculative_topk"]),
                "--speculative-num-draft-tokens",
                str(settings["speculative_draft_tokens"]),
            )
        )
    print(
        f"Loading {mode} model {args.target_model}, "
        f"checkpoint={args.checkpoint}, seed={settings['seed']}",
        flush=True,
    )
    source_root = str(Path(__file__).resolve().parents[1])
    bootstrap_dir = str(Path(__file__).resolve().parent / "sglang_bootstrap")
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (bootstrap_dir, source_root, python_path) if part
    )
    # Refuse an occupied port so an unrelated SGLang instance cannot satisfy health.
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    server = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    logs: deque[str] = deque(maxlen=100)

    def drain_logs() -> None:
        if server.stdout is not None:
            for line in server.stdout:
                logs.append(line)

    log_thread = threading.Thread(target=drain_logs, daemon=True)
    log_thread.start()
    try:
        deadline = time.monotonic() + args.startup_timeout
        ready = False
        started = time.monotonic()
        while time.monotonic() < deadline:
            if server.poll() is not None:
                break
            try:
                with _local_opener.open(
                    f"http://127.0.0.1:{args.port}/health", timeout=2
                ):
                    ready = True
                    break
            except (URLError, TimeoutError, OSError):
                time.sleep(2)
        if not ready:
            raise RuntimeError("SGLang server did not become ready: " + "".join(logs))
        print(f"{mode} ready", flush=True)
        yield time.monotonic() - started
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log_thread.join(timeout=2)


SMOKE_PROMPT = "What is 2 plus 2? Answer briefly."


def load_prompts(args):
    if not args.suite:
        return [
            {
                "id": "single" if args.prompt else "smoke",
                "category": "interactive" if args.prompt else "arithmetic",
                "language": "und" if args.prompt else "en",
                "context_bucket": "short",
                "messages": [{"role": "user", "content": args.prompt or SMOKE_PROMPT}],
                "max_new_tokens": args.max_new_tokens,
                "check_type": "manual",
            }
        ]
    rows = []
    seen = set()
    with args.suite.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            fields = {
                "id",
                "category",
                "language",
                "context_bucket",
                "messages",
                "max_new_tokens",
                "check_type",
            }
            if fields - row.keys() or row["id"] in seen:
                raise ValueError(f"Invalid or duplicate prompt entry: {row}")
            if not isinstance(row["max_new_tokens"], int) or row["max_new_tokens"] < 1:
                raise ValueError(f"Invalid generation budget for {row['id']}")
            if row["check_type"] not in ("exact", "json", "manual"):
                raise ValueError(f"Unsupported prompt check for {row['id']}")
            if row["check_type"] != "manual" and "expected" not in row:
                raise ValueError(f"Missing expected answer for {row['id']}")
            if (
                not row["messages"]
                or row["messages"][-1].get("role") != "user"
                or any(
                    item.get("role") not in ("user", "assistant", "system")
                    or not isinstance(item.get("content"), str)
                    for item in row["messages"]
                )
            ):
                raise ValueError(f"Invalid messages for {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Prompt suite is empty")
    return rows


def _answer_check(prompt, text):
    if prompt["check_type"] == "manual":
        return None
    if prompt["check_type"] == "exact":
        return text.strip() == prompt["expected"]
    try:
        return json.dumps(json.loads(text), sort_keys=True) == json.dumps(
            prompt["expected"], sort_keys=True
        )
    except ValueError:
        return False


def _request(args, mode, prompt, input_ids, settings):
    start = time.monotonic()
    record = {
        "mode": mode,
        "id": prompt["id"],
        "category": prompt["category"],
        "context_bucket": prompt["context_bucket"],
        "input_tokens": len(input_ids),
    }
    if len(input_ids) + prompt["max_new_tokens"] > settings["context_length"]:
        return {**record, "error": "prompt_and_completion_exceed_context_limit"}
    try:
        result = _post(
            f"http://127.0.0.1:{args.port}/generate",
            {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": prompt["max_new_tokens"],
                },
            },
            timeout=settings["request_timeout"],
        )
    except (OSError, TimeoutError, ValueError) as exc:
        return {**record, "error": type(exc).__name__}
    elapsed = time.monotonic() - start
    tokens = result.get("output_ids")
    response = result.get("text")
    meta = result.get("meta_info") or {}
    if (
        not isinstance(tokens, list)
        or not tokens
        or not isinstance(response, str)
        or not response.strip()
    ):
        return {**record, "error": "missing_output_tokens_or_text", "response": result}
    metrics = {
        "generation/latency_s": elapsed,
        "generation/output_tokens_per_s": len(tokens) / elapsed,
    }
    if mode == "speculative":
        accepted = meta.get("spec_num_correct_drafts")
        proposed = meta.get("spec_num_proposed_drafts")
        calls = meta.get("spec_verify_ct")
        for key, value in (
            ("accepted_draft_tokens", accepted),
            ("proposed_draft_slots", proposed),
            ("verify_calls", calls),
            ("completion_tokens_per_verify", meta.get("spec_accept_length")),
        ):
            if value is not None:
                metrics[f"generation/{key}"] = value
        if isinstance(proposed, int) and proposed > 0 and isinstance(accepted, int):
            metrics["generation/acceptance_rate"] = accepted / proposed
        else:
            record["acceptance_rate_unavailable"] = (
                "runtime reported no positive proposal-slot count"
            )
    return {
        **record,
        "metrics": metrics,
        "response": result,
        "task_check": _answer_check(prompt, response),
    }


def _summary(records):
    grouped = {}
    for row in records:
        for group in (
            (row["mode"], "all", "all"),
            (row["mode"], row["category"], row["context_bucket"]),
        ):
            grouped.setdefault(group, []).append(row)
    summary = []
    for (mode, category, context), rows in grouped.items():
        valid = [row for row in rows if "error" not in row]
        entry = {
            "mode": mode,
            "category": category,
            "context_bucket": context,
            "successes": len(valid),
            "errors": len(rows) - len(valid),
        }
        if valid:
            latency = sorted(row["metrics"]["generation/latency_s"] for row in valid)
            entry["latency_p50_s"] = statistics.median(latency)
            entry["mean_output_tokens_per_s"] = statistics.mean(
                row["metrics"]["generation/output_tokens_per_s"] for row in valid
            )
            scored = [
                row["task_check"] for row in valid if row["task_check"] is not None
            ]
            if scored:
                entry["task_checks_passed"] = sum(scored)
                entry["task_checks_total"] = len(scored)
            entry["latency_p95_s"] = (
                latency[-1]
                if len(latency) == 1
                else statistics.quantiles(latency, n=20, method="inclusive")[-1]
            )
            rates = [
                row["metrics"]["generation/acceptance_rate"]
                for row in valid
                if "generation/acceptance_rate" in row["metrics"]
            ]
            if rates:
                entry["mean_acceptance_rate"] = statistics.mean(rates)
        summary.append(entry)
    return summary


def main() -> None:
    import yaml
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--draft-model")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--mode",
        choices=("target-only", "speculative", "compare"),
        default="speculative",
    )
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--startup-timeout", type=int, default=900)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--suite", type=Path)
    inputs.add_argument("--prompt")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.mode != "target-only" and not args.draft_model:
        parser.error("--draft-model is required for speculative and compare modes")
    settings = {
        "dtype": "bfloat16",
        "seed": 42,
        "context_length": 4096,
        "mem_fraction_static": 0.55,
        "attention_backend": "triton",
        "speculative_steps": 3,
        "speculative_topk": 1,
        "speculative_draft_tokens": 4,
        "request_timeout": 120,
        "benchmark_warmup": 1,
        "benchmark_repeats": 3,
    }
    if args.settings:
        settings.update(
            yaml.safe_load(args.settings.read_text(encoding="utf-8"))["evaluation"].get(
                "generation", {}
            )
        )
    if settings["benchmark_repeats"] <= 0 or settings["benchmark_warmup"] < 0:
        parser.error("Benchmark repeats must be positive and warmup nonnegative")
    prompts = load_prompts(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.target_model)
    inputs = [
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
        for row in prompts
    ]
    if args.trace:
        print("Per-round trace unavailable in the pinned SGLang response", flush=True)
    modes = ("target-only", "speculative") if args.mode == "compare" else (args.mode,)
    repeats = settings["benchmark_repeats"] if args.mode == "compare" else 1
    rows = []
    startup = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".jsonl").open("w", encoding="utf-8") as events:
        for mode in modes:
            with sglang_server(args, mode, settings) as ready_seconds:
                startup[mode] = ready_seconds
                if args.mode == "compare":
                    for _ in range(settings["benchmark_warmup"]):
                        _request(args, mode, prompts[0], inputs[0], settings)
                for prompt, input_ids in zip(prompts, inputs):
                    for repetition in range(repeats):
                        record = _request(args, mode, prompt, input_ids, settings)
                        record["repetition"] = repetition
                        rows.append(record)
                        events.write(json.dumps(record, ensure_ascii=False) + "\n")
                        events.flush()
                    message = record.get("error") or record["response"]["text"]
                    print(f"{mode} {prompt['id']} {message}", flush=True)
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_model": args.target_model,
        "checkpoint": args.checkpoint,
        "draft_model": args.draft_model,
        "settings": settings,
        "prompts": prompts,
        "prompt_suite_sha256": (
            hashlib.sha256(args.suite.read_bytes()).hexdigest() if args.suite else None
        ),
        "trace": "unavailable" if args.trace else "not_requested",
        "startup_seconds": startup,
        "summary": _summary(rows),
        "requests": rows,
    }
    paired = {}
    for record in rows:
        if "error" not in record:
            paired.setdefault((record["id"], record["repetition"]), {})[
                record["mode"]
            ] = record
    comparisons = []
    for (prompt_id, repetition), pair in paired.items():
        if len(pair) != 2:
            continue
        baseline, speculative = pair["target-only"], pair["speculative"]
        comparisons.append(
            {
                "id": prompt_id,
                "repetition": repetition,
                "greedy_parity": baseline["response"]["output_ids"]
                == speculative["response"]["output_ids"],
                "speedup": baseline["metrics"]["generation/latency_s"]
                / speculative["metrics"]["generation/latency_s"],
            }
        )
    report["comparisons"] = comparisons
    if len(rows) == 1 and "response" in rows[0]:
        report["response"] = rows[0]["response"]
        report["spec_accept_length"] = (report["response"].get("meta_info") or {}).get(
            "spec_accept_length"
        )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    parts = [
        "<!doctype html><meta charset=utf-8><title>EAGLE-3 generation</title>",
        "<h1>EAGLE-3 generation</h1>",
        f"<p>Per-round trace: {report['trace']}</p>",
        "<details open><summary>Metrics and parity</summary><pre>",
        html.escape(
            json.dumps(
                {"summary": report["summary"], "comparisons": comparisons}, indent=2
            )
        ),
        "</pre></details>",
    ]
    for row in rows:
        label = html.escape(f"{row['id']} · {row['mode']} · {row['repetition'] + 1}")
        text = html.escape(row.get("error") or row["response"]["text"])
        parts.append(f"<details><summary>{label}</summary><pre>{text}</pre></details>")
    args.output.with_suffix(".html").write_text("\n".join(parts), encoding="utf-8")
    if all("error" in row for row in rows):
        raise RuntimeError(f"Every generation request failed; inspect {args.output}")
    print(f"Saved generation results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
