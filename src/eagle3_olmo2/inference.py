"""SGLang target-plus-drafter generation smoke for an exported EAGLE-3 draft."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.target_model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 plus 2? Answer briefly."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.target_model,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", args.draft_model,
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--mem-fraction-static", "0.55",
        "--max-running-requests", "1",
        "--attention-backend", "triton",
        "--disable-cuda-graph",
    ]
    source_root = str(Path(__file__).resolve().parents[1])
    bootstrap_dir = str(Path(__file__).resolve().parent / "sglang_bootstrap")
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (bootstrap_dir, source_root, python_path) if part
    )
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
            raise RuntimeError("SGLang target/drafter server did not become ready: " + "".join(logs))
        result = _post(
            f"http://127.0.0.1:{args.port}/generate",
            {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens}},
        )
        output = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "prompt": prompt,
            "response": result,
            "spec_accept_length": (result.get("meta_info") or {}).get("spec_accept_length"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Inference smoke passed; saved response to {args.output}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log_thread.join(timeout=2)


if __name__ == "__main__":
    main()
