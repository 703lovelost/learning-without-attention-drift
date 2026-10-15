"""Small runtime adapter for OLMo-2 EAGLE3 capture in pinned SGLang."""

from __future__ import annotations


def install_olmo2_eagle3_capture_adapter() -> None:
    """Add the hidden-state capture hook missing from pinned SGLang 0.5.18.

    This patch is process-local and leaves the SpecForge/SGLang submodules
    untouched. SGLang's Llama implementation captures the state immediately
    before block ``layer_id + 1`` for each requested zero-based layer ID; this
    adapter follows the same indexing and output contract for Olmo2.
    """
    from sglang.srt.models.olmo2 import Olmo2ForCausalLM, Olmo2Model

    if getattr(Olmo2ForCausalLM, "_eagle3_capture_adapter_installed", False):
        return

    native_model_forward = Olmo2Model.forward
    native_target_forward = Olmo2ForCausalLM.forward

    def olmo2_model_forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
    ):
        if not getattr(self, "capture_aux_hidden_states", False):
            return native_model_forward(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
            )

        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        layers_to_capture = set(self.layers_to_capture)
        aux_hidden_states = []
        for layer_id, decoder_layer in enumerate(self.layers):
            if layer_id in layers_to_capture:
                aux_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(positions, hidden_states, forward_batch)
        hidden_states = self.norm(hidden_states)
        return hidden_states, aux_hidden_states

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        self.capture_aux_hidden_states = True
        self.model.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = int(self.config.num_hidden_layers)
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            # SGLang captures the hidden state at the input to block N+1.
            self.model.layers_to_capture = [int(layer_id) + 1 for layer_id in layer_ids]

    def get_embed_and_head(self):
        """Expose the embedding/head pair expected by SGLang's EAGLE worker."""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def olmo2_target_forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        **kwargs,
    ):
        if not getattr(self, "capture_aux_hidden_states", False):
            return native_target_forward(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                **kwargs,
            )

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )

    Olmo2Model.forward = olmo2_model_forward
    Olmo2ForCausalLM.forward = olmo2_target_forward
    Olmo2ForCausalLM.set_eagle3_layers_to_capture = set_eagle3_layers_to_capture
    Olmo2ForCausalLM.get_embed_and_head = get_embed_and_head
    Olmo2ForCausalLM._eagle3_capture_adapter_installed = True

