"""
vLLM game engine: loads ONE model (bf16), generates buyer moves either under
guided JSON (constrained, primary) or free decoding (strict/lenient diagnostics).
Reuses the probe's isolated venv + off-root caches (set in game_lib).
"""
import os
import game_lib as G

_GPU = os.environ.get("PROBE_GPU", "1")


class GameLLM:
    def __init__(self, role, gpu_mem=0.45, max_model_len=2048):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", _GPU)
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from transformers import AutoTokenizer
        self.role = role
        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams
        path = G.MODELS[role]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.llm = LLM(model=path, dtype="bfloat16",
                       gpu_memory_utilization=gpu_mem,
                       max_model_len=max_model_len, seed=G.SEED,
                       enforce_eager=True)

    def render(self, role, v_b, history, ask, t, T=G.T_ROUNDS):
        if role == "aligned":
            msgs = G.build_aligned_buyer_messages(v_b, history, ask, t, T)
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        return G.build_base_buyer_prompt(v_b, history, ask, t, T)

    def generate(self, prompts, role, guided, seeds=None):
        """prompts: list[str]; returns list[str] (one completion each)."""
        so = (self.StructuredOutputsParams(json=G.ACTION_JSON_SCHEMA)
              if guided else None)
        stop = None if role == "aligned" else G.BASE_STOP
        if seeds is None:
            seeds = [G.SEED] * len(prompts)
        sps = [self.SamplingParams(
                   temperature=G.TEMPERATURE, top_p=1.0,
                   max_tokens=G.MAX_TOKENS, seed=int(sd),
                   structured_outputs=so, stop=stop)
               for sd in seeds]
        outs = self.llm.generate(prompts, sps)
        return [o.outputs[0].text for o in outs]
