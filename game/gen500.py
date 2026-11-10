"""
N=500 generation for one checkpoint: load the model once, run BOTH passes
(free Pass A for strict/lenient/matched, guided Pass B for constrained), save
to runs/n500/. Sequential checkpoint loading is handled by the caller (one
process per checkpoint).
"""
import argparse, json, os
import game_lib as G
from engine_game import GameLLM
from runner import run_episodes
from freepass import run_free_pass

CKPT = {
    "base": "base", "sft": None, "dpo": None, "instruct": "aligned",
}
# map study checkpoint name -> (role-for-prompting, hf id)
# role "base" = frozen few-shot completion fallback (no chat template);
# role "aligned" = chat template. Mistral base reuses the SAME fallback as
# OLMo base, so both families' base conditions are constructed identically.
HF = {
    "base":     ("base",    "allenai/OLMo-2-1124-7B"),
    "sft":      ("aligned", "allenai/OLMo-2-1124-7B-SFT"),
    "dpo":      ("aligned", "allenai/OLMo-2-1124-7B-DPO"),
    "instruct": ("aligned", "allenai/OLMo-2-1124-7B-Instruct"),
    # Zephyr family (Mistral-7B-v0.1 lineage)
    "z_base":   ("base",    "mistralai/Mistral-7B-v0.1"),
    "z_sft":    ("aligned", "alignment-handbook/zephyr-7b-sft-full"),
    "z_dpo":    ("aligned", "HuggingFaceH4/zephyr-7b-beta"),
    # Tulu-3 family (Llama-3.1-8B lineage; base via ungated NousResearch mirror)
    "t_base":   ("base",    "NousResearch/Meta-Llama-3.1-8B"),
    "t_sft":    ("aligned", "allenai/Llama-3.1-Tulu-3-8B-SFT"),
    "t_dpo":    ("aligned", "allenai/Llama-3.1-Tulu-3-8B-DPO"),
    "t_rlvr":   ("aligned", "allenai/Llama-3.1-Tulu-3-8B"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(HF))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    args = ap.parse_args()

    role, hfid = HF[args.ckpt]
    # base uses few-shot completion; SFT/DPO/Instruct use the chat template.
    G.MODELS[role] = hfid          # point the loader at this checkpoint
    grid = G.make_grid(args.n)
    os.makedirs("runs/n500", exist_ok=True)

    eng = GameLLM(role, gpu_mem=args.gpu_mem)

    # Pass A (free)
    run_free_pass(eng, role, grid, f"runs/n500/passA_{args.ckpt}.jsonl")
    print(f"[gen500] {args.ckpt} passA done", flush=True)

    # Pass B (constrained)
    states = run_episodes(eng, role, grid, "constrained")
    with open(f"runs/n500/constrained_{args.ckpt}.jsonl", "w") as f:
        for s in states:
            f.write(json.dumps({"state": s}) + "\n")
    print(f"[gen500] {args.ckpt} constrained done", flush=True)


if __name__ == "__main__":
    main()
