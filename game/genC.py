"""Constrained-only generation at N=2000 for one checkpoint (Milestone C).
Saves runs/n2000/constrained_<ckpt>.jsonl. No Pass A (constrained only)."""
import argparse, json, os
import game_lib as G
from engine_game import GameLLM
from runner import run_episodes
from gen500 import HF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(HF))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gpu-mem", type=float, default=0.60)
    args = ap.parse_args()
    role, hfid = HF[args.ckpt]
    G.MODELS[role] = hfid
    grid = G.make_grid(args.n)
    os.makedirs("runs/n2000", exist_ok=True)
    eng = GameLLM(role, gpu_mem=args.gpu_mem)
    states = run_episodes(eng, role, grid, "constrained")
    with open(f"runs/n2000/constrained_{args.ckpt}.jsonl", "w") as f:
        for s in states:
            f.write(json.dumps({"state": s}) + "\n")
    print(f"[genC] {args.ckpt} constrained N={args.n} done", flush=True)


if __name__ == "__main__":
    main()
