"""TASK B1 (GPU) — FREE-decode forced-horizon (min_accept_round=4) for the 3 BASE
models x sellers {B,C,D}, N=500, same seeds. Pairs with the constrained
forced-horizon already in results/forced_horizon/ (same seeds, guided=True) for
the context-matched override labelling (B2). Saves results/base_artifact/
free_<seller>_<ckpt>.jsonl ({"state": s}, identical schema to the constrained run)."""
import argparse, json, os
import game_lib as G
from engine_game import GameLLM
from gen500 import HF
from seller_exp import run_vs_seller

BASE = {"base", "z_base", "t_base"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=sorted(BASE))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    args = ap.parse_args()
    role, hfid = HF[args.ckpt]
    G.MODELS[role] = hfid
    grid = G.make_grid(args.n)
    os.makedirs("results/base_artifact", exist_ok=True)
    eng = GameLLM(role, gpu_mem=args.gpu_mem)
    for sk in ["B", "C", "D"]:
        states = run_vs_seller(eng, role, grid, sk, min_accept_round=args.k,
                               guided=False)        # FREE decode
        with open(f"results/base_artifact/free_{sk}_{args.ckpt}.jsonl", "w") as f:
            for s in states:
                f.write(json.dumps({"state": s}) + "\n")
        print(f"[b1-free] {args.ckpt} vs {sk} (k={args.k}) done", flush=True)


if __name__ == "__main__":
    main()
