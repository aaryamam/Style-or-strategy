"""Run the seller-parameterization experiment for one checkpoint against all
four sellers (constrained decoding, N=500 paired, same seeded grid).
Saves runs/seller_exp/<seller>_<ckpt>.jsonl."""
import argparse, json, os
import game_lib as G
from engine_game import GameLLM
from gen500 import HF
from seller_exp import run_vs_seller, SELLERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(HF))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    args = ap.parse_args()
    role, hfid = HF[args.ckpt]
    G.MODELS[role] = hfid
    grid = G.make_grid(args.n)
    os.makedirs("runs/seller_exp", exist_ok=True)
    eng = GameLLM(role, gpu_mem=args.gpu_mem)
    for sk in ["A", "B", "C", "D"]:
        states = run_vs_seller(eng, role, grid, sk)
        with open(f"runs/seller_exp/{sk}_{args.ckpt}.jsonl", "w") as f:
            for s in states:
                f.write(json.dumps({"state": s}) + "\n")
        print(f"[gen_seller] {args.ckpt} vs {sk} done", flush=True)


if __name__ == "__main__":
    main()
