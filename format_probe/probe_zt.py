"""
Minimal AXIS-1 teacher-forced probe re-run for Zephyr + Tulu (base+final).
Reuses the FROZEN OLMo probe harness (engine.py / gen.generate_cells / probe_lib)
exactly: same 5 forced action_types x 3 history_bins, same free single-shot
decoding (NOT constrained), same parser. Only the model is swapped and the
`model` field re-tagged to the checkpoint name. Appends to
out/probe_runs_zephyr_tulu.jsonl (schema-compatible with probe_runs.jsonl).
"""
import argparse, json, os
import engine as E
import probe_lib as PL
from gen import generate_cells

CK = {
    "z_base": ("base",    "mistralai/Mistral-7B-v0.1"),
    "z_dpo":  ("aligned", "HuggingFaceH4/zephyr-7b-beta"),
    "t_base": ("base",    "NousResearch/Meta-Llama-3.1-8B"),
    "t_rlvr": ("aligned", "allenai/Llama-3.1-Tulu-3-8B"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(CK))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--out", default="out/probe_runs_zephyr_tulu.jsonl")
    args = ap.parse_args()

    role, hfid = CK[args.ckpt]
    E.MODELS[role] = hfid                      # point the frozen harness here
    eng = E.make_engine(role, backend="vllm", gpu_mem=args.gpu_mem)

    cells = [(a, h) for a in PL.ACTION_ORDER for h in PL.HISTORY_ORDER]
    rows = generate_cells(eng, role, cells, args.n, backend="vllm")
    for r in rows:
        r["model"] = args.ckpt                 # re-tag (was the role)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = sum(r["parse_success"] for r in rows)
    print(f"[probe_zt] {args.ckpt} ({hfid}) role={role}: wrote {len(rows)} rows "
          f"(overall parse_success={ok/len(rows):.3f})", flush=True)


if __name__ == "__main__":
    main()
