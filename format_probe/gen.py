"""
Generation driver. Loads ONE model via the chosen backend, generates n
completions per (action_type x history_bin) cell under free decoding
(temp 0.7, fixed seed, NO constraints), and APPENDS one rich record per
generation (full failure diagnosis) to a jsonl.

Sequential model loading: run one role per process.

Usage:
  python gen.py --role aligned --backend vllm --n 20 \
      --histories early_low,mid,late_high --out out/m2_runs.jsonl
"""
import argparse, json, os, time

import probe_lib as PL
from engine import make_engine


def generate_cells(eng, role, cells, n, greedy=False, backend=None,
                   mode=None):
    """Render prompts for each (action_type, history_bin) cell, generate, and
    return a flat list of rich records."""
    prompts = [eng._render(h, a) for (a, h) in cells]
    rows = []
    # base needs stop strings; aligned ends naturally via the chat template.
    for (a, h), comps in zip(
            cells,
            eng.generate(prompts, n=n,
                         stop=(None if role == "aligned" else PL.BASE_STOP),
                         greedy=greedy)):
        for raw in comps:
            rows.append(PL.make_record(role, a, h, raw, backend=backend,
                                       mode=mode))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["base", "aligned"])
    ap.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--actions", default=",".join(PL.ACTION_ORDER))
    ap.add_argument("--histories", default=",".join(PL.HISTORY_ORDER))
    ap.add_argument("--out", default="out/probe_runs.jsonl")
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    args = ap.parse_args()

    actions = [a for a in args.actions.split(",") if a]
    histories = [h for h in args.histories.split(",") if h]
    cells = [(a, h) for a in actions for h in histories]

    t0 = time.time()
    kw = {"gpu_mem": args.gpu_mem} if args.backend == "vllm" else {}
    eng = make_engine(args.role, backend=args.backend, **kw)
    print(f"[gen] engine ready in {time.time()-t0:.0f}s; "
          f"{len(cells)} cells x n={args.n}", flush=True)

    t0 = time.time()
    rows = generate_cells(eng, args.role, cells, args.n, backend=args.backend)
    print(f"[gen] generated {len(rows)} rows in {time.time()-t0:.0f}s",
          flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = sum(r["parse_success"] for r in rows)
    print(f"[gen] wrote {len(rows)} rows to {args.out}  "
          f"(overall parse_success={ok/len(rows):.3f})", flush=True)


if __name__ == "__main__":
    main()
