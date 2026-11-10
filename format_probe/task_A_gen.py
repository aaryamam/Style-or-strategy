"""TASK A (step 2, GPU) — robustness re-generation of the realized probe for the
two fired checkpoints (olmo_base, t_rlvr). Reuses task2_realized_gen.build_prompt
+ probe_lib parser/actions verbatim. Raised gens (n=10) and a --seed knob (sets
PL.SEED before engine construction → genuinely independent sampling) for A1's
second-seed test. Generates on early(90)+late(72) pool; analysis applies the A2
cutoff/eff_room bands. Saves results/realized_robust/probe_<ckpt>_seed<seed>.jsonl."""
import json, argparse, os
import engine as E
import probe_lib as PL
from task2_realized_gen import build_prompt, CK

HIST = "../game/results/realized_robust/hist_pool.json"
OUT = "../game/results/realized_robust"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=["olmo_base", "t_rlvr"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hist", default=HIST)
    ap.add_argument("--tag", default="")        # output filename tag (e.g. "_own")
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    args = ap.parse_args()
    PL.SEED = args.seed                       # drives LLM seed + SamplingParams seed
    role, hfid = CK[args.ckpt]
    E.MODELS[role] = hfid
    eng = E.make_engine(role, backend="vllm", gpu_mem=args.gpu_mem)
    tok = eng.tok
    hists = json.load(open(args.hist))

    prompts, meta = [], []
    for poolname in hists:                    # early / late / late_loose (A3)
        for h in hists[poolname]:
            for a in PL.ACTION_ORDER:
                prompts.append(build_prompt(role, tok, h["history"], a))
                meta.append((a, poolname, h["eff_room"], h["round"]))
    stop = None if role == "aligned" else PL.BASE_STOP
    outs = eng.generate(prompts, n=args.n, stop=stop, greedy=False)

    os.makedirs(OUT, exist_ok=True)
    fp = f"{OUT}/probe_{args.ckpt}{args.tag}_seed{args.seed}.jsonl"
    rows = 0
    with open(fp, "w") as f:
        for (a, pool, ro, rd), comps in zip(meta, outs):
            for raw in comps:
                d = PL.parse_diagnose(raw)
                f.write(json.dumps({
                    "model": args.ckpt, "action_type": a, "pool": pool,
                    "eff_room": ro, "round": rd, "seed": args.seed,
                    "parse_success": d["parse_success"],
                    "parsed_action": d["parsed_action"]}) + "\n")
                rows += 1
    print(f"[A {args.ckpt} seed{args.seed}] wrote {rows} rows", flush=True)


if __name__ == "__main__":
    main()
