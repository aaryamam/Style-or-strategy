"""TASK 2 (step 2, GPU) — realized-history firing probe generation.
Reuses the FROZEN probe harness (engine.py, probe_lib: ROLE, FORMAT_REQUEST,
ACTION_TYPES, few-shot demos, parse_diagnose) verbatim; the ONLY change is that
the 'Negotiation so far' text is a REALIZED history (results/realized_probe/
histories.json) instead of a synthetic HISTORIES bin. Free single-shot decoding,
~150 gen/action/bin (30 shared histories/bin x n=5). Saves
../game/results/realized_probe/realized_<ckpt>.jsonl (probe_runs schema)."""
import json, argparse, os
import engine as E
import probe_lib as PL

CK = {
    "olmo_base": ("base", "allenai/OLMo-2-1124-7B"),
    "olmo_inst": ("aligned", "allenai/OLMo-2-1124-7B-Instruct"),
    "z_base": ("base", "mistralai/Mistral-7B-v0.1"),
    "z_dpo": ("aligned", "HuggingFaceH4/zephyr-7b-beta"),
    "t_base": ("base", "NousResearch/Meta-Llama-3.1-8B"),
    "t_rlvr": ("aligned", "allenai/Llama-3.1-Tulu-3-8B"),
}
HIST = "../game/results/realized_probe/histories.json"
OUT = "../game/results/realized_probe"


def build_prompt(role, tok, hist, action_type):
    instr = PL.ACTION_TYPES[action_type]["instruction"]
    if role == "aligned":
        msgs = [{"role": "system", "content": PL.ROLE + "\n\n" + PL.FORMAT_REQUEST},
                {"role": "user", "content":
                 f"Negotiation so far:\n{hist}\n\nYour private decision for this "
                 f"turn: {instr}\n\nNow give your move."}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    parts = [PL._BASE_PREAMBLE]
    for d in PL._FEWSHOT_DEMOS:
        parts.append(PL._base_turn_block(d["situation"], d["decision"], d["move"]))
    parts.append(PL._base_turn_block(hist, instr, None))
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, choices=list(CK))
    ap.add_argument("--n", type=int, default=5)        # gens per (history,action)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    args = ap.parse_args()
    role, hfid = CK[args.ckpt]
    E.MODELS[role] = hfid
    eng = E.make_engine(role, backend="vllm", gpu_mem=args.gpu_mem)
    tok = eng.tok
    hists = json.load(open(HIST))

    prompts, meta = [], []
    for b in ["early", "mid", "late"]:
        for h in hists[b]:
            for a in PL.ACTION_ORDER:
                prompts.append(build_prompt(role, tok, h["history"], a))
                meta.append((a, b, h["eff_room"], h["round"]))
    stop = None if role == "aligned" else PL.BASE_STOP
    outs = eng.generate(prompts, n=args.n, stop=stop, greedy=False)

    os.makedirs(OUT, exist_ok=True)
    rows = 0
    with open(f"{OUT}/realized_{args.ckpt}.jsonl", "w") as f:
        for (a, b, ro, rd), comps in zip(meta, outs):
            for raw in comps:
                d = PL.parse_diagnose(raw)
                f.write(json.dumps({
                    "model": args.ckpt, "action_type": a, "history_bin": b,
                    "eff_room": ro, "round": rd,
                    "parse_success": d["parse_success"],
                    "parsed_action": d["parsed_action"], "raw_output": raw}) + "\n")
                rows += 1
    ok = sum(1 for l in open(f"{OUT}/realized_{args.ckpt}.jsonl")
             if json.loads(l)["parse_success"])
    print(f"[realized] {args.ckpt} wrote {rows} rows, overall parse_success="
          f"{ok/rows:.4f}", flush=True)


if __name__ == "__main__":
    main()
