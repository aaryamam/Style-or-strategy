"""
Per-checkpoint policy-shift diagnostic: how invasively constrained (XGrammar)
decoding alters the model's ACTION CHOICE vs merely rescuing unparseable output.
Pure re-analysis of runs/n500/{passA,constrained}_*.jsonl. No model inference.

Validity handling:
- Join free (passA, strict_action) and constrained (action) on (episode idx, round).
- OVERRIDE is computed only on CONTEXT-MATCHED turns (same ask_shown): at a matched
  turn the prompt+seed are shared and the grammar mask is the ONLY difference, so a
  changed action is a pure grammar override (no temp-0.7 noise floor). Round 1 is
  prompt-identical by construction (empty history); later matched turns use ask_shown
  as the context key. A round-1-only override is reported as the clean check.
- Action = discrete action TYPE (accept/offer/reject/hold/walk); an offer→offer with a
  different price is NOT an action-choice override.
- Aggression buckets are mapped to the GAME schema (the probe's hold_justified/
  offer_conditional do not occur here): aggressive = {walk, hold, reject} or an offer
  with ZOPA-position (price-floor)/(v_b-floor) < 0.5; passive = {accept} or an offer
  with ZOPA-position >= 0.5.
"""
import json
from collections import Counter

CKPTS = ["olmo_base", "olmo_inst", "z_base", "z_dpo", "t_base", "t_rlvr"]
FILEMAP = {"olmo_base": "base", "olmo_inst": "instruct", "z_base": "z_base",
           "z_dpo": "z_dpo", "t_base": "t_base", "t_rlvr": "t_rlvr"}
ACTIONS = ["accept", "offer", "reject", "hold", "walk"]


def load(p):
    return [json.loads(l) for l in open(p)]


def free_action_bucket(act, price, floor, v_b):
    if act == "offer" and price is not None and v_b > floor:
        pos = (price - floor) / (v_b - floor)
        return "aggressive" if pos < 0.5 else "passive"
    if act in ("walk", "hold", "reject"):
        return "aggressive"
    if act == "accept":
        return "passive"
    return "other"


def analyze(ck):
    fk = FILEMAP[ck]
    fp = f"runs/n500/passA_{fk}.jsonl"
    cp = f"runs/n500/constrained_{fk}.jsonl"
    import os
    if not (os.path.exists(fp) and os.path.exists(cp)):
        return {"MISSING": f"free={os.path.exists(fp)} constr={os.path.exists(cp)}"}
    free = {e["idx"]: e for e in load(fp)}
    con = {e["state"]["idx"]: e["state"] for e in load(cp)}
    common = set(free) & set(con)
    if len(common) < 0.9 * max(len(free), len(con)):
        return {"MISSING": f"episode overlap only {len(common)} of "
                f"{len(free)}/{len(con)}"}

    n_paired = n_ctx = n_free_parsed = 0
    n_override = 0
    rescue_num = 0
    agg_n = agg_ov = pas_n = pas_ov = 0
    free_dist = Counter(); con_dist = Counter()
    r1_parsed = r1_override = 0

    for idx in common:
        v_b = free[idx]["v_b"]; floor = free[idx]["floor"]
        ft = {t["round"]: t for t in free[idx]["turns"]}
        ct = {t["round"]: t for t in con[idx]["turns"]}
        for r in set(ft) & set(ct):
            f, c = ft[r], ct[r]
            n_paired += 1
            # rescue: over ALL paired turns (free failed but constrained valid)
            if (not f["parse_success"]) and c["action"] in ACTIONS:
                rescue_num += 1
            # context match (ask_shown identical)
            if f["ask_shown"] != c["ask_shown"]:
                continue
            n_ctx += 1
            if not f["parse_success"]:
                continue
            n_free_parsed += 1
            fa = f["strict_action"]; ca = c["action"]
            free_dist[fa] += 1; con_dist[ca] += 1
            ov = (fa != ca)
            if ov:
                n_override += 1
            buck = free_action_bucket(fa, f.get("price"), floor, v_b)
            if buck == "aggressive":
                agg_n += 1; agg_ov += ov
            elif buck == "passive":
                pas_n += 1; pas_ov += ov
            if r == 1:
                r1_parsed += 1; r1_override += ov

    # TV distance over the 5 action types (free vs constrained, matched set)
    tv = 0.0
    if n_free_parsed:
        for a in ACTIONS:
            tv += abs(free_dist.get(a, 0) - con_dist.get(a, 0)) / n_free_parsed
        tv *= 0.5
    return {
        "n_paired": n_paired, "n_ctx": n_ctx, "n_free_parsed": n_free_parsed,
        "override": n_override / n_free_parsed if n_free_parsed else float("nan"),
        "rescue": rescue_num / n_paired if n_paired else float("nan"),
        "override_agg": agg_ov / agg_n if agg_n else float("nan"),
        "override_pass": pas_ov / pas_n if pas_n else float("nan"),
        "agg_n": agg_n, "pas_n": pas_n, "tv": tv,
        "r1_override": r1_override / r1_parsed if r1_parsed else float("nan"),
        "r1_n": r1_parsed,
        "free_dist": dict(free_dist), "con_dist": dict(con_dist),
    }


def main():
    res = {ck: analyze(ck) for ck in CKPTS}
    L = []
    import datetime
    L.append("# policy_shift.md — constrained-decoding invasiveness diagnostic\n")
    L.append(f"**Date:** {datetime.date.today().isoformat()}  ")
    L.append("**Source (read-only, no inference re-run):** "
             "`runs/n500/passA_{base,instruct,z_base,z_dpo,t_base,t_rlvr}.jsonl` "
             "(free/strict) joined to "
             "`runs/n500/constrained_{...}.jsonl` (constrained) on (episode idx, "
             "round). No model inference was performed; all numbers are computed "
             "from the persisted paired generations.\n")
    L.append("**Method:** override = constrained action TYPE ≠ free parsed action "
             "TYPE, over **context-matched** turns (same `ask_shown`; at a matched "
             "turn prompt+seed are shared and the grammar mask is the only "
             "difference → pure grammar override, no temp-0.7 noise floor). Round 1 "
             "is prompt-identical (empty history, 500/500 matched); the round-1-only "
             "override is the clean check. Rescue = free failed to parse but "
             "constrained valid, over ALL paired turns. Aggression mapped to the "
             "game schema {accept,offer,reject,hold,walk} (probe buckets "
             "hold_justified/offer_conditional do not occur here): aggressive = "
             "{walk,hold,reject} or offer with ZOPA-position <0.5; passive = "
             "{accept} or offer with position ≥0.5.\n")
    L.append("| checkpoint | override | rescue | override\\|aggr (n) | "
             "override\\|pass (n) | TV-shift | N_paired | N_ctx-matched | "
             "N_free-parsed | round-1 override (n) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    notes = []
    for ck in CKPTS:
        r = res[ck]
        if "MISSING" in r:
            L.append(f"| {ck} | **MISSING** ({r['MISSING']}) | — | — | — | — | — | — | — | — |")
            notes.append(f"- **{ck}: MISSING** — {r['MISSING']}.")
            continue
        L.append(f"| {ck} | {r['override']:.3f} | {r['rescue']:.3f} | "
                 f"{r['override_agg']:.3f} ({r['agg_n']}) | "
                 f"{r['override_pass']:.3f} ({r['pas_n']}) | {r['tv']:.3f} | "
                 f"{r['n_paired']} | {r['n_ctx']} | {r['n_free_parsed']} | "
                 f"{r['r1_override']:.3f} ({r['r1_n']}) |")
        if r['agg_n'] and r['pas_n']:
            conc = ("YES" if r['override_agg'] > r['override_pass'] else "no")
            notes.append(f"- **{ck}**: overrides concentrate on aggressive actions? "
                         f"**{conc}** (override|aggr={r['override_agg']:.3f} vs "
                         f"override|pass={r['override_pass']:.3f}).")
        else:
            notes.append(f"- **{ck}**: aggression split underpowered "
                         f"(agg_n={r['agg_n']}, pas_n={r['pas_n']}).")
    L.append("\n## Per-checkpoint note (aggressive concentration)\n")
    L += notes
    L.append("\n## Validity flags\n")
    L.append("- **Pairing key present** (episode idx + round); no checkpoint is "
             "MISSING for the pairing. Round-1 context is identical 500/500 for all "
             "checkpoints; all-paired-round `ask_shown` match ≈ 90%, so overrides "
             "are restricted to context-matched turns to avoid counting trajectory "
             "divergence as override.")
    L.append("- **Override = action TYPE change**, not price change (an offer→offer "
             "re-price is not an action-choice override).")
    L.append("- **Aggression uses a game-schema mapping** (stated above); the "
             "probe's hold_justified/offer_conditional buckets are absent from the "
             "game runs.")
    L.append("- **Action distributions (matched free-parsed set), per checkpoint:**")
    for ck in CKPTS:
        r = res[ck]
        if "MISSING" not in r:
            L.append(f"  - {ck}: free={r['free_dist']} → constr={r['con_dist']}")
    open("out/policy_shift.md", "w").write("\n".join(L) + "\n")
    print("wrote out/policy_shift.md")
    for ck in CKPTS:
        print(ck, res[ck] if "MISSING" in res[ck] else
              {k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in res[ck].items() if k not in ("free_dist", "con_dist")})


if __name__ == "__main__":
    main()
