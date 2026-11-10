"""T9 — Lenient-recovery counts.
Per checkpoint in the four-condition (passA free) runs: number of turns where
STRICT parsing forfeited but the LENIENT re-scorer recovered a valid buyer
action. Totals per family. Also strict vs lenient D_fam to 4 decimals per family
base->final so the sub-display-precision gap is visible.

Reuses the frozen pipeline: study.cond_records (strict/lenient replay via
freepass) and study.dfam. No model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import game_lib as G
import study as ST
import _prov, _emit

# passA (free-decode) availability = the four-condition runs
PASSA = {"OLMo": ["base", "sft", "dpo", "instruct"],
         "Zephyr": ["z_base", "z_sft", "z_dpo"],
         "Tulu": ["t_base", "t_rlvr"]}
FINAL = {"OLMo": ("base", "instruct"), "Zephyr": ("z_base", "z_dpo"),
         "Tulu": ("t_base", "t_rlvr")}


def recover_counts(ckpt):
    """(turns where strict forfeited but lenient recovered a valid buyer action,
    total turns, strict-fail turns)."""
    rows = [json.loads(l) for l in open(f"runs/n500/passA_{ckpt}.jsonl")]
    recovered = total = strict_fail = 0
    for r in rows:
        for t in r["turns"]:
            total += 1
            strict_ok = t["strict_action"] in G.BUYER_ACTIONS
            lenient_ok = t["lenient_action"] in G.BUYER_ACTIONS
            if not t["parse_success"]:
                strict_fail += 1
            # strict forfeited (not a creditable buyer action) but lenient valid
            if (not strict_ok) and lenient_ok:
                recovered += 1
    return recovered, total, strict_fail


def run():
    out = {}
    per_ckpt = {}
    fam_tot = {}
    for fam, cks in PASSA.items():
        tot = 0
        for ck in cks:
            rec, total, sf = recover_counts(ck)
            per_ckpt[ck] = {"recovered": rec, "turns": total, "strict_fail": sf}
            tot += rec
        fam_tot[fam] = tot

    # strict vs lenient D_fam base->final, 4 decimals
    dfam = {}
    for fam, (b, a) in FINAL.items():
        mask = ST.matched_mask(b, a)
        bs = ST.cond_records(b, "strict", mask)  # mask unused for strict/lenient
        rs = {}
        for cond in ("strict", "lenient"):
            rb = ST.cond_records(b, cond)
            ra = ST.cond_records(a, cond)
            r = ST.dfam(rb, ra, cond)
            rs[cond] = r["D"]
        dfam[fam] = {"strict": rs["strict"], "lenient": rs["lenient"],
                     "gap": rs["lenient"] - rs["strict"]}

    total_recovered = sum(fam_tot.values())
    verdict_lenient = "vacuous" if total_recovered <= 5 else "substantial"

    stamp = _prov.stamp("analysis/revision/t9_lenient.py",
                        ["runs/n500/passA_*.jsonl"],
                        notes="strict-forfeit-but-lenient-recovered buyer-action turns")
    out["pnLenientCounts"] = {"value": {"per_checkpoint": per_ckpt,
                                        "per_family": fam_tot,
                                        "total": total_recovered},
                             **stamp}
    for fam, d in dfam.items():
        out[f"lenient.dfam.{fam}"] = {"value": d,
            "source_script": "analysis/revision/t9_lenient.py",
            "source_data": f"runs/n500/passA_{FINAL[fam][0]},{FINAL[fam][1]}.jsonl",
            "seed": 1234, "repo_fingerprint": _prov.FINGERPRINT,
            "notes": "strict vs lenient base->final D_fam (N=500), 4-dp"}
    _emit.merge(out)   # NOTE: do NOT put a 'verdicts' key in `out` — merge() does
    # cur.update(out) and would clobber other tasks' verdicts. Set it separately.
    cur = _emit.load_all()
    cur.setdefault("verdicts", {})["lenient"] = verdict_lenient
    json.dump(cur, open(_emit.OUT, "w"), indent=1)
    print("T9 lenient: total recovered =", total_recovered, "verdict =", verdict_lenient)
    print("  per family:", fam_tot)
    for fam, d in dfam.items():
        print(f"  {fam}: strict D={d['strict']:.4f} lenient D={d['lenient']:.4f} "
              f"gap={d['gap']:+.4f}")
    return out


if __name__ == "__main__":
    run()
