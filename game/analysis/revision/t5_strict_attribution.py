"""T5 — Strict-regime stage attribution.
Wherever free-decode (passA) data exist for BOTH endpoints of a step, compute
consecutive-step D_fam under STRICT scoring (freepass strict replay) with per-step
pooled Sigma and bootstrap CIs (same B=2000, seed=1234 as the published
constrained attribution). Then run the locus (argmax-step + CI-separation) test.

Coverage (passA availability, N=500):
  OLMo   : base,sft,dpo,instruct  -> full chain (3 steps)
  Zephyr : z_base,z_sft,z_dpo     -> full chain (2 steps)
  Tulu   : t_base,t_rlvr          -> base->final only (t_sft/t_dpo passA absent)

Verdict per family (pnStrictLocusVerdict): locus ordering preserved vs Table 7.
Reuses study.cond_records('strict') + study.dfam. No model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import study as ST
import _prov, _emit

# constrained (Table 7) loci for comparison
CONSTRAINED_LOCUS = {"OLMo": "base->sft", "Zephyr": "z_sft->z_dpo",
                     "Tulu": "t_sft->t_dpo"}
FAM = {"OLMo": ["base", "sft", "dpo", "instruct"],
       "Zephyr": ["z_base", "z_sft", "z_dpo"],
       "Tulu": ["t_base", "t_rlvr"]}   # only base->final computable under strict


def run():
    recs = {}
    results = {}
    for fam, cks in FAM.items():
        for ck in cks:
            if ck not in recs:
                recs[ck] = ST.cond_records(ck, "strict")
        steps = []
        for i in range(len(cks) - 1):
            r = ST.dfam(recs[cks[i]], recs[cks[i + 1]], f"{cks[i]}->{cks[i+1]}",
                        nboot=2000)
            steps.append({"step": r["label"], "d": r["D"], "lo": r["lo"],
                          "hi": r["hi"]})
        results[fam] = steps

    # locus test per family with >=2 steps
    verdict = {}
    for fam, steps in results.items():
        if len(steps) < 2:
            verdict[fam] = {"computable": False,
                            "reason": "only base->final under strict (intermediate "
                                      "passA absent)"}
            continue
        big = max(steps, key=lambda s: s["d"])
        rest = [s for s in steps if s is not big]
        ci_sep = all(big["lo"] > s["hi"] for s in rest)
        preserved = (big["step"] == CONSTRAINED_LOCUS[fam])
        verdict[fam] = {"computable": True, "strict_locus": big["step"],
                        "constrained_locus": CONSTRAINED_LOCUS[fam],
                        "ci_separated": ci_sep,
                        "preserved": preserved,
                        "ordering": "preserved" if preserved else "not_preserved"}

    out = {}
    src = {"source_script": "analysis/revision/t5_strict_attribution.py",
           "source_data": "runs/n500/passA_*.jsonl (strict replay via freepass)",
           "seed": 1234, "repo_fingerprint": _prov.FINGERPRINT}
    for fam, steps in results.items():
        for s in steps:
            out[f"strict_locus.{fam}.{s['step']}"] = {
                "value": {"d": s["d"]}, "ci": [s["lo"], s["hi"]],
                "n": len(recs[FAM[fam][0]]), **src,
                "notes": "strict-scoring consecutive-step D_fam, N=500"}
    out["pnStrictLocusVerdict"] = {"value": verdict, **src}
    _emit.merge(out)
    cur = _emit.load_all()
    v = cur.setdefault("verdicts", {})
    for fam in verdict:
        v[f"strict_locus.{fam}"] = (verdict[fam].get("ordering")
                                    or "not_computable")
    json.dump(cur, open(_emit.OUT, "w"), indent=1)

    print("T5 strict-regime attribution (N=500):")
    for fam, steps in results.items():
        print(f"  {fam}:")
        for s in steps:
            print(f"    {s['step']:20} D={s['d']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]")
        print(f"    verdict: {verdict[fam]}")
    return out


if __name__ == "__main__":
    run()
