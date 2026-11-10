"""T6 — Sign-consistency across turn sets.
Forced-horizon (k=4) Test-4 caving slope beta against adaptive seller D, for the
three base models x 2 operationalizations x 3 turn sets:
  (i)   all constrained turns
  (ii)  constrained EXCLUDING overridden offer-turns  (paper-reported)
  (iii) free natively-parsed turns                     (Defect-B selection flagged)
Cross-check the published override-excluded values:
  OLMo(base) opa +0.175 opb -0.976 ; Mistral(z_base) +0.148/-0.893 ;
  Llama(t_base) +0.073/-0.609.
Verdict: sign-consistent across the three sets, per model x op.
Reuses b_analyze verbatim (overridden_rounds/per_ep_all/per_ep_excl/betaCI).
No model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import b_analyze as BA
import _prov, _emit

MODELS = [("base", "OLMo"), ("z_base", "Mistral/Zephyr"), ("t_base", "Llama/Tulu")]
OPS = {"opa_hold_rate": 1, "opb_conc_granted": 2}
PUBLISHED = {"base": {"opa_hold_rate": 0.175, "opb_conc_granted": -0.976},
             "z_base": {"opa_hold_rate": 0.148, "opb_conc_granted": -0.893},
             "t_base": {"opa_hold_rate": 0.073, "opb_conc_granted": -0.609}}


def run():
    out = {}
    src = {"source_script": "analysis/revision/t6_sign_consistency.py",
           "source_data": "results/forced_horizon/D_*.jsonl, "
                          "results/base_artifact/free_D_*.jsonl",
           "seed": 1234, "repo_fingerprint": _prov.FINGERPRINT}
    verdicts = {}
    print("T6 — forced-horizon beta(D) across three turn sets:")
    for ck, name in MODELS:
        cs = BA.load_states(f"{BA.CON}/D_{ck}.jsonl")
        fs = BA.load_states(f"{BA.FREE}/free_D_{ck}.jsonl")
        ovr, _st = BA.overridden_rounds(fs, cs)
        pe = {"i_all_constrained": BA.per_ep_all(cs),
              "ii_excl_override": BA.per_ep_excl(cs, ovr),
              "iii_free_parsed": BA.per_ep_all(fs)}
        for opname, hidx in OPS.items():
            slopes = {}
            for tsname, p in pe.items():
                b, lo, hi, nep, nob, nn = BA.betaCI(p, hidx)
                slopes[tsname] = {"slope": b, "ci": [lo, hi], "n_ep": nep,
                                  "n_obs": nob, "nonnull": nn}
                out[f"sign_consistency.{ck}.{opname}.{tsname}"] = {
                    "value": {"slope": b}, "ci": [lo, hi], "n": nob, **src,
                    "notes": f"{name} forced-horizon beta(D) {tsname}"}
            # sign consistency across the three sets (ignore NaN sets)
            signs = [(1 if s["slope"] > 0 else -1 if s["slope"] < 0 else 0)
                     for s in slopes.values() if s["slope"] == s["slope"]]
            consistent = len(set(signs)) == 1 and len(signs) >= 2
            # cross-check published (ii)
            pub = PUBLISHED[ck][opname]
            got = slopes["ii_excl_override"]["slope"]
            xcheck = (got == got and abs(got - pub) <= 0.02)
            verdicts[f"{ck}.{opname}"] = {"consistent": consistent,
                "signs": signs, "published_ii": pub, "got_ii": got,
                "xcheck_ok": xcheck}
            print(f"  {name:14} {opname:16} "
                  f"(i)={slopes['i_all_constrained']['slope']!r} "
                  f"(ii)={got!r} (iii)={slopes['iii_free_parsed']['slope']!r} "
                  f"consistent={consistent} xcheck(pub={pub})={xcheck}")
    all_consistent = all(v["consistent"] for v in verdicts.values())
    out["sign_consistency.verdicts"] = {"value": verdicts,
        "all_consistent": all_consistent, **src,
        "notes": "turn set (iii) free-parsed carries Defect-B selection bias"}
    _emit.merge(out)
    cur = _emit.load_all()
    cur.setdefault("verdicts", {})["sign_consistency"] = (
        "consistent" if all_consistent else "mixed")
    json.dump(cur, open(_emit.OUT, "w"), indent=1)
    print("all sign-consistent:", all_consistent)
    return out


if __name__ == "__main__":
    run()
