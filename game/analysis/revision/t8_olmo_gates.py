"""T8 — OLMo admissibility gates, retrospective.
Runs the SAME numeric gates applied to Zephyr/Tulu in section 6 (zephyr_gates.py /
tulu_gates.py) on OLMo's constrained base + final episodes:
  beta  : aggression_penalty (influence-fold OLS coef) per checkpoint + the
          rejected-opening construct check (fraction of rejected-opening deals
          resolving BELOW the accepted-group surplus mean).
  var   : per-component variance at each checkpoint (non-degeneracy).
  walk  : walkaway base->final standardized gap + raw rates (non-nullity).
All outputs labelled 'retrospective'. Reuses study._wrap/_agg/pooled_stats/mvec/cov
and the zephyr_gates.g1_construct logic verbatim. No model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import study as ST
import zephyr_gates as ZG   # reuse g1_construct + variance
import _prov, _emit

FIVE = ST.FIVE


def run():
    ob = ST.load_constrained("base")
    oi = ST.load_constrained("instruct")

    # --- beta: aggression_penalty + rejected-opening construct ---
    _, beta_base = ST._agg(ob)
    _, beta_final = ST._agg(oi)
    am, below, nrej, nacc = ZG.g1_construct(oi)
    frac_worse = below / nrej if nrej else float("nan")

    # --- variance per component per checkpoint ---
    var = {}
    for c in FIVE:
        vb, nb = ZG.variance([r["components"][c] for r in ob])
        vd, nd = ZG.variance([r["components"][c] for r in oi])
        var[c] = {"base": vb, "final": vd,
                  "flag": "NEAR-CONSTANT" if (min(vb, vd) < 0.005) else "ok"}

    # --- walkaway non-nullity ---
    stats = {c: ST.pooled_stats(ob, oi, c) for c in FIVE}
    vbv = ST.mvec(ob, stats); viv = ST.mvec(oi, stats)
    wi = FIVE.index("walkaway")
    walk_gap = viv[wi] - vbv[wi]

    def raw(recs, c):
        vals = [r["components"][c] for r in recs
                if r["components"][c] == r["components"][c]]
        return sum(vals) / len(vals) if vals else float("nan")
    walk_base, walk_final = raw(ob, "walkaway"), raw(oi, "walkaway")

    out = {}
    S = lambda: {"source_script": "analysis/revision/t8_olmo_gates.py",
                 "source_data": "runs/n500/constrained_{base,instruct}.jsonl",
                 "repo_fingerprint": _prov.FINGERPRINT, "notes": "retrospective"}
    out["pnOLMoBeta"] = {"value": {"beta_base": beta_base, "beta_final": beta_final,
                                   "construct_accepted_mean_surpEff": am,
                                   "rejected_opening_below_frac": frac_worse,
                                   "n_rejected": nrej, "n_accepted": nacc,
                                   "label": "retrospective"}, **S()}
    out["pnOLMoVar"] = {"value": {c: var[c] for c in FIVE}, "label": "retrospective",
                        **S()}
    out["pnOLMoWalk"] = {"value": {"std_gap_final_minus_base": walk_gap,
                                   "raw_base": walk_base, "raw_final": walk_final,
                                   "nonnull": abs(walk_gap) > 0.15,
                                   "label": "retrospective"}, **S()}
    _emit.merge(out)
    print("T8 OLMo gates (retrospective):")
    print(f"  beta: base={beta_base:+.3f} final={beta_final:+.3f}; "
          f"rejected-opening worse frac={frac_worse:.2f} ({below}/{nrej})")
    print("  variance:")
    for c in FIVE:
        print(f"    {c:24} base={var[c]['base']:.4f} final={var[c]['final']:.4f} "
              f"{var[c]['flag']}")
    print(f"  walkaway: std gap={walk_gap:+.3f} raw {walk_base:.3f}->{walk_final:.3f} "
          f"nonnull={abs(walk_gap)>0.15}")
    return out


if __name__ == "__main__":
    run()
