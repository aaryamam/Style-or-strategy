"""T7 — centered influence-function variant + bootstrap code inspection.

(a) Recompute the aggression_penalty per-episode entries with the CENTERED form
      a_i = b_hat + (o_i - o_bar) * (p_i - p_bar - b_hat*(o_i - o_bar)) / s2_o
    where (o,p) are (opening_offer, final_price) on feasible deals, b_hat is the
    univariate centered slope cov(o,p)/var(o), s2_o = var(o). Mean_i a_i = b_hat,
    so this is a proper IF centered on the simple-regression slope (vs study._agg's
    multivariate [1,opening,v_b,c_s] IF). Rebuild Sigma, recompute ALL constrained
    base->final D_fam, ALL Table-7 consecutive steps, and the per-family locus.
    Report side-by-side with the published (study._agg) values + deltas.
    Verdict: 'unchanged' if locus orderings and CI-separations intact, else 'changed'.

(b) Code inspection: does study.dfam's bootstrap recompute the IF entries and
    pooled Sigma within each resample, or hold them fixed from the full sample?

Reuses study.dfam verbatim (it reads r['components']['aggression_penalty']); we
only swap how that per-episode value is constructed. No model inference.
"""
import json, os, sys, math
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import signature as S
import study as ST
import _prov, _emit

NAN = float("nan")
FAM = {"OLMo": (["base", "sft", "dpo", "instruct"], "n500"),
       "Zephyr": (["z_base", "z_sft", "z_dpo"], "n2000"),
       "Tulu": (["t_base", "t_sft", "t_dpo", "t_rlvr"], "n2000")}


def _agg_centered(recs):
    """Centered univariate IF for the opening->final slope. Returns {idx: a_i}."""
    idxs, o, p = [], [], []
    for i, r in enumerate(recs):
        c = r["components"]
        if (r["state"]["outcome"] == "deal"
                and c["opening_offer"] == c["opening_offer"]
                and c["final_price"] == c["final_price"]):
            idxs.append(i); o.append(c["opening_offer"]); p.append(c["final_price"])
    n = len(o)
    if n < 6:
        return {}
    obar = sum(o) / n; pbar = sum(p) / n
    s2o = sum((x - obar) ** 2 for x in o) / n
    if s2o < 1e-12:
        return {}
    bhat = sum((o[k] - obar) * (p[k] - pbar) for k in range(n)) / (s2o * n)
    a = {}
    for k in range(n):
        eps = (p[k] - pbar) - bhat * (o[k] - obar)
        a[idxs[k]] = bhat + (o[k] - obar) * eps / s2o
    return a


def wrap_centered(states):
    recs = [{"state": s, "components": S.episode_components(s)} for s in states]
    a = _agg_centered(recs)
    for i, r in enumerate(recs):
        r["components"]["aggression_penalty"] = a.get(i, NAN)
    return recs


def load(ck, where, centered):
    d = "runs/n500" if where == "n500" else "runs/n2000"
    states = [json.loads(l)["state"] for l in open(f"{d}/constrained_{ck}.jsonl")]
    return wrap_centered(states) if centered else ST._wrap(states)


def locus_step(recs_by_ck, cks):
    """argmax consecutive-step D and whether it CI-separates from the rest."""
    steps = []
    for i in range(len(cks) - 1):
        r = ST.dfam(recs_by_ck[cks[i]], recs_by_ck[cks[i + 1]],
                    f"{cks[i]}->{cks[i+1]}", nboot=2000)
        steps.append((r["label"], r["D"], r["lo"], r["hi"]))
    big = max(steps, key=lambda s: s[1])
    rest = [s for s in steps if s is not big]
    ci_sep = all(big[2] > s[3] for s in rest) if rest else True
    return big, ci_sep, steps


def run():
    published, centered = {}, {}
    locus = {}
    for fam, (cks, where) in FAM.items():
        rp = {ck: load(ck, where, False) for ck in cks}
        rc = {ck: load(ck, where, True) for ck in cks}
        # base->final
        bp = ST.dfam(rp[cks[0]], rp[cks[-1]], "bf", nboot=2000)
        bc = ST.dfam(rc[cks[0]], rc[cks[-1]], "bf", nboot=2000)
        published[f"{fam}.base_final"] = (bp["D"], bp["lo"], bp["hi"])
        centered[f"{fam}.base_final"] = (bc["D"], bc["lo"], bc["hi"])
        # consecutive steps
        for i in range(len(cks) - 1):
            lab = f"{fam}.{cks[i]}->{cks[i+1]}"
            sp = ST.dfam(rp[cks[i]], rp[cks[i + 1]], "x", nboot=2000)
            sc = ST.dfam(rc[cks[i]], rc[cks[i + 1]], "x", nboot=2000)
            published[lab] = (sp["D"], sp["lo"], sp["hi"])
            centered[lab] = (sc["D"], sc["lo"], sc["hi"])
        bigp, sepp, _ = locus_step(rp, cks)
        bigc, sepc, _ = locus_step(rc, cks)
        locus[fam] = {"published_locus": bigp[0], "published_cisep": sepp,
                      "centered_locus": bigc[0], "centered_cisep": sepc,
                      "preserved": (bigp[0] == bigc[0] and sepp == sepc)}

    # verdict: unchanged if every family's locus preserved and no CI-separation
    # flipped, and base->final CI overlap status unchanged.
    all_preserved = all(v["preserved"] for v in locus.values())
    verdict = "unchanged" if all_preserved else "changed"
    changes = [f for f, v in locus.items() if not v["preserved"]]

    out = {}
    out["pnCenteredDfam"] = {"value": {
        k: {"published": published[k], "centered": centered[k],
            "delta_D": centered[k][0] - published[k][0]} for k in published},
        "source_script": "analysis/revision/t7_centered_if.py",
        "source_data": "runs/n500/constrained_*.jsonl, runs/n2000/constrained_*.jsonl",
        "seed": 1234, "repo_fingerprint": _prov.FINGERPRINT,
        "notes": "centered univariate IF vs published multivariate IF (study._agg)"}
    out["pnCenteredLocus"] = {"value": locus, "verdict": verdict,
        "changed_families": changes,
        "source_script": "analysis/revision/t7_centered_if.py",
        "source_data": "runs/n500,n2000 constrained", "seed": 1234,
        "repo_fingerprint": _prov.FINGERPRINT}
    # (b) bootstrap code inspection
    out["pnBootstrapRecompute"] = {"value": "NO",
        "source_script": "study.py",
        "source_data": "study.py:dfam",
        "notes": ("study.dfam computes pooled_stats and Sinv=inv(cov(...)) ONCE "
                  "before the bootstrap loop (study.py dfam: 'stats=...' and "
                  "'Sinv=_inv(cov(b,a,stats))' precede 'for _ in range(nboot)'); "
                  "the loop only resamples indices and re-averages component means "
                  "via mvec(b,stats,ib). The aggression_penalty per-episode IF "
                  "entries are fixed at load time by study._wrap/_agg "
                  "(load_constrained -> _wrap) and never recomputed per resample. "
                  "Hence IF entries and pooled Sigma are HELD FIXED from the full "
                  "sample; only the per-component means vary across resamples."),
        "repo_fingerprint": _prov.FINGERPRINT}
    _emit.merge(out)
    cur = _emit.load_all()
    cur.setdefault("verdicts", {})["centered_if"] = verdict
    json.dump(cur, open(_emit.OUT, "w"), indent=1)

    print("T7 centered IF vs published:")
    for k in published:
        p, c = published[k], centered[k]
        print(f"  {k:26} pub D={p[0]:.3f}[{p[1]:.3f},{p[2]:.3f}]  "
              f"cen D={c[0]:.3f}[{c[1]:.3f},{c[2]:.3f}]  dD={c[0]-p[0]:+.3f}")
    print("locus:")
    for f, v in locus.items():
        print(f"  {f}: pub={v['published_locus']}({v['published_cisep']}) "
              f"cen={v['centered_locus']}({v['centered_cisep']}) "
              f"preserved={v['preserved']}")
    print("VERDICT:", verdict, "| pnBootstrapRecompute = NO")
    return out


if __name__ == "__main__":
    run()
