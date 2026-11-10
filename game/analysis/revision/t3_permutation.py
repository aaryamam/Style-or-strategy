"""T3 — Paired permutation nulls + empirical floors (numpy-vectorized).

For a pair of checkpoints on the SHARED episode grid (constrained), permute by
swapping the two checkpoints' per-episode signatures within each paired episode
with prob 1/2; recompute the FULL D_fam pipeline per permutation:
  - per-checkpoint condition means (mvec),
  - the aggression_penalty influence-function entries (recomputed per group; the
    SAME multivariate IF as study._agg),
  - pooled Sigma (study.cov, ridge 0.05).
B = 5000, fixed seed. p = fraction of permuted D_hat >= observed.

The per-permutation D is a numpy re-implementation of study's pipeline (verified
to reproduce study.dfam's point D on the unpermuted split to 1e-9). Runs for every
Table-3 base->final constrained distance and every consecutive Table-7 step.
Reports p, null median/mean/95th pct, and the unpaired approx sqrt(2k/N) (k=5).
No model inference.
"""
import json, os, sys, math
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import signature as S
import study as ST
import _prov, _emit

SEED = 1234
B = 5000
RIDGE = 0.05
BASE4 = ["opening_aggressiveness", "surplus_efficiency", "haggling", "walkaway"]
NAN = float("nan")


def precompute(states):
    """Return arrays: C4 (N,4) non-aggression comps (NaN allowed); deal mask,
    opening, v_b, c_s, final (N,) for the aggression IF."""
    n = len(states)
    C4 = np.full((n, 4), np.nan)
    deal = np.zeros(n, bool)
    opening = np.full(n, np.nan); final = np.full(n, np.nan)
    vb = np.zeros(n); cs = np.zeros(n)
    for i, s in enumerate(states):
        c = S.episode_components(s)
        for j, name in enumerate(BASE4):
            v = c[name]
            C4[i, j] = v if v == v else np.nan
        vb[i] = s["v_b"]; cs[i] = s["c_s"]
        if (s["outcome"] == "deal" and c["opening_offer"] == c["opening_offer"]
                and c["final_price"] == c["final_price"]):
            deal[i] = True
            opening[i] = c["opening_offer"]; final[i] = c["final_price"]
    return dict(C4=C4, deal=deal, opening=opening, final=final, vb=vb, cs=cs)


def agg_if(group):
    """Multivariate aggression_penalty IF per episode (NaN where not a deal).
    Mirrors study._agg: X=[1,opening,vb,cs], y=final on deals; a_k = beta1 +
    (inv(X'X/n)[1] . X_k) * resid_k."""
    n = len(group["deal"])
    a = np.full(n, np.nan)
    m = group["deal"]
    nd = int(m.sum())
    if nd < 6:
        return a
    X = np.column_stack([np.ones(nd), group["opening"][m], group["vb"][m],
                         group["cs"][m]])
    y = group["final"][m]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    XtXn = (X.T @ X) / nd
    try:
        inv4 = np.linalg.inv(XtXn)
    except np.linalg.LinAlgError:
        inv4 = np.linalg.pinv(XtXn)
    e = y - X @ beta
    a[m] = beta[1] + (X @ inv4[1]) * e
    return a


def comp5(group):
    """(N,5) per-episode component matrix incl. aggression_penalty (NaN allowed)."""
    a = agg_if(group)
    return np.column_stack([group["C4"], a])


def dfam_point_np(gb, ga):
    """numpy re-impl of study.dfam point D: pooled standardize, cov (ridge 0.05,
    pairwise-complete), Mahalanobis of the mean-vector difference."""
    Zb = comp5(gb); Za = comp5(ga)               # (N,5) each
    pooled = np.vstack([Zb, Za])                  # (2N,5)
    # pooled mean/sd per column (population, NaN-dropped)
    mean = np.nanmean(pooled, axis=0)
    sd = np.sqrt(np.nanmean((pooled - mean) ** 2, axis=0))
    sd = np.where(sd > 1e-9, sd, 1.0)
    Std = (pooled - mean) / sd                    # (2N,5) standardized
    # per-model mean vectors (standardized)
    vb = (np.nanmean(Zb, axis=0) - mean) / sd
    va = (np.nanmean(Za, axis=0) - mean) / sd
    # covariance: pairwise-complete, population, ridge on diagonal
    k = 5
    Sig = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            mask = ~np.isnan(Std[:, i]) & ~np.isnan(Std[:, j])
            if mask.sum() < 5:
                Sig[i, j] = 1.0 if i == j else 0.0
                continue
            xi = Std[mask, i]; xj = Std[mask, j]
            mi = xi.mean(); mj = xj.mean()
            Sig[i, j] = ((xi - mi) * (xj - mj)).mean()
    Sig[np.diag_indices(k)] += RIDGE
    Sinv = np.linalg.inv(Sig)
    d = vb - va
    return math.sqrt(max(0.0, float(d @ Sinv @ d)))


def load_states(ck, where):
    d = "runs/n500" if where == "n500" else "runs/n2000"
    return [json.loads(l)["state"] for l in open(f"{d}/constrained_{ck}.jsonl")]


def perm_null(pb, pa, observed, b=B, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(pb["deal"])
    keys = ["C4", "deal", "opening", "final", "vb", "cs"]
    nulls = np.empty(b)
    for t in range(b):
        swap = rng.rand(n) < 0.5
        gb, ga = {}, {}
        for key in keys:
            Bv, Av = pb[key], pa[key]
            if Bv.ndim == 2:
                sw = swap[:, None]
            else:
                sw = swap
            gb[key] = np.where(sw, Av, Bv)
            ga[key] = np.where(sw, Bv, Av)
        nulls[t] = dfam_point_np(gb, ga)
    nulls.sort()
    p = float((nulls >= observed).mean())
    return p, nulls


COMPARISONS = [
    ("OLMo", "base->instruct", "base", "instruct", "n500", "base_final"),
    ("Zephyr", "z_base->z_dpo", "z_base", "z_dpo", "n500", "base_final"),
    ("Tulu", "t_base->t_rlvr", "t_base", "t_rlvr", "n500", "base_final"),
    ("OLMo", "base->sft", "base", "sft", "n500", "step"),
    ("OLMo", "sft->dpo", "sft", "dpo", "n500", "step"),
    ("OLMo", "dpo->instruct", "dpo", "instruct", "n500", "step"),
    ("Zephyr", "z_base->z_sft", "z_base", "z_sft", "n2000", "step"),
    ("Zephyr", "z_sft->z_dpo", "z_sft", "z_dpo", "n2000", "step"),
    ("Tulu", "t_base->t_sft", "t_base", "t_sft", "n2000", "step"),
    ("Tulu", "t_sft->t_dpo", "t_sft", "t_dpo", "n2000", "step"),
    ("Tulu", "t_dpo->t_rlvr", "t_dpo", "t_rlvr", "n2000", "step"),
]


def run():
    cache = {}
    results = {}
    # sanity: verify numpy point D reproduces study.dfam on an unpermuted split
    sb = load_states("base", "n500"); si = load_states("instruct", "n500")
    pb = precompute(sb); pi = precompute(si)
    d_np = dfam_point_np(pb, pi)
    d_st = ST.dfam(ST._wrap(sb), ST._wrap(si), "chk", nboot=1)["D"]
    assert abs(d_np - d_st) < 1e-6, f"numpy pipeline mismatch {d_np} vs {d_st}"
    print(f"[sanity] numpy D={d_np:.6f} vs study.dfam D={d_st:.6f}  OK")
    cache[("base", "n500")] = pb; cache[("instruct", "n500")] = pi

    for fam, lab, ckb, cka, where, kind in COMPARISONS:
        for ck in (ckb, cka):
            if (ck, where) not in cache:
                cache[(ck, where)] = precompute(load_states(ck, where))
        pbb, paa = cache[(ckb, where)], cache[(cka, where)]
        n = len(pbb["deal"])
        observed = dfam_point_np(pbb, paa)
        p, nulls = perm_null(pbb, paa, observed)
        approx = math.sqrt(2 * 5 / n)
        results[lab] = {"family": fam, "kind": kind, "N": n,
                        "observed_D": observed, "p": p,
                        "null_median": float(np.median(nulls)),
                        "null_mean": float(nulls.mean()),
                        "null_p95": float(np.percentile(nulls, 95)),
                        "null_min": float(nulls[0]), "null_max": float(nulls[-1]),
                        "unpaired_approx_sqrt2k_N": approx,
                        "percentile_of_observed": float((nulls <= observed).mean())}
        print(f"{fam:7} {lab:16} N={n:5} D={observed:.3f} p={p:.4f} "
              f"null(med={results[lab]['null_median']:.3f} "
              f"p95={results[lab]['null_p95']:.3f}) approx={approx:.3f}")

    out = {}
    src = {"source_script": "analysis/revision/t3_permutation.py",
           "source_data": "runs/n500,n2000 constrained_*.jsonl",
           "seed": SEED, "repo_fingerprint": _prov.FINGERPRINT,
           "notes": f"paired label-swap permutation, B={B}, full-pipeline recompute"}
    out["pnPermOLMo"] = {"value": results["base->instruct"]["p"],
                         "n": results["base->instruct"]["N"], **src}
    out["pnPermZephyr"] = {"value": results["z_base->z_dpo"]["p"],
                           "n": results["z_base->z_dpo"]["N"], **src}
    out["pnPermTulu"] = {"value": results["t_base->t_rlvr"]["p"],
                         "n": results["t_base->t_rlvr"]["N"], **src}
    n500 = {k: v for k, v in results.items() if v["N"] == 500}
    n2k = {k: v for k, v in results.items() if v["N"] == 2000}
    out["pnNullFloorFiveHundred"] = {"value": {
        "median_range": [min(v["null_median"] for v in n500.values()),
                         max(v["null_median"] for v in n500.values())],
        "p95_range": [min(v["null_p95"] for v in n500.values()),
                      max(v["null_p95"] for v in n500.values())],
        "per_comparison": {k: {"median": v["null_median"], "p95": v["null_p95"]}
                           for k, v in n500.items()}}, **src}
    out["pnNullFloorTwoK"] = {"value": {
        "median_range": [min(v["null_median"] for v in n2k.values()),
                         max(v["null_median"] for v in n2k.values())],
        "p95_range": [min(v["null_p95"] for v in n2k.values()),
                      max(v["null_p95"] for v in n2k.values())],
        "per_comparison": {k: {"median": v["null_median"], "p95": v["null_p95"]}
                           for k, v in n2k.items()}}, **src}
    out["pnRLVRFloorCompareOLMo"] = {"value": {
        "step": "dpo->instruct", "observed_D": results["dpo->instruct"]["observed_D"],
        "p": results["dpo->instruct"]["p"], "N": 500,
        "percentile_position": results["dpo->instruct"]["percentile_of_observed"],
        "null_median": results["dpo->instruct"]["null_median"],
        "null_p95": results["dpo->instruct"]["null_p95"]}, **src}
    out["pnRLVRFloorCompareTulu"] = {"value": {
        "step": "t_dpo->t_rlvr", "observed_D": results["t_dpo->t_rlvr"]["observed_D"],
        "p": results["t_dpo->t_rlvr"]["p"], "N": 2000,
        "percentile_position": results["t_dpo->t_rlvr"]["percentile_of_observed"],
        "null_median": results["t_dpo->t_rlvr"]["null_median"],
        "null_p95": results["t_dpo->t_rlvr"]["null_p95"]}, **src}
    out["permutation.all"] = {"value": results, **src}
    _emit.merge(out)
    print("\nT3 done. pnPerm OLMo=%.4f Zephyr=%.4f Tulu=%.4f" %
          (out["pnPermOLMo"]["value"], out["pnPermZephyr"]["value"],
           out["pnPermTulu"]["value"]))
    return out


if __name__ == "__main__":
    run()
