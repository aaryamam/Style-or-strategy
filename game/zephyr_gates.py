"""
A3 + A4: signature-validity gates on Zephyr and covariance persistence.

Gates (mirror the OLMo overpay-cut / beta-robustness / dispersion checks):
  G1 aggression_penalty construct check — on Zephyr-DPO rejected-opening deals,
     fraction resolving above the accepted-group surplus mean.
  G2 component non-degeneracy — variance of each of the 5 components on z_base
     and z_dpo (constrained); flag near-constant.
  G3 walkaway sanity — base->final walkaway standardized gap (near-shared-null?).
A4 persists per-family Zephyr Sigma and the per-episode standardized component
   matrix for both families (OLMo recomputed from its saved runs) for a future
   shared-pooled Sigma — without regenerating OLMo.
"""
import json, math, os
import signature as S
import study as ST   # reuse _wrap, load_constrained, cov, pooled_stats, FIVE

FIVE = ST.FIVE
DIR = "runs/n500"


def variance(vals):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return float("nan"), len(vals)
    m = sum(vals) / len(vals)
    return sum((x - m) ** 2 for x in vals) / len(vals), len(vals)


def g1_construct(dpo_recs):
    """rejected-opening deals: fraction with surplus_efficiency BELOW the
    accepted-group mean (rejected openings resolve worse = aggression backfires
    = positive aggression_penalty signal). Reported as the OLMo beta-check
    analogue: fraction of rejected-opening deals worse than accepted-group."""
    acc, rej = [], []
    for r in dpo_recs:
        s = r["state"]
        if s["outcome"] != "deal":
            continue
        offers = [t for t in s["turns"] if t["action"] == "offer"
                  and t["price"] is not None]
        if not offers:
            continue
        surplus = r["components"]["surplus_efficiency"]
        if surplus != surplus:
            continue
        seal = next((t for t in s["turns"] if t["effect"] == "deal"), None)
        first_round = offers[0]["round"]
        if seal and seal["action"] == "offer" and seal["round"] == first_round:
            acc.append(surplus)
        else:
            rej.append(surplus)
    acc_mean = sum(acc) / len(acc) if acc else float("nan")
    below = sum(1 for x in rej if x < acc_mean)
    return acc_mean, below, len(rej), len(acc)


def main():
    zb = ST.load_constrained("z_base")
    zd = ST.load_constrained("z_dpo")

    print("=" * 70)
    print("A3 GATE G1 — aggression_penalty construct (Zephyr-DPO)")
    am, below, nrej, nacc = g1_construct(zd)
    frac = below / nrej if nrej else float("nan")
    print(f"  accepted-group mean surplus_efficiency = {am:.3f} (n={nacc})")
    print(f"  rejected-opening deals resolving BELOW it (worse): "
          f"{below}/{nrej} = {frac:.2f}")
    # also the influence-fold beta on z_dpo
    _, beta = ST._agg(zd)
    print(f"  aggression_penalty (beta) on z_dpo = {beta:+.3f}")

    print("\n" + "=" * 70)
    print("A3 GATE G2 — component non-degeneracy (variance, constrained)")
    print(f"{'component':24} {'var(z_base)':>12} {'var(z_dpo)':>12} {'flag':>14}")
    for c in FIVE:
        vb, nb = variance([r["components"][c] for r in zb])
        vd, nd = variance([r["components"][c] for r in zd])
        flag = ""
        for v in (vb, vd):
            if v == v and v < 0.005:
                flag = "NEAR-CONSTANT"
        print(f"{c:24} {vb:12.4f} {vd:12.4f} {flag:>14}")

    print("\n" + "=" * 70)
    print("A3 GATE G3 — walkaway sanity (base->final standardized gap)")
    stats = {c: ST.pooled_stats(zb, zd, c) for c in FIVE}
    vbv = ST.mvec(zb, stats); vdv = ST.mvec(zd, stats)
    wi = FIVE.index("walkaway")
    print(f"  walkaway std gap (z_dpo - z_base) = {vdv[wi]-vbv[wi]:+.3f}")
    print(f"  raw walkaway rate: z_base="
          f"{_raw(zb,'walkaway'):.3f} z_dpo={_raw(zd,'walkaway'):.3f}")

    # ---- A4 persist covariances ----
    os.makedirs("out", exist_ok=True)
    Sig_z = ST.cov(zb, zd, stats)
    json.dump({"FIVE": FIVE, "Sigma_zephyr": Sig_z,
               "stats": {c: stats[c] for c in FIVE}},
              open("out/cov_zephyr.json", "w"), indent=1)
    # OLMo per-episode standardized rows (recomputed, not regenerated)
    ob = ST.load_constrained("base"); od = ST.load_constrained("instruct")
    ostats = {c: ST.pooled_stats(ob, od, c) for c in FIVE}
    Sig_o = ST.cov(ob, od, ostats)
    json.dump({"FIVE": FIVE, "Sigma_olmo": Sig_o,
               "stats": {c: ostats[c] for c in FIVE}},
              open("out/cov_olmo.json", "w"), indent=1)
    # pooled raw component values both families (for a shared-pooled Sigma later)
    pooled = {c: {"olmo": [r["components"][c] for r in ob + od
                           if r["components"][c] == r["components"][c]],
                  "zephyr": [r["components"][c] for r in zb + zd
                            if r["components"][c] == r["components"][c]]}
              for c in FIVE}
    json.dump(pooled, open("out/components_pooled.json", "w"))
    print("\nA4: persisted out/cov_zephyr.json, out/cov_olmo.json, "
          "out/components_pooled.json")


def _raw(recs, c):
    vals = [r["components"][c] for r in recs if r["components"][c] == r["components"][c]]
    return sum(vals) / len(vals) if vals else float("nan")


if __name__ == "__main__":
    main()
