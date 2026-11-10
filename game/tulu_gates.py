"""
DA3 + DA4 for Tulu-3: signature-validity gates and covariance persistence.
Mirrors zephyr_gates.py. Reads runs/n500 constrained for t_base/t_rlvr; compares
per-component direction vs OLMo (base->instruct) and Zephyr (z_base->z_dpo).
Does NOT regenerate OLMo/Zephyr.
"""
import json, math, os
import study as ST

FIVE = ST.FIVE


def variance(vals):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return sum((x - m) ** 2 for x in vals) / len(vals)


def g1_construct(recs):
    acc, rej = [], []
    for r in recs:
        s = r["state"]
        if s["outcome"] != "deal":
            continue
        offers = [t for t in s["turns"] if t["action"] == "offer"
                  and t["price"] is not None]
        sur = r["components"]["surplus_efficiency"]
        if not offers or sur != sur:
            continue
        seal = next((t for t in s["turns"] if t["effect"] == "deal"), None)
        if seal and seal["action"] == "offer" and seal["round"] == offers[0]["round"]:
            acc.append(sur)
        else:
            rej.append(sur)
    am = sum(acc) / len(acc) if acc else float("nan")
    below = sum(1 for x in rej if x < am)
    return am, below, len(rej), len(acc)


def gaps(b, a):
    stats = {c: ST.pooled_stats(b, a, c) for c in FIVE}
    vb, va = ST.mvec(b, stats), ST.mvec(a, stats)
    return {c: va[i] - vb[i] for i, c in enumerate(FIVE)}


def main():
    tb = ST.load_constrained("t_base")
    tr = ST.load_constrained("t_rlvr")

    print("=" * 70)
    print("DA3 GATE G1 — aggression_penalty construct (Tulu-RLVR)")
    am, below, nrej, nacc = g1_construct(tr)
    print(f"  accepted-group mean surplus = {am:.3f} (n={nacc})")
    print(f"  rejected-opening deals resolving WORSE: {below}/{nrej} = "
          f"{below/nrej if nrej else float('nan'):.2f}")
    _, beta = ST._agg(tr)
    print(f"  aggression_penalty beta (t_rlvr) = {beta:+.3f}")

    print("\n" + "=" * 70)
    print("DA3 GATE G2 — non-degeneracy (variance, constrained)")
    print(f"{'component':24} {'var(t_base)':>12} {'var(t_rlvr)':>12} {'flag':>18}")
    for c in FIVE:
        vb = variance([r["components"][c] for r in tb])
        vd = variance([r["components"][c] for r in tr])
        flag = ""
        if min(vb, vd) < 0.005:
            flag = "NEAR-CONSTANT"
        if max(vb, vd) > 1.0:
            flag = (flag + " ANOM-HIGH").strip()
        print(f"{c:24} {vb:12.4f} {vd:12.4f} {flag:>18}")

    print("\n" + "=" * 70)
    print("DA3 GATE G3 — walkaway behavior (base->final)")
    g = gaps(tb, tr)
    print(f"  walkaway std gap (t_rlvr - t_base) = {g['walkaway']:+.3f}")
    print(f"  raw walkaway rate: t_base={_raw(tb,'walkaway'):.3f} "
          f"t_rlvr={_raw(tr,'walkaway'):.3f}")
    print(f"  -> {'shared-null-like (OLMo)' if abs(g['walkaway'])<0.05 else 'non-null (Zephyr-like or third)'}")

    print("\n" + "=" * 70)
    print("DA3 — per-component DIRECTION vs OLMo and Zephyr (constrained signs)")
    ob, oi = ST.load_constrained("base"), ST.load_constrained("instruct")
    zb, zd = ST.load_constrained("z_base"), ST.load_constrained("z_dpo")
    go, gz, gt = gaps(ob, oi), gaps(zb, zd), g
    def sg(x): return "+" if x > 0 else ("-" if x < 0 else "0")
    print(f"{'component':24} {'OLMo':>8} {'Zephyr':>8} {'Tulu':>8}  agreement")
    for c in FIVE:
        agree = []
        if sg(gt[c]) == sg(go[c]): agree.append("OLMo")
        if sg(gt[c]) == sg(gz[c]): agree.append("Zephyr")
        a = ",".join(agree) if agree else "NEITHER (third direction)"
        print(f"{c:24} {go[c]:+8.3f} {gz[c]:+8.3f} {gt[c]:+8.3f}  {a}")

    # ---- DA4 persist ----
    os.makedirs("out", exist_ok=True)
    stats = {c: ST.pooled_stats(tb, tr, c) for c in FIVE}
    json.dump({"FIVE": FIVE, "Sigma_tulu": ST.cov(tb, tr, stats),
               "stats": stats}, open("out/cov_tulu.json", "w"), indent=1)
    pooled_path = "out/components_pooled.json"
    pooled = json.load(open(pooled_path)) if os.path.exists(pooled_path) else {}
    for c in FIVE:
        pooled.setdefault(c, {})
        pooled[c]["tulu"] = [r["components"][c] for r in tb + tr
                             if r["components"][c] == r["components"][c]]
    json.dump(pooled, open(pooled_path, "w"))
    print("\nDA4: persisted out/cov_tulu.json; appended tulu to "
          "out/components_pooled.json")


def _raw(recs, c):
    v = [r["components"][c] for r in recs if r["components"][c] == r["components"][c]]
    return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    main()
