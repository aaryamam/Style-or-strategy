"""
AXIS-1 firing-test analysis for Zephyr + Tulu (and OLMo reference).
The firing condition is NOT marginal eta > 0. It is the history-binned
differential: Dη = η(late_high) − η(early_low) > 0 with CI excluding 0
(the aggressive−passive parse gap GROWS into the high-stakes bin).
"""
import json, random, os
from collections import defaultdict

random.seed(1234)
AGG = {"hold_justified", "offer_conditional"}
PASS = {"accept", "offer_simple"}

ZT = "out/probe_runs_zephyr_tulu.jsonl"
OLMO = "out/probe_runs.jsonl"


def load():
    rows = []
    for r in (json.loads(l) for l in open(ZT)):
        rows.append(r)
    # OLMo reference (relabel base/aligned -> olmo_base/olmo_inst)
    if os.path.exists(OLMO):
        for r in (json.loads(l) for l in open(OLMO)):
            r = dict(r)
            r["model"] = "olmo_base" if r["model"] == "base" else "olmo_inst"
            rows.append(r)
    return rows


def mean(b):
    return sum(b) / len(b) if b else float("nan")


def eta_from(bucket_pass, bucket_agg):
    return mean(bucket_agg) - mean(bucket_pass)   # signed: agg - pass


def boot_pair(agg, pas, n=4000):
    out = []
    for _ in range(n):
        ra = [agg[random.randrange(len(agg))] for _ in agg] if agg else []
        rp = [pas[random.randrange(len(pas))] for _ in pas] if pas else []
        out.append(mean(ra) - mean(rp))
    out.sort()
    return out[int(.025 * n)], out[int(.975 * n)]


def boot_delta(agg_e, pas_e, agg_l, pas_l, n=4000):
    out = []
    for _ in range(n):
        def m(x):
            return mean([x[random.randrange(len(x))] for _ in x]) if x else float("nan")
        eta_e = m(agg_e) - m(pas_e)
        eta_l = m(agg_l) - m(pas_l)
        out.append(eta_l - eta_e)
    out.sort()
    return out[int(.025 * n)], out[int(.975 * n)]


def collect(rows, model):
    # action -> bits ; (bin, bucket) -> bits
    act = defaultdict(list)
    binbucket = defaultdict(list)
    for r in rows:
        if r["model"] != model:
            continue
        a = r["action_type"]; h = r["history_bin"]
        ok = 1 if r["parse_success"] else 0
        act[a].append(ok)
        if a in AGG:
            binbucket[(h, "agg")].append(ok)
        elif a in PASS:
            binbucket[(h, "pass")].append(ok)
        else:
            binbucket[(h, "walk")].append(ok)
    return act, binbucket


def verdict(marg, dlo, dhi):
    """eta is signed agg-pass; predicted direction is eta<0 (aggressive fails
    more). FIRING = the gap WIDENS into late_high, i.e. eta more negative late,
    i.e. Dη = eta_late - eta_early < 0 with CI excluding 0 (dhi < 0)."""
    if abs(marg) < 0.03:
        return "CLEAN (marginal eta ~0; (X) holds)"
    if marg >= 0:
        return "wrong-direction (aggressive fails LESS than passive)"
    if dhi < 0:
        return "FIRES (gap widens into late_high; Dη CI excludes 0 below)"
    if dlo > 0:
        return "PARTIAL-inverted (gap narrows/moves onto opening turn)"
    return "PARTIAL-flat (action-dependent but history-homogeneous)"


def main():
    rows = load()
    models = ["z_base", "z_dpo", "t_base", "t_rlvr", "olmo_base", "olmo_inst"]
    present = [m for m in models if any(r["model"] == m for r in rows)]

    print("=" * 78)
    print("PER-ACTION q_F (pooled over history bins)")
    order = ["accept", "offer_simple", "walk_reasoned", "hold_justified",
             "offer_conditional"]
    print(f"{'model':10} " + "".join(f"{a[:9]:>11}" for a in order))
    data = {}
    for m in present:
        act, bb = collect(rows, m)
        data[m] = (act, bb)
        print(f"{m:10} " + "".join(f"{mean(act[a]):11.3f}" for a in order))

    print("\n" + "=" * 78)
    print("FIRING TEST — marginal eta (agg−pass) and history-binned differential")
    print(f"{'model':10} {'marg_eta':>9} {'eta@early':>10} {'eta@mid':>9} "
          f"{'eta@late':>9} {'Dlate-early':>12} {'Dη 95% CI':>16}  verdict")
    summary = {}
    for m in present:
        act, bb = data[m]
        agg_all = [b for a in AGG for b in act[a]]
        pas_all = [b for a in PASS for b in act[a]]
        marg = mean(agg_all) - mean(pas_all)
        eta = {}
        for h in ("early_low", "mid", "late_high"):
            eta[h] = mean(bb[(h, "agg")]) - mean(bb[(h, "pass")])
        d = eta["late_high"] - eta["early_low"]
        dlo, dhi = boot_delta(bb[("early_low", "agg")], bb[("early_low", "pass")],
                              bb[("late_high", "agg")], bb[("late_high", "pass")])
        v = verdict(marg, dlo, dhi)
        summary[m] = {"marg": marg, "eta": eta, "d": d, "dci": [dlo, dhi],
                      "verdict": v}
        print(f"{m:10} {marg:9.3f} {eta['early_low']:10.3f} {eta['mid']:9.3f} "
              f"{eta['late_high']:9.3f} {d:12.3f} [{dlo:+.3f},{dhi:+.3f}]  {v}")

    print("\n(eta = signed q_F(agg)-q_F(pass); NEGATIVE = aggressive fails more "
          "= predicted. Dη = eta@late - eta@early. FIRING needs the gap to WIDEN "
          "late: eta more negative late => Dη<0 with CI upper bound < 0.)")
    return summary


if __name__ == "__main__":
    main()
