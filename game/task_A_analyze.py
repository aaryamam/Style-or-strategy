"""TASK A analysis (CPU) — revised fire verdict for olmo_base & t_rlvr.
A1: larger-N (n=10, all 34 decisive histories) + 2nd seed; per-seed and pooled.
A2: threshold grid {cutoff 5/6} x {eff_room >=10, 15-29, >=20} (seeds pooled).
A3: own-history probe (own-decisive scarce; own-reached any-eff_room) vs pooled.
Same firing condition + boot_delta(n=4000) as the frozen probe."""
import json, os, random
random.seed(1234)
AGG = {"hold_justified", "offer_conditional"}
PASS = {"accept", "offer_simple"}
RR = "results/realized_robust"


def load(fp):
    return [json.loads(l) for l in open(fp)] if os.path.exists(fp) else None


def bits(rows, pred, bucket):
    s = AGG if bucket == "agg" else PASS
    return [1 if r["parse_success"] else 0 for r in rows
            if r["action_type"] in s and pred(r)]


def mean(b):
    return sum(b) / len(b) if b else float("nan")


def boot_delta(ae, pe, al, pl, n=4000):
    out = []
    for _ in range(n):
        def m(x):
            return mean([x[random.randrange(len(x))] for _ in x]) if x else float("nan")
        d = (m(al) - m(pl)) - (m(ae) - m(pe))
        if d == d:
            out.append(d)
    out.sort()
    if len(out) < 20:
        return float("nan"), float("nan")
    return out[int(.025*len(out))], out[int(.975*len(out))]


def verdict(rows, early_pred, late_pred):
    ae, pe = bits(rows, early_pred, "agg"), bits(rows, early_pred, "pass")
    al, pl = bits(rows, late_pred, "agg"), bits(rows, late_pred, "pass")
    eta_e = mean(ae) - mean(pe)
    eta_l = mean(al) - mean(pl)
    marg = mean(ae + al) - mean(pe + pl)
    deta = eta_l - eta_e
    lo, hi = boot_delta(ae, pe, al, pl)
    nlate = len(al) + len(pl)
    if marg >= 0:
        v = "wrong-dir"
    elif hi == hi and hi < 0:
        v = "FIRES"
    elif lo == lo and lo > 0:
        v = "inverted"
    else:
        v = "flat"
    return dict(eta_e=eta_e, eta_l=eta_l, marg=marg, deta=deta, lo=lo, hi=hi,
                v=v, nlate=nlate)


EARLY = lambda r: r["pool"] == "early"
def latep(cut, lo, hi):
    return lambda r: r["pool"] == "late" and r["round"] >= cut and lo <= r["eff_room"] <= hi


def fmt(d):
    return (f"marg={d['marg']!r} η@early={d['eta_e']!r} η@late={d['eta_l']!r} "
            f"Δη={d['deta']!r} CI=[{d['lo']!r},{d['hi']!r}] N_late={d['nlate']} → **{d['v']}**")


def main():
    out = ["# TASK A — realized-probe robustness (olmo_base, t_rlvr)\n"]

    out.append("## A1 — larger N (n=10, all 34 decisive histories) + second seed")
    out.append("Decisive late = pool 'late' ∩ round≥6 ∩ eff_room∈[15,29] (original def). "
               "The passA decisive universe is only 34, so 3× MORE histories is "
               "infeasible; effective N raised via n=10 + 2 seeds.\n")
    a1 = {}
    for ck in ["olmo_base", "t_rlvr"]:
        out.append(f"### {ck}")
        per = {}
        for s in [1234, 777]:
            rows = load(f"{RR}/probe_{ck}_seed{s}.jsonl")
            if rows is None:
                out.append(f"- seed {s}: MISSING"); continue
            d = verdict(rows, EARLY, latep(6, 15, 29))
            per[s] = d
            out.append(f"- seed {s}: {fmt(d)}")
        both = (load(f"{RR}/probe_{ck}_seed1234.jsonl") or []) + \
               (load(f"{RR}/probe_{ck}_seed777.jsonl") or [])
        if both:
            d = verdict(both, EARLY, latep(6, 15, 29))
            per["pooled"] = d
            out.append(f"- **pooled (both seeds, ~{d['nlate']} late obs)**: {fmt(d)}")
        a1[ck] = per
        out.append("")

    out.append("## A2 — threshold sensitivity grid (both seeds pooled)")
    out.append("| ckpt | cutoff | eff_room band | Δη | 95% CI | N_late | verdict |")
    out.append("|---|---|---|---|---|---|---|")
    a2 = {}
    for ck in ["olmo_base", "t_rlvr"]:
        both = (load(f"{RR}/probe_{ck}_seed1234.jsonl") or []) + \
               (load(f"{RR}/probe_{ck}_seed777.jsonl") or [])
        fires = 0; cells = 0
        for cut in (5, 6):
            for band, lo, hi in [("≥10", 10, 999), ("15-29", 15, 29), ("≥20", 20, 999)]:
                d = verdict(both, EARLY, latep(cut, lo, hi))
                cells += 1; fires += (d["v"] == "FIRES")
                out.append(f"| {ck} | ≥{cut} | {band} | {d['deta']!r} | "
                           f"[{d['lo']!r},{d['hi']!r}] | {d['nlate']} | {d['v']} |")
        a2[ck] = (fires, cells)
    out.append("")

    out.append("## A3 — own-history probe (fired checkpoints only)")
    out.append("own-DECISIVE = own realized round≥6 ∩ eff_room≥15 (scarce); "
               "own-REACHED = own realized round≥6 any eff_room (what it actually "
               "reaches on-policy). 0 decisive-late came from either model's OWN "
               "CONSTRAINED runs.\n")
    a3 = {}
    for ck in ["olmo_base", "t_rlvr"]:
        rows = load(f"{RR}/probe_{ck}_own_seed1234.jsonl")
        meta = json.load(open(f"{RR}/own_{ck}.json"))
        out.append(f"### {ck}  (own early={len(meta['early'])}, "
                   f"own decisive-late={len(meta['late'])}, "
                   f"own reached-late={len(meta['late_loose'])})")
        if rows is None:
            out.append("- MISSING\n"); continue
        dd = verdict(rows, lambda r: r["pool"] == "early",
                     lambda r: r["pool"] == "late")
        dl = verdict(rows, lambda r: r["pool"] == "early",
                     lambda r: r["pool"] == "late_loose")
        out.append(f"- own-DECISIVE ({len(meta['late'])} hist): {fmt(dd)}")
        out.append(f"- own-REACHED ({len(meta['late_loose'])} hist): {fmt(dl)}")
        a3[ck] = (dd, dl)
        out.append("")

    # consolidated verdict per checkpoint
    out.append("## Revised fire verdict")
    out.append("| ckpt | (i) larger N+2nd seed | (ii) threshold grid | (iii) own-history | BOTTOM LINE |")
    out.append("|---|---|---|---|---|")
    for ck in ["olmo_base", "t_rlvr"]:
        pooled = a1[ck].get("pooled", {})
        s1 = a1[ck].get(1234, {}).get("v", "?"); s2 = a1[ck].get(777, {}).get("v", "?")
        i = f"pooled={pooled.get('v','?')} (s1={s1},s2={s2})"
        ii = f"{a2[ck][0]}/{a2[ck][1]} cells fire"
        dd, dl = a3.get(ck, ({}, {}))
        iii = f"decisive={dd.get('v','?')}(N={dd.get('nlate','?')}), reached={dl.get('v','?')}(N={dl.get('nlate','?')})"
        # bottom line
        robustN = pooled.get("v") == "FIRES" and s1 == "FIRES" and s2 == "FIRES"
        robustThr = a2[ck][0] >= a2[ck][1] - 1
        ownfire = dl.get("v") == "FIRES"
        if robustN and robustThr and ownfire:
            bl = "fires (robust)"
        elif (pooled.get("v") == "FIRES" or a2[ck][0] > 0) and not ownfire:
            bl = "fires only under pooled/counterfactual histories"
        elif pooled.get("v") != "FIRES":
            bl = "washes out — synthetic-flat verdict stands"
        else:
            bl = "ambiguous — see cells"
        out.append(f"| {ck} | {i} | {ii} | {iii} | **{bl}** |")
    open(f"{RR}/taskA_report.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
