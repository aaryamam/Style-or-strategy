"""
TASK 3 — leave-one-component-out + Euclidean locus robustness (CPU only).
Reuses study.py's D_fam (Mahalanobis) by swapping study.FIVE to a 4-component
subset (LOCO); Euclidean uses study's standardization + identity covariance.
Frozen episode-level data: OLMo n500, Zephyr/Tulu n2000 (as in the main result).
"""
import json, math, random
import study as ST

FULL = ["opening_aggressiveness", "surplus_efficiency", "haggling", "walkaway",
        "aggression_penalty"]
SEED = 1234
NBOOT = 2000   # labeled; locus(argmax) is bootstrap-independent, CI-sep robust at 2k

FAM = {
    "OLMo":   (["base", "sft", "dpo", "instruct"], "n500"),
    "Zephyr": (["z_base", "z_sft", "z_dpo"], "n2000"),
    "Tulu":   (["t_base", "t_sft", "t_dpo", "t_rlvr"], "n2000"),
}
POS = {0: "supervised", 1: "preference", 2: "RLVR"}


def load(ck, where):
    d = "runs/n500" if where == "n500" else "runs/n2000"
    return ST._wrap([json.loads(l)["state"]
                     for l in open(f"{d}/constrained_{ck}.jsonl")])


def euclid_dfam(b, a, comps):
    stats = {c: ST.pooled_stats(b, a, c) for c in comps}
    def vec(recs, idxs=None):
        rng = recs if idxs is None else [recs[i] for i in idxs]
        v = []
        for c in comps:
            m, sd = stats[c]
            vals = [r["components"][c] for r in rng
                    if r["components"][c] == r["components"][c]]
            v.append((sum(vals)/len(vals) - m)/sd if vals else 0.0)
        return v
    def dist(vb, va):
        return math.sqrt(sum((vb[i]-va[i])**2 for i in range(len(vb))))
    rng = random.Random(SEED)
    D = dist(vec(b), vec(a))
    n = len(b); Ds = []
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        Ds.append(dist(vec(b, idx), vec(a, idx)))
    Ds.sort()
    return {"D": D, "lo": Ds[int(.025*NBOOT)], "hi": Ds[int(.975*NBOOT)]}


def steps_for(recs_by_ck, cks, variant):
    """variant: ('maha', comps) reuses study.dfam after setting ST.FIVE=comps;
    ('eucl', comps) uses euclid_dfam."""
    kind, comps = variant
    out = []
    for i in range(len(cks)-1):
        b, a = recs_by_ck[cks[i]], recs_by_ck[cks[i+1]]
        if kind == "maha":
            ST.FIVE = comps                      # <-- reuse study.dfam on subset
            r = ST.dfam(b, a, "x", nboot=NBOOT)
        else:
            r = euclid_dfam(b, a, comps)
        out.append((f"{cks[i]}->{cks[i+1]}", i, r["D"], r["lo"], r["hi"]))
    return out


def locus(steps):
    big = max(steps, key=lambda s: s[2])
    rest = [s for s in steps if s is not big]
    ci_sep = all(big[3] > s[4] for s in rest) if rest else True
    return big, ci_sep


def main():
    rows = []
    print("TASK 3 — LOCO + Euclidean locus robustness "
          f"(NBOOT={NBOOT}; OLMo N=500, Zephyr/Tulu N=2000)\n")
    recs = {}
    for fam, (cks, where) in FAM.items():
        recs[fam] = {ck: load(ck, where) for ck in ck_list(cks)}
    variants = [("FULL-5 Mahalanobis", ("maha", FULL))]
    for drop in FULL:
        variants.append((f"LOCO drop {drop}",
                         ("maha", [c for c in FULL if c != drop])))
    variants.append(("Euclidean FULL-5", ("eucl", FULL)))

    hdr = f"{'variant':28} {'family':8} {'locus step':22} {'locus pos':12} {'CI-sep':7}"
    print(hdr); print("-"*len(hdr))
    summary = {}
    for vname, variant in variants:
        for fam, (cks, where) in FAM.items():
            st = steps_for(recs[fam], cks, variant)
            big, sep = locus(st)
            summary[(vname, fam)] = (big, sep, st)
            print(f"{vname:28} {fam:8} {big[0]:22} {POS[big[1]]:12} "
                  f"{'yes' if sep else 'NO':7}")
        print()
    ST.FIVE = FULL    # restore

    # full-precision per-step detail
    print("\nFULL-PRECISION per-step D_fam [lo, hi] (every variant):")
    for vname, variant in variants:
        print(f"\n[{vname}]")
        for fam, (cks, where) in FAM.items():
            _, _, st = summary[(vname, fam)]
            for name, pos, D, lo, hi in st:
                print(f"  {fam:8} {name:22} D={D!r} [{lo!r}, {hi!r}]")

    # invariance verdict
    expect = {"OLMo": "supervised", "Zephyr": "preference", "Tulu": "preference"}
    print("\n" + "="*70)
    print("INVARIANCE VERDICT (expected: OLMo=supervised, Zephyr/Tulu=preference)")
    flips = []
    for (vname, fam), (big, sep, st) in summary.items():
        got = POS[big[1]]
        if got != expect[fam]:
            flips.append((vname, fam, got, sep))
    if not flips:
        print("  INVARIANT — locus ordering preserved across ALL LOCO variants "
              "and Euclidean.")
    else:
        print("  FLIPS found:")
        for vname, fam, got, sep in flips:
            print(f"    {vname} / {fam}: locus -> {got} (CI-sep={sep})")


def ck_list(cks):
    return cks


if __name__ == "__main__":
    main()
