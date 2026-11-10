"""TASK 1 analysis — forced-horizon Test-4 concession β, reusing the FROZEN
metric_concession regression (same op-a/op-b, same controls, same bootstrap).
Side-by-side: original (runs/seller_exp) vs forced-horizon (results/forced_horizon)."""
import json, random
import metric_concession as M

random.seed(1234)
CKPTS = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]
HARD = {"a_hold_rate": 1, "b_conc_granted": 2}


def cell(DIR, ck, seller, hidx):
    M.DIR = DIR
    pe = M.load_obs(seller, ck)
    n_ep = len(pe); n_obs = sum(len(e) for e in pe)
    b, _ = M.fit_beta(pe, hidx)
    lo, hi = M.boot_ci(pe, hidx, n=2000) if n_obs >= 8 else (float("nan"), float("nan"))
    return b, lo, hi, n_ep, n_obs


def contrast_CmB(DIR, ck, hidx):
    M.DIR = DIR
    peB, peC = M.load_obs("B", ck), M.load_obs("C", ck)
    m = min(len(peB), len(peC))
    if m < 5:
        return float("nan"), float("nan"), float("nan")
    ds = []
    for _ in range(2000):
        idx = [random.randrange(m) for _ in range(m)]
        bB, _ = M.fit_beta(peB, hidx, idx)
        bC, _ = M.fit_beta(peC, hidx, idx)
        if bB == bB and bC == bC:
            ds.append(bC - bB)
    ds.sort()
    if len(ds) < 20:
        return float("nan"), float("nan"), float("nan")
    pt = (M.fit_beta(peC, hidx)[0] - M.fit_beta(peB, hidx)[0])
    return pt, ds[int(.025*len(ds))], ds[int(.975*len(ds))]


def main():
    ORIG = "runs/seller_exp"; FH = "results/forced_horizon"
    out = ["# TASK 1 — forced-minimum-horizon concession test (k_min=4)\n",
           "Reuses metric_concession.py (frozen Test-4 regression) verbatim; only "
           "the input DIR changes. Full precision.\n",
           "Forced-horizon D_fam is NOT computed here (different regime); only the "
           "concession-adaptation β is the deliverable.\n"]

    out.append("## β against adaptive seller D (+ usable obs counts), per op\n")
    for hk, hidx in HARD.items():
        out.append(f"### operationalization {hk}")
        out.append("| ckpt | orig β(D) [CI] | orig obs (ep/obs) | **FH β(D) [CI]** | "
                   "**FH obs (ep/obs)** | FH non-null? |")
        out.append("|---|---|---|---|---|---|")
        for ck in CKPTS:
            ob, olo, ohi, oep, oobs = cell(ORIG, ck, "D", hidx)
            fb, flo, fhi, fep, fobs = cell(FH, ck, "D", hidx)
            nn = "—"
            if fb == fb and flo == flo:
                nn = "**YES (CI excl 0)**" if (flo > 0 or fhi < 0) else "no"
            out.append(f"| {ck} | {ob!r} [{olo!r}, {ohi!r}] | {oep}/{oobs} | "
                       f"{fb!r} [{flo!r}, {fhi!r}] | {fep}/{fobs} | {nn} |")
        out.append("")

    out.append("## Cross-seller slope difference β(C)−β(B), forced-horizon\n")
    for hk, hidx in HARD.items():
        out.append(f"### operationalization {hk}")
        out.append("| ckpt | FH Δ=β(C)−β(B) | FH 95% CI | differs? |")
        out.append("|---|---|---|---|")
        for ck in CKPTS:
            pt, lo, hi = contrast_CmB(FH, ck, hidx)
            diff = "—"
            if lo == lo:
                diff = "**YES**" if (lo > 0 or hi < 0) else "no"
            out.append(f"| {ck} | {pt!r} | [{lo!r}, {hi!r}] | {diff} |")
        out.append("")

    # power achieved vs starved
    out.append("## Power check: forced-horizon vs original usable obs (seller D)\n")
    out.append("| ckpt | orig D obs | FH D obs | powered now? (≥~50 obs) |")
    out.append("|---|---|---|---|")
    for ck in CKPTS:
        _, _, _, _, oobs = cell(ORIG, ck, "D", 1)
        _, _, _, _, fobs = cell(FH, ck, "D", 1)
        out.append(f"| {ck} | {oobs} | {fobs} | "
                   f"{'YES' if fobs >= 50 else 'still thin'} |")
    open("results/forced_horizon/task1_report.md", "w").write("\n".join(out) + "\n")
    print("wrote results/forced_horizon/task1_report.md")
    print("\n".join(out))


if __name__ == "__main__":
    main()
