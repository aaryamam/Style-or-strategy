"""TASK B (CPU) — base-concession artifact check.
B2: context-matched override labelling (policy_shift logic) on the forced-horizon
free vs constrained runs. B3: re-fit the FROZEN Test-4 regression (metric_concession)
for β(D) and β(C)-β(B), both ops, on three turn sets:
  (i)  all constrained turns                         (sanity = Task-1 base finding)
  (ii) constrained EXCLUDING overridden offer-turns  (artifact-isolating)
  (iii) free natively-parsed turns                   (Defect-B selection bias flagged)
"""
import json, os, random, copy
import metric_concession as M

random.seed(1234)
BASE = ["base", "z_base", "t_base"]
SELL = ["B", "C", "D"]
HARD = {"a_hold_rate": 1, "b_conc_granted": 2}
VALID = {"accept", "offer", "reject", "hold", "walk"}
CON = "results/forced_horizon"        # constrained forced-horizon (Task 1)
FREE = "results/base_artifact"        # free forced-horizon (B1)


def load_states(path):
    return [json.loads(l)["state"] for l in open(path)] if os.path.exists(path) else None


def overridden_rounds(free_states, con_states):
    """Per-episode set of rounds where the constrained action TYPE differs from
    the context-matched (same ask_shown) free strict action (free must have parsed).
    Returns (ovr_by_idx, stats)."""
    free = {s["idx"]: s for s in free_states}
    con = {s["idx"]: s for s in con_states}
    common = set(free) & set(con)
    ovr = {}
    n_ctx = n_ov = n_paired = 0
    for idx in common:
        ft = {t["round"]: t for t in free[idx]["turns"]}
        ct = {t["round"]: t for t in con[idx]["turns"]}
        rounds = set()
        for r in set(ft) & set(ct):
            f, c = ft[r], ct[r]
            n_paired += 1
            if f["ask_shown"] != c["ask_shown"]:
                continue
            if not f.get("parse_success") or f.get("action") not in VALID:
                continue
            n_ctx += 1
            if f["action"] != c["action"]:
                n_ov += 1
                rounds.add(r)
        ovr[idx] = rounds
    return ovr, dict(paired=n_paired, ctx_matched=n_ctx, overrides=n_ov,
                     override_rate=(n_ov / n_ctx if n_ctx else float("nan")))


def per_ep_excl(con_states, ovr):
    """constrained per-episode obs with overridden OFFER turns removed."""
    pe = []
    for s in con_states:
        bad = ovr.get(s["idx"], set())
        s2 = copy.deepcopy(s)
        s2["turns"] = [t for t in s["turns"]
                       if not (t["round"] in bad and t["action"] == "offer")]
        o = M.episode_obs(s2)
        if o:
            pe.append(o)
    return pe


def per_ep_all(states):
    pe = []
    for s in states:
        o = M.episode_obs(s)
        if o:
            pe.append(o)
    return pe


def betaCI(pe, hidx):
    n_obs = sum(len(e) for e in pe)
    b, _ = M.fit_beta(pe, hidx)
    lo, hi = M.boot_ci(pe, hidx, n=2000) if n_obs >= 8 else (float("nan"), float("nan"))
    nn = "—"
    if b == b and lo == lo:
        nn = "**non-null**" if (lo > 0 or hi < 0) else "null"
    return b, lo, hi, len(pe), n_obs, nn


def contrast(peB, peC, hidx):
    m = min(len(peB), len(peC))
    if m < 5:
        return float("nan"), float("nan"), float("nan")
    ds = []
    for _ in range(2000):
        idx = [random.randrange(m) for _ in range(m)]
        bB, _ = M.fit_beta(peB, hidx, idx); bC, _ = M.fit_beta(peC, hidx, idx)
        if bB == bB and bC == bC:
            ds.append(bC - bB)
    ds.sort()
    if len(ds) < 20:
        return float("nan"), float("nan"), float("nan")
    pt = M.fit_beta(peC, hidx)[0] - M.fit_beta(peB, hidx)[0]
    return pt, ds[int(.025*len(ds))], ds[int(.975*len(ds))]


def main():
    out = ["# TASK B — base-concession artifact check (forced-horizon, k=4)\n"]
    # B2 override rates
    out.append("## B2 — context-matched grammar override rate (free vs constrained)")
    out.append("| ckpt | seller | paired turns | ctx-matched | overrides | override rate |")
    out.append("|---|---|---|---|---|---|")
    OVR = {}
    for ck in BASE:
        for sk in SELL:
            fs = load_states(f"{FREE}/free_{sk}_{ck}.jsonl")
            cs = load_states(f"{CON}/{sk}_{ck}.jsonl")
            if fs is None or cs is None:
                out.append(f"| {ck} | {sk} | MISSING | | | |"); continue
            ovr, st = overridden_rounds(fs, cs)
            OVR[(ck, sk)] = ovr
            out.append(f"| {ck} | {sk} | {st['paired']} | {st['ctx_matched']} | "
                       f"{st['overrides']} | {st['override_rate']!r} |")
    out.append("")

    # B3 three-way refit
    for hk, hidx in HARD.items():
        out.append(f"## B3 — β against seller D, operationalization {hk}")
        out.append("| ckpt | (i) all-constrained β(D) [CI] (ep/obs) | "
                   "(ii) excl-overridden β(D) [CI] (ep/obs) | "
                   "(iii) free-parsed β(D) [CI] (ep/obs) |")
        out.append("|---|---|---|---|")
        for ck in BASE:
            cs = load_states(f"{CON}/D_{ck}.jsonl")
            fs = load_states(f"{FREE}/free_D_{ck}.jsonl")
            if cs is None or fs is None:
                out.append(f"| {ck} | MISSING | | |"); continue
            ovr = OVR.get((ck, "D"), {})
            pe_i = per_ep_all(cs)
            pe_ii = per_ep_excl(cs, ovr)
            pe_iii = per_ep_all(fs)
            cells = []
            for pe in (pe_i, pe_ii, pe_iii):
                b, lo, hi, nep, nob, nn = betaCI(pe, hidx)
                cells.append(f"{b!r} [{lo!r},{hi!r}] {nn} ({nep}/{nob})")
            out.append(f"| {ck} | {cells[0]} | {cells[1]} | {cells[2]} |")
        out.append("")

    # B3 cross-seller contrast under (ii)
    out.append("## B3 — β(C)−β(B) under (ii) excl-overridden")
    out.append("| ckpt | op | Δ=β(C)−β(B) | 95% CI | differs? |")
    out.append("|---|---|---|---|---|")
    for ck in BASE:
        for hk, hidx in HARD.items():
            csB = load_states(f"{CON}/B_{ck}.jsonl"); csC = load_states(f"{CON}/C_{ck}.jsonl")
            if csB is None or csC is None:
                out.append(f"| {ck} | {hk} | MISSING | | |"); continue
            peB = per_ep_excl(csB, OVR.get((ck, "B"), {}))
            peC = per_ep_excl(csC, OVR.get((ck, "C"), {}))
            pt, lo, hi = contrast(peB, peC, hidx)
            diff = "—" if lo != lo else ("**YES**" if (lo > 0 or hi < 0) else "no")
            out.append(f"| {ck} | {hk} | {pt!r} | [{lo!r},{hi!r}] | {diff} |")
    out.append("")

    # verdict per base model
    out.append("## Verdict — does base caving SURVIVE on non-overridden turns?")
    out.append("| ckpt | op | (i) all | (ii) excl-ovr | survives (ii)? | read |")
    out.append("|---|---|---|---|---|---|")
    for ck in BASE:
        cs = load_states(f"{CON}/D_{ck}.jsonl")
        for hk, hidx in HARD.items():
            if cs is None:
                continue
            ovr = OVR.get((ck, "D"), {})
            bi, loi, hii, *_n = betaCI(per_ep_all(cs), hidx)
            bii, loii, hiii, *_n2 = betaCI(per_ep_excl(cs, ovr), hidx)
            surv = (loii > 0 or hiii < 0) if loii == loii else False
            read = "real (survives)" if surv else "GRAMMAR-PROJECTION ARTIFACT (collapses to null)"
            out.append(f"| {ck} | {hk} | {bi!r} | {bii!r} [{loii!r},{hiii!r}] | "
                       f"{'YES' if surv else 'NO'} | {read} |")
    open("results/base_artifact/taskB_report.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
