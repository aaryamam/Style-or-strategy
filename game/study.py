"""
N=500 study compute: base->final four-condition trajectory (re-confirm go/no-go)
and consecutive-checkpoint stage attribution under constrained decoding.
Reuses the frozen signature + influence-function fold-in + effect-size
Mahalanobis from stage2/stage1_final.

Usage:
  python study.py trajectory          # base vs instruct, 4 conditions
  python study.py attribution         # all 4 checkpoints, consecutive D_fam
"""
import json, math, random, sys, os
import signature as S
import freepass as FP

FIVE = ["opening_aggressiveness", "surplus_efficiency", "haggling",
        "walkaway", "aggression_penalty"]
ORDER = ["base", "sft", "dpo", "instruct"]
DIR = "runs/n500"
SEED = 1234
NAN = float("nan")


# ---------- aggression influence-function fold-in ----------
def _inv(M):
    n = len(M)
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(M)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        d = A[c][c]
        A[c] = [x / d for x in A[c]]
        for r in range(n):
            if r != c:
                f = A[r][c]
                A[r] = [A[r][i] - f * A[c][i] for i in range(2 * n)]
    return [row[n:] for row in A]


def _agg(records):
    idxs, rows = [], []
    for i, r in enumerate(records):
        c = r["components"]
        if (r["state"]["outcome"] == "deal"
                and c["opening_offer"] == c["opening_offer"]
                and c["final_price"] == c["final_price"]):
            idxs.append(i)
            rows.append((c["opening_offer"], r["state"]["v_b"],
                         r["state"]["c_s"], c["final_price"]))
    n = len(rows)
    if n < 6:
        return {}, NAN
    X = [[1.0, a, b, cc] for (a, b, cc, _) in rows]
    y = [d for *_, d in rows]
    beta = S._ols(X, y)
    XtXn = [[sum(X[r][i] * X[r][j] for r in range(n)) / n for j in range(4)]
            for i in range(4)]
    inv4 = _inv(XtXn)
    a = {}
    for k in range(n):
        e = y[k] - sum(X[k][j] * beta[j] for j in range(4))
        a[idxs[k]] = beta[1] + sum(inv4[1][j] * X[k][j] for j in range(4)) * e
    return a, beta[1]


def _wrap(states):
    recs = [{"state": s, "components": S.episode_components(s)} for s in states]
    a, beta = _agg(recs)
    for i, r in enumerate(recs):
        r["components"]["aggression_penalty"] = a.get(i, NAN)
    return recs


# ---------- loaders ----------
def load_constrained(ckpt):
    rows = [json.loads(l) for l in open(f"{DIR}/constrained_{ckpt}.jsonl")]
    return _wrap([r["state"] for r in rows])


def load_passA(ckpt):
    return [json.loads(l) for l in open(f"{DIR}/passA_{ckpt}.jsonl")]


def cond_records(ckpt, condition, matched_mask=None):
    if condition == "constrained":
        return load_constrained(ckpt)
    pa = load_passA(ckpt)
    rule = "strict" if condition in ("strict", "matched") else "lenient"
    states = [FP.replay(r, rule) for r in pa]
    if condition == "matched":
        states = [s for s, keep in zip(states, matched_mask) if keep]
    return _wrap(states)


# ---------- distance ----------
def pooled_stats(b, a, comp):
    vals = [r["components"][comp] for r in b + a
            if r["components"][comp] == r["components"][comp]]
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
    return m, (sd if sd > 1e-9 else 1.0)


def mvec(recs, stats, idxs=None):
    rng = recs if idxs is None else [recs[i] for i in idxs]
    v = []
    for c in FIVE:
        m, sd = stats[c]
        vals = [r["components"][c] for r in rng
                if r["components"][c] == r["components"][c]]
        v.append((sum(vals) / len(vals) - m) / sd if vals else 0.0)
    return v


def cov(b, a, stats, ridge=0.05):
    k = len(FIVE)
    rows = [[((r["components"][c] - stats[c][0]) / stats[c][1])
             if r["components"][c] == r["components"][c] else NAN
             for c in FIVE] for r in b + a]
    Sig = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            pr = [(r[i], r[j]) for r in rows if r[i] == r[i] and r[j] == r[j]]
            if len(pr) < 5:
                Sig[i][j] = 1.0 if i == j else 0.0
                continue
            n = len(pr)
            mi = sum(x for x, _ in pr) / n
            mj = sum(y for _, y in pr) / n
            Sig[i][j] = sum((x - mi) * (y - mj) for x, y in pr) / n
    for i in range(k):
        Sig[i][i] += ridge
    return Sig


def maha(vb, va, Sinv):
    d = [vb[i] - va[i] for i in range(len(vb))]
    Sd = [sum(Sinv[i][j] * d[j] for j in range(len(d))) for i in range(len(d))]
    return math.sqrt(max(0.0, sum(d[i] * Sd[i] for i in range(len(d))))), d


def dfam(b, a, label, nboot=2000):
    rng = random.Random(SEED)
    stats = {c: pooled_stats(b, a, c) for c in FIVE}
    Sinv = _inv(cov(b, a, stats))
    vb, va = mvec(b, stats), mvec(a, stats)
    D, _ = maha(vb, va, Sinv)
    Ds = []
    nb, na = len(b), len(a)
    for _ in range(nboot):
        ib = [rng.randrange(nb) for _ in range(nb)]
        ia = ib if nb == na else [rng.randrange(na) for _ in range(na)]
        Ds.append(maha(mvec(b, stats, ib), mvec(a, stats, ia), Sinv)[0])
    Ds.sort()
    lo, hi = Ds[int(0.025 * len(Ds))], Ds[int(0.975 * len(Ds))]
    gaps = {c: va[i] - vb[i] for i, c in enumerate(FIVE)}
    return {"label": label, "D": D, "lo": lo, "hi": hi, "gaps": gaps,
            "n_b": nb, "n_a": na}


def fail_turns(ckpt):
    pa = load_passA(ckpt)
    return sum(1 for r in pa for t in r["turns"] if not t["parse_success"])


def matched_mask(c1, c2):
    a = load_passA(c1)
    b = load_passA(c2)
    return [all(t["parse_success"] for t in a[i]["turns"])
            and all(t["parse_success"] for t in b[i]["turns"])
            for i in range(len(a))]


def ptable(results):
    print(f"{'component':24} " + "".join(f"{r['label']:>13}" for r in results))
    for c in FIVE:
        print(f"{c:24} " +
              "".join(f"{r['gaps'][c]:13.3f}" if r['gaps'][c] == r['gaps'][c]
                      else f"{'n/a':>13}" for r in results))


def trajectory(c1="base", c2="instruct"):
    mask = matched_mask(c1, c2)
    conds = ["strict", "lenient", "constrained", "matched"]
    res = []
    for cond in conds:
        b = cond_records(c1, cond, mask)
        a = cond_records(c2, cond, mask)
        res.append(dfam(b, a, cond))
    print("=" * 74)
    print(f"{c1.upper()} -> {c2.upper()} four-condition trajectory, N=500 paired")
    print(f"  Pass A strict-fail turns: {c1}={fail_turns(c1)}, "
          f"{c2}={fail_turns(c2)}; matched-valid={sum(mask)}/500")
    print(f"\n{'condition':12} {'N':>10} {'D_fam':>7} {'95% CI':>20}")
    for r in res:
        ci = f"[{r['lo']:.3f}, {r['hi']:.3f}]"
        nlabel = f"{r['n_b']}/{r['n_a']}"
        print(f"{r['label']:12} {nlabel:>10} {r['D']:7.3f} {ci:>20}")
    print("\nPER-COMPONENT STANDARDIZED GAP (aligned - base) BY CONDITION")
    ptable(res)
    sc = next(r for r in res if r["label"] == "strict")["D"]
    cc = next(r for r in res if r["label"] == "constrained")["D"]
    print(f"\nstrict->constrained sign check: strict={sc:.3f} "
          f"constrained={cc:.3f}  ({'INCREASE' if cc > sc else 'decrease'})")


def attribution():
    present = [c for c in ORDER if os.path.exists(f"{DIR}/constrained_{c}.jsonl")]
    print("=" * 74)
    print("STAGE ATTRIBUTION (constrained decoding, N=500)")
    print("per-checkpoint Pass A strict-fail turns:")
    for c in present:
        if os.path.exists(f"{DIR}/passA_{c}.jsonl"):
            print(f"  {c:10}: {fail_turns(c)}")
    recs = {c: load_constrained(c) for c in present}
    steps = [(present[i], present[i + 1]) for i in range(len(present) - 1)]
    if "base" in present and "instruct" in present:
        steps.append(("base", "instruct"))
    res = []
    for c1, c2 in steps:
        res.append(dfam(recs[c1], recs[c2], f"{c1}->{c2}"))
    print(f"\n{'step':22} {'D_fam':>7} {'95% CI':>20}")
    for r in res:
        print(f"{r['label']:22} {r['D']:7.3f} "
              f"[{r['lo']:.3f}, {r['hi']:.3f}]")
    print("\nPER-COMPONENT STANDARDIZED GAP BY STEP (consecutive + base->final)")
    ptable(res)

    # ----- stage map: which CONSECUTIVE step moves each component most -----
    cons = [r for r in res if "->" in r["label"] and r["label"] != "base->instruct"]
    print("\nSTAGE MAP — largest-magnitude consecutive step per component:")
    print(f"{'component':24} {'stage':16} {'gap':>8}  (consecutive gaps)")
    for c in FIVE:
        best = max(cons, key=lambda r: abs(r["gaps"][c]) if r["gaps"][c] == r["gaps"][c] else -1)
        seq = " ".join(f"{r['gaps'][c]:+.2f}" for r in cons)
        print(f"{c:24} {best['label']:16} {best['gaps'][c]:8.3f}  [{seq}]")

    # ----- walkaway-null check across stages -----
    wk = [(r["label"], r["gaps"]["walkaway"]) for r in cons]
    flag = [l for l, g in wk if g == g and abs(g) > 0.15]
    print("\nwalkaway across stages: " +
          " ".join(f"{l.split('->')[0]}->{l.split('->')[1][:3]}={g:+.2f}"
                   for l, g in wk) +
          ("  [NULL at all stages]" if not flag else f"  [MOVES at: {flag}]"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "trajectory"
    if cmd == "trajectory":
        c1 = sys.argv[2] if len(sys.argv) > 2 else "base"
        c2 = sys.argv[3] if len(sys.argv) > 3 else "instruct"
        trajectory(c1, c2)
    else:
        attribution()
