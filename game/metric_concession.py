"""
Within-episode concession-adaptation metric (pure re-analysis of seller_exp
transcripts). Tests whether the buyer's per-turn concession responds to REVEALED
seller behavior (hold rate / concession granted so far) — the move where
adaptation could occur, unlike the opening. No model inference.
"""
import json, random
import signature as S   # for _ols / _solve

random.seed(1234)
DIR = "runs/seller_exp"
CKPTS = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]
FAM = {"OLMo": ("base", "instruct"), "Zephyr": ("z_base", "z_dpo"),
       "Tulu": ("t_base", "t_rlvr")}
SELLERS = ["A", "B", "C", "D"]


def episode_obs(state):
    """Per-episode list of concession observations for t>=2 (consecutive buyer
    offers). Each obs: (concession, hold_rate_prior, conc_granted_prior,
    turn, room_prev)."""
    v_b, floor = state["v_b"], state["floor"]
    room = v_b - floor
    if room <= 0:
        return []
    offers = [(t["round"], t["price"]) for t in state["turns"]
              if t["action"] == "offer" and t["price"] is not None]
    # seller hold/concede per offer round (from effect "seller_counter:VAL")
    resp = {}
    ask_by_round = {}
    for t in state["turns"]:
        ask_by_round[t["round"]] = t["ask_shown"]
        if isinstance(t["effect"], str) and t["effect"].startswith("seller_counter:"):
            new_ask = int(t["effect"].split(":")[1])
            resp[t["round"]] = (t["ask_shown"], new_ask,
                                1 if new_ask >= t["ask_shown"] else 0)  # hold
    ask0 = ask_by_round.get(offers[0][0], state["ask"]) if offers else None
    obs = []
    for i in range(1, len(offers)):
        r_prev, p_prev = offers[i - 1]
        r, p = offers[i]
        concession = (p - p_prev) / room
        prior = [resp[rr] for (rr, _) in offers[:i] if rr in resp]
        hold_rate = (sum(h for _, _, h in prior) / len(prior)) if prior else 0.0
        ask_t = ask_by_round.get(r, ask0)
        conc_granted = (ask0 - ask_t) / (ask0 - floor) if ask0 and ask0 > floor else 0.0
        room_prev = (v_b - p_prev) / room
        obs.append((concession, hold_rate, conc_granted, float(r), room_prev))
    return obs


def load_obs(seller, ckpt):
    rows = [json.loads(l) for l in open(f"{DIR}/{seller}_{ckpt}.jsonl")]
    per_ep = []
    for r in rows:
        o = episode_obs(r["state"])
        if o:
            per_ep.append(o)
    return per_ep                       # list of per-episode obs-lists


def fit_beta(per_ep, hard_idx, idxs=None):
    """OLS concession ~ 1 + hardness + turn + room; return coef of hardness.
    hard_idx: 1=hold_rate, 2=conc_granted (position in the obs tuple)."""
    use = per_ep if idxs is None else [per_ep[i] for i in idxs]
    rows, y = [], []
    for ep in use:
        for ob in ep:
            rows.append([ob[hard_idx], ob[3], ob[4]])   # hardness, turn, room
            y.append(ob[0])
    if len(y) < 8:
        return float("nan"), len(y)
    # drop controls with no variance (else singular); always keep hardness+intercept
    keep = [0]
    for j in (1, 2):
        col = [r[j] for r in rows]
        if max(col) - min(col) > 1e-9:
            keep.append(j)
    X = [[1.0] + [r[j] for j in keep] for r in rows]
    try:
        b = S._ols(X, y)
        return b[1], len(y)            # coef of hardness (first kept predictor)
    except Exception:
        try:                            # last resort: hardness only
            b = S._ols([[1.0, r[0]] for r in rows], y)
            return b[1], len(y)
        except Exception:
            return float("nan"), len(y)


def boot_ci(per_ep, hard_idx, n=1000):
    betas = []
    m = len(per_ep)
    for _ in range(n):
        idxs = [random.randrange(m) for _ in range(m)]
        b, _ = fit_beta(per_ep, hard_idx, idxs)
        if b == b:
            betas.append(b)
    if len(betas) < 20:
        return (float("nan"), float("nan"))
    betas.sort()
    return betas[int(.025 * len(betas))], betas[int(.975 * len(betas))]


def main():
    HARD = {"hold_rate(a)": 1, "conc_granted(b)": 2}
    data = {}   # (ckpt,seller) -> per_ep
    nobs = {}
    for ck in CKPTS:
        for sel in SELLERS:
            pe = load_obs(sel, ck)
            data[(ck, sel)] = pe
            nobs[(ck, sel)] = (len(pe), sum(len(e) for e in pe))

    print("=" * 90)
    print("EFFECTIVE N per cell: (multi-turn episodes / concession observations)")
    print(f"{'ckpt':10} " + "".join(f"{s:>16}" for s in SELLERS))
    for ck in CKPTS:
        print(f"{ck:10} " + "".join(
            f"{f'{nobs[(ck,s)][0]}ep/{nobs[(ck,s)][1]}obs':>16}" for s in SELLERS))

    for hname, hidx in HARD.items():
        sign = ("adaptive sign: NEGATIVE (concede less as seller holds more)"
                if hidx == 1 else
                "adaptive sign: nonzero (concede differently as seller has conceded)")
        print("\n" + "=" * 90)
        print(f"ADAPTATION SLOPE beta  [{hname}]   {sign}")
        print(f"{'ckpt':10} {'beta(A)':>16} {'beta(B)':>16} {'beta(C)':>16} "
              f"{'beta(D)':>16}")
        betas = {}
        for ck in CKPTS:
            cells = []
            for s in SELLERS:
                b, _ = fit_beta(data[(ck, s)], hidx)
                lo, hi = boot_ci(data[(ck, s)], hidx)
                betas[(ck, s)] = (b, lo, hi)
                star = "*" if (lo == lo and (lo > 0 or hi < 0)) else " "
                cells.append(f"{b:+.3f}[{lo:+.2f},{hi:+.2f}]{star}")
            print(f"{ck:10} " + "".join(f"{c:>16}" for c in cells))

        # contrast 1: cross-seller slope difference C - B (paired bootstrap)
        print(f"\n  CONTRAST 1 — cross-seller slope diff beta(C)-beta(B) "
              "(paired episode bootstrap):")
        for ck in CKPTS:
            peB, peC = data[(ck, "B")], data[(ck, "C")]
            m = min(len(peB), len(peC))
            ds = []
            for _ in range(1000):
                idx = [random.randrange(m) for _ in range(m)]
                bB, _ = fit_beta(peB, hidx, idx)
                bC, _ = fit_beta(peC, hidx, idx)
                if bB == bB and bC == bC:
                    ds.append(bC - bB)
            ds.sort()
            if len(ds) > 20:
                lo, hi = ds[int(.025*len(ds))], ds[int(.975*len(ds))]
                pt = betas[(ck, "C")][0] - betas[(ck, "B")][0]
                sep = "DIFFERS (CI excl 0)" if (lo > 0 or hi < 0) else "no diff (CI incl 0)"
                print(f"    {ck:10} Δ={pt:+.3f} [{lo:+.3f},{hi:+.3f}]  {sep}")

        # contrast 2 already in beta(D) column (CI excludes 0?)
        # post-training change
        print(f"\n  POST-TRAINING change in beta (final - base), seller A & D:")
        for fam, (bk, fk) in FAM.items():
            for s in ["A", "D"]:
                bb = betas[(bk, s)][0]; bf = betas[(fk, s)][0]
                print(f"    {fam:8} seller {s}: base {bb:+.3f} -> final {bf:+.3f} "
                      f"(Δ={bf-bb:+.3f})")


if __name__ == "__main__":
    main()
