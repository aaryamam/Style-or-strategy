"""
Per-episode strategic-signature components (no distance here). The six spec
components, computed per episode where they have a per-episode value, plus the
raw material for the cross-episode anchoring regression and the cooperativeness
diagnostics.

All ZOPA-relative quantities use known v_b, c_s (position in [c_s, v_b], so
0 = seller floor / most aggressive, 1 = buyer's own valuation / least
aggressive).
"""
import math

NAN = float("nan")


def _buyer_offers(state):
    """List of (round, price, ask_shown) for each buyer OFFER action."""
    out = []
    for r in state["turns"]:
        if r["action"] == "offer" and r["price"] is not None:
            out.append((r["round"], r["price"], r["ask_shown"]))
    return out


def episode_components(state):
    v_b, c_s = state["v_b"], state["c_s"]
    # Normalize to the ACHIEVABLE bargaining set [floor, v_b] (what the buyer can
    # actually win against this seller), not the theoretical [c_s, v_b]. This
    # keeps ZOPA-relative components in sane bounds and isolates strategy from
    # the seller's fixed margin. 0 = floor (best achievable), 1 = v_b (zero
    # surplus); >1 = overpaid above own valuation.
    import game_lib as _G
    floor = state.get("floor", _G.seller_floor(c_s))
    feasible = (v_b - floor) > 0          # a profitable deal is possible
    width = (v_b - floor) if feasible else NAN
    offers = _buyer_offers(state)
    deal = state["outcome"] == "deal"
    dp = state["deal_price"]

    # opening offer + aggressiveness (position of first offer in [floor, v_b])
    opening_offer = offers[0][1] if offers else NAN
    opening_aggr = ((opening_offer - floor) / width
                    if (offers and feasible) else NAN)

    # concession_rate: mean per-step fraction of remaining distance to seller
    concs = []
    for i in range(1, len(offers)):
        prev_p = offers[i - 1][1]
        p = offers[i][1]
        ask = offers[i][2]            # seller ask the buyer faced this step
        denom = ask - prev_p
        if denom > 0:
            concs.append((p - prev_p) / denom)
    concession_rate = (sum(concs) / len(concs)) if concs else NAN

    # surplus_efficiency (feasible deals only)
    surplus_eff = ((v_b - dp) / width if (deal and feasible) else NAN)

    # acceptance_threshold: position of the agreed price in [floor, v_b]
    acceptance_thr = ((dp - floor) / width if (deal and feasible) else NAN)

    # walkaway: feasible episodes only, 1 if no deal
    walkaway = (0.0 if deal else 1.0) if feasible else NAN

    # haggling: binary, defined on ALL episodes — did the buyer make >=2 offers
    # (iterative bargaining) vs a single offer then resolve. Replaces the
    # starved continuous concession_rate.
    haggling = 1.0 if len(offers) >= 2 else 0.0

    # cooperativeness diagnostics
    opens_above_vb = (1.0 if (offers and opening_offer >= v_b) else 0.0) \
        if offers else NAN
    # accepted via an 'accept' action within the first two rounds
    early_accept = 0.0
    for r in state["turns"]:
        if r["action"] == "accept" and r["round"] <= 2:
            early_accept = 1.0
            break

    return {
        "idx": state["idx"], "v_b": v_b, "c_s": c_s, "feasible": feasible,
        "outcome": state["outcome"], "rounds_used": state["rounds_used"],
        "opening_offer": opening_offer, "final_price": (dp if deal else NAN),
        # six components (per-episode views)
        "opening_aggressiveness": opening_aggr,
        "concession_rate": concession_rate,
        "surplus_efficiency": surplus_eff,
        "acceptance_threshold": acceptance_thr,
        "walkaway": walkaway,
        "haggling": haggling,
        # diagnostics
        "opens_above_vb": opens_above_vb,
        "early_accept": early_accept,
        "n_offers": len(offers),
    }


def _stats(xs):
    xs = [x for x in xs if x == x]   # drop NaN
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": NAN, "std": NAN, "min": NAN, "max": NAN}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n) if n > 1 else 0.0
    return {"n": n, "mean": m, "std": sd, "min": min(xs), "max": max(xs)}


def dispersion_report(records):
    """Return per-component dispersion stats + the two cooperativeness fractions
    + anchoring identifiability."""
    comps = ["opening_aggressiveness", "concession_rate", "surplus_efficiency",
             "acceptance_threshold", "walkaway"]
    disp = {c: _stats([r[c] for r in records]) for c in comps}

    # anchoring is a cross-episode regression coefficient; its identifiability
    # depends on opening_offer dispersion among feasible deals.
    deals = [r for r in records if r["outcome"] == "deal" and r["feasible"]]
    disp["anchoring_input(opening_offer|deals)"] = _stats(
        [r["opening_offer"] for r in deals])
    disp["anchoring_beta"] = {"beta": _anchoring_beta(deals),
                              "n_deals": len(deals)}

    coop = {
        "frac_open_at_or_above_vb": _frac([r["opens_above_vb"] for r in records]),
        "frac_early_accept_round<=2": _frac([r["early_accept"] for r in records]),
    }
    return disp, coop


def _frac(xs):
    xs = [x for x in xs if x == x]
    return (sum(xs) / len(xs)) if xs else NAN


def _anchoring_beta(deals):
    """OLS coef of opening_offer in final_price ~ opening_offer + v_b + c_s."""
    rows = [(r["opening_offer"], r["v_b"], r["c_s"], r["final_price"])
            for r in deals
            if r["opening_offer"] == r["opening_offer"]
            and r["final_price"] == r["final_price"]]
    n = len(rows)
    if n < 5:
        return NAN
    # design matrix [1, opening, v_b, c_s]
    X = [[1.0, a, b, c] for (a, b, c, _) in rows]
    y = [d for (_, _, _, d) in rows]
    try:
        beta = _ols(X, y)
        return beta[1]            # opening_offer coefficient
    except Exception:
        return NAN


def _ols(X, y):
    k = len(X[0])
    XtX = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(k)]
           for i in range(k)]
    Xty = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(k)]
    return _solve(XtX, Xty)


def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            raise ValueError("singular")
        d = M[col][col]
        M[col] = [x / d for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [M[r][i] - f * M[col][i] for i in range(n + 1)]
    return [M[i][n] for i in range(n)]
