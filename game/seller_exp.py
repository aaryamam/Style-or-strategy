"""
Seller-parameterization experiment: 4 sellers (A reference, B concession-
rewarding, C aggression-punishing, D adaptive) sharing the FROZEN opening,
floor, grid, parser, signature, and buyer prompts. Only the concession /
stubbornness / acceptance mechanics differ. Same achievable range [floor, v_b]
across sellers (open+floor frozen) so opening_aggressiveness is comparable.
"""
import random
import game_lib as G

# Seller A = frozen reference (reproduces the main study's seller exactly).
SELLERS = {
    "A": {"gamma": 0.15, "stub": 0.35, "accept_frac": 0.50, "adaptive": False},
    # B: concedes fast, low hold, accepts lower offers -> aggression pays,
    #    optimal opening MORE aggressive (lower) than A.
    "B": {"gamma": 0.40, "stub": 0.10, "accept_frac": 0.40, "adaptive": False},
    # C: barely concedes, high hold, strict acceptance bar -> aggression
    #    backfires, optimal opening MORE conservative (higher) than A.
    "C": {"gamma": 0.08, "stub": 0.50, "accept_frac": 0.85, "adaptive": False},
    # D: adaptive -> concession conditioned on the buyer's aggression history
    #    (real opponent structure). Deterministic given history.
    "D": {"gamma": 0.15, "stub": 0.35, "accept_frac": 0.50, "adaptive": True},
}


def seller_open(c_s):
    return G.seller_open(c_s)           # frozen


def seller_floor(c_s):
    return G.seller_floor(c_s)          # frozen


def acceptance_bar(seller, ask, c_s):
    floor = seller_floor(c_s)
    return int(round(floor + seller["accept_frac"] * (ask - floor)))


def stub_schedule(seed, p, T=G.T_ROUNDS):
    rng = random.Random((seed ^ 0x5EED) + int(p * 1000))
    return [rng.random() < p for _ in range(T)]


def _aggression_signal(buyer_hist, ask, c_s):
    """How aggressive has the buyer been: mean (bar - offer)/(ask-floor) over
    its offers so far; >0 = offering below the bar (aggressive)."""
    floor = seller_floor(c_s)
    offs = [h["price"] for h in buyer_hist if h["who"] == "buyer"]
    if not offs:
        return 0.0
    span = max(1, ask - floor)
    return sum((acceptance_bar({"accept_frac": 0.5}, ask, c_s) - p) / span
               for p in offs) / len(offs)


def seller_respond(seller, ask, buyer_price, c_s, stub_this_round, buyer_hist,
                   min_accept_round=1, round_idx=1):
    """Returns ('accept', price) or ('counter', new_ask). For D, gamma/hold
    adapt to the buyer's aggression history.

    ADDITIVE FLAG (default-off; min_accept_round=1, round_idx=1 reproduce the
    frozen behavior bit-identically): when round_idx < min_accept_round the
    seller's ACCEPT branch is suppressed (it counters instead, even if the offer
    clears its bar). Open/concession/hold/walkaway are unchanged; the buyer may
    still walk. Used only by the forced-minimum-horizon experiment."""
    floor = seller_floor(c_s)
    gamma = seller["gamma"]
    hold = stub_this_round
    if seller["adaptive"]:
        agg = _aggression_signal(buyer_hist, ask, c_s)
        # punish aggression: concede less and hold more if buyer lowballed;
        # reward patience: concede more if buyer has been near the ask.
        gamma = max(0.05, 0.15 - 0.45 * agg)          # agg>0 -> less concession
        hold = stub_this_round or (agg > 0.35)         # extra hold vs aggressors
    if buyer_price >= acceptance_bar(seller, ask, c_s) and round_idx >= min_accept_round:
        return ("accept", buyer_price)
    if hold:
        return ("counter", ask)
    new_ask = ask - gamma * (ask - floor)
    new_ask = max(floor, min(ask, int(round(new_ask))))
    return ("counter", new_ask)


# ---------------------------------------------------------------------------
# GATE: optimal fixed-opening policy per seller (no model inference).
# Benchmark buyer: each round offer a FIXED price P(alpha)=floor+alpha*(v_b-
# floor); deal the first round P clears the acceptance bar; surplus=v_b-P if a
# deal occurs within T else 0. argmax_alpha differs across sellers iff the
# strategically-correct opening genuinely differs.
# ---------------------------------------------------------------------------
def optimal_opening(seller_key, n=500, alphas=None):
    if alphas is None:
        alphas = [i / 40 for i in range(41)]      # 0.000 .. 1.000 step .025
    sel = SELLERS[seller_key]
    grid = [e for e in G.make_grid(n) if e["eff_room"] > 0]   # feasible
    best_alpha, best_surplus = None, -1.0
    curve = {}
    for a in alphas:
        tot = 0.0
        for e in grid:
            c_s, v_b = e["c_s"], e["v_b"]
            floor = seller_floor(c_s)
            P = int(round(floor + a * (v_b - floor)))
            ask = seller_open(c_s)
            sched = stub_schedule(e["seed"], sel["stub"])
            surplus = 0.0
            hist = []
            for t in range(1, G.T_ROUNDS + 1):
                kind, val = seller_respond(sel, ask, P, c_s,
                                           sched[t - 1], hist)
                hist.append({"who": "buyer", "price": P})
                if kind == "accept":
                    surplus = v_b - val
                    break
                ask = val
            tot += surplus
        m = tot / len(grid)
        curve[a] = m
        if m > best_surplus:
            best_surplus, best_alpha = m, a
    return best_alpha, best_surplus, curve, len(grid)


OPTIMAL_ALPHA = {"A": 0.450, "B": 0.050, "C": 0.750}   # from the gate


def run_vs_seller(engine, role, episodes, seller_key, min_accept_round=1,
                  guided=True):
    """Play N episodes (constrained decoding) against `seller_key`. Buyer prompt
    is frozen (no seller-type leakage); the buyer experiences the seller only
    through realized asks/history. Returns states for signature.episode_components."""
    sel = SELLERS[seller_key]
    states = []
    for ep in episodes:
        states.append({
            "seed": ep["seed"], "idx": ep["idx"], "v_b": ep["v_b"],
            "c_s": ep["c_s"], "floor": seller_floor(ep["c_s"]),
            "zopa": ep["zopa"], "eff_room": ep.get("eff_room", 0),
            "ask": seller_open(ep["c_s"]),
            "stub": stub_schedule(ep["seed"], sel["stub"]),
            "history": [], "turns": [], "done": False,
            "outcome": None, "deal_price": None, "rounds_used": 0,
        })
    for t in range(1, G.T_ROUNDS + 1):
        live = [s for s in states if not s["done"]]
        if not live:
            break
        prompts = [engine.render(role, s["v_b"], s["history"], s["ask"], t)
                   for s in live]
        seeds = [s["seed"] * 100 + t for s in live]
        raws = engine.generate(prompts, role, guided=guided, seeds=seeds)
        for s, raw in zip(live, raws):
            ex = G.extract_buyer_action(raw)
            a, p = ex["action"], ex["price"]
            s["rounds_used"] = t
            rec = {"round": t, "ask_shown": s["ask"], "raw": raw,
                   "parse_success": ex["parse_success"], "action": a,
                   "price": p, "effect": None}
            if a not in G.BUYER_ACTIONS or (a == "offer" and p is None):
                rec["effect"] = "noop"; s["turns"].append(rec)
                if t >= G.T_ROUNDS:
                    s["done"] = True; s["outcome"] = "no_deal_timeout"
                continue
            if a == "accept":
                rec["effect"] = "deal"; s["deal_price"] = s["ask"]
                s["done"] = True; s["outcome"] = "deal"
            elif a == "walk":
                rec["effect"] = "walk"; s["done"] = True; s["outcome"] = "walk"
            elif a == "offer":
                kind, val = seller_respond(sel, s["ask"], p, s["c_s"],
                                           s["stub"][t - 1], s["history"],
                                           min_accept_round=min_accept_round,
                                           round_idx=t)
                s["history"].append({"who": "buyer", "price": p})
                if kind == "accept":
                    rec["effect"] = "deal"; s["deal_price"] = p
                    s["done"] = True; s["outcome"] = "deal"
                else:
                    rec["effect"] = f"seller_counter:{val}"; s["ask"] = val
                    s["history"].append({"who": "seller", "ask": val})
                    if t >= G.T_ROUNDS:
                        s["done"] = True; s["outcome"] = "no_deal_timeout"
            s["turns"].append(rec)
    for s in states:
        if not s["done"]:
            s["done"] = True; s["outcome"] = "no_deal_timeout"
    return states


if __name__ == "__main__":
    print("GATE — optimal fixed-opening alpha per seller "
          "(alpha = opening position in [floor, v_b]; LOWER = more aggressive)")
    print(f"{'seller':8} {'argmax_alpha':>12} {'mean_surplus':>13} "
          f"{'opt_offer_pos':>14}")
    res = {}
    for k in ["A", "B", "C"]:
        a, s, curve, nfeas = optimal_opening(k)
        res[k] = a
        print(f"{k:8} {a:12.3f} {s:13.3f} {a:14.3f}   (n_feas={nfeas})")
    print("\nDivergence check (need optima to ORDER differently):")
    print(f"  argmax_alpha:  B={res['B']:.3f}  A={res['A']:.3f}  C={res['C']:.3f}")
    ordered = res['B'] < res['A'] < res['C']
    print(f"  B < A < C (B most aggressive optimum, C most conservative)? "
          f"{ordered}")
    spread = max(res.values()) - min(res.values())
    print(f"  spread = {spread:.3f}  -> "
          f"{'DIVERGE (gate PASSES)' if spread >= 0.10 and ordered else 'TOO CLOSE / wrong order — RETUNE'}")
