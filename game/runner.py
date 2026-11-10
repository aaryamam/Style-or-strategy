"""
Batched-across-episodes episode stepper. Maintains N parallel episode states;
each round batches all live buyer prompts into ONE vLLM call, parses, advances
the scripted seller, to T or terminal. Logs per-turn parse-failure location.

A turn's buyer action is resolved with extract_buyer_action (frozen parser).
For the constrained condition every turn is a valid frozen action. A turn that
fails to parse, or is a well-formed non-buyer action (hold/reject), or an offer
without a usable price, is a strategic NO-OP that round (seller stands).
"""
import game_lib as G


def _init_state(ep):
    return {
        "seed": ep["seed"], "idx": ep["idx"], "v_b": ep["v_b"],
        "c_s": ep["c_s"], "zopa": ep["zopa"],
        "ask": G.seller_open(ep["c_s"]),
        "opening_ask": G.seller_open(ep["c_s"]),
        "floor": G.seller_floor(ep["c_s"]),
        "stubborn": G.stubborn_schedule(ep["seed"], G.T_ROUNDS),
        "history": [], "turns": [],
        "done": False, "outcome": None, "deal_price": None, "rounds_used": 0,
    }


def _resolve(s, ex, t):
    """Apply the buyer action to state s at round t. Returns the turn record."""
    rec = {"round": t, "ask_shown": s["ask"], "raw": ex["raw"],
           "parse_success": ex["parse_success"], "action": ex["action"],
           "price": ex["price"], "fail_reason": ex["fail_reason"],
           "is_noop": ex["is_noop"], "effect": None}
    s["rounds_used"] = t

    valid_buyer = (ex["parse_success"] and ex["action"] in G.BUYER_ACTIONS
                   and not ex["is_noop"])
    if not valid_buyer:
        rec["effect"] = "noop"          # forfeit this round; seller stands
        s["turns"].append(rec)
        if t >= G.T_ROUNDS:
            s["done"] = True; s["outcome"] = "no_deal_timeout"
        return rec

    a = ex["action"]
    if a == "accept":
        rec["effect"] = "deal"; s["deal_price"] = s["ask"]
        s["done"] = True; s["outcome"] = "deal"
    elif a == "walk":
        rec["effect"] = "walk"; s["done"] = True; s["outcome"] = "walk"
    elif a == "offer":
        p = ex["price"]
        if p is None:
            rec["effect"] = "noop"
            s["turns"].append(rec)
            if t >= G.T_ROUNDS:
                s["done"] = True; s["outcome"] = "no_deal_timeout"
            return rec
        s["history"].append({"who": "buyer", "price": p})
        stub = s["stubborn"][t - 1] if t - 1 < len(s["stubborn"]) else False
        kind, val = G.seller_respond(s["ask"], p, s["c_s"], stubborn=stub)
        if kind == "accept":
            rec["effect"] = "deal"; s["deal_price"] = p
            s["done"] = True; s["outcome"] = "deal"
        else:
            rec["effect"] = f"seller_counter:{val}"
            s["ask"] = val
            s["history"].append({"who": "seller", "ask": val})
            if t >= G.T_ROUNDS:
                s["done"] = True; s["outcome"] = "no_deal_timeout"
    s["turns"].append(rec)
    return rec


def run_episodes(engine, role, episodes, condition, verbose=False):
    guided = (condition == "constrained")
    states = [_init_state(ep) for ep in episodes]
    for t in range(1, G.T_ROUNDS + 1):
        live = [s for s in states if not s["done"]]
        if not live:
            break
        prompts = [engine.render(role, s["v_b"], s["history"], s["ask"], t)
                   for s in live]
        seeds = [s["seed"] * 100 + t for s in live]
        raws = engine.generate(prompts, role, guided, seeds)
        for s, raw in zip(live, raws):
            ex = G.extract_buyer_action(raw)
            ex["raw"] = raw
            r = _resolve(s, ex, t)
            if verbose:
                print(f"  [ep {s['idx']} r{t}] ask={r['ask_shown']} "
                      f"-> {r['action']}/{r['price']} ({r['effect']})")
    # finalize any still-live (shouldn't happen past T)
    for s in states:
        if not s["done"]:
            s["done"] = True; s["outcome"] = "no_deal_timeout"
    return states
