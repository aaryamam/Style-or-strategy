"""
Free-decoding Pass A + re-simulation for the strict / lenient / matched
conditions. ONE generation pass per model (no guided decoding); the realized
trajectory is played under STRICT credit (parse_success), and each turn records
both the strict and lenient (core-valid) interpretations + the raw output.
strict/lenient/matched signatures are then RE-SCORED by re-simulating the
deterministic seller against the recorded per-turn actions (no regeneration).

Caveat (documented): a turn that is strict-fail-but-core-valid (foreign key)
branches the lenient re-sim from the strict-played trajectory; the buyer's later
recorded actions were generated against the strict asks. In this game such turns
are rare (base's format deficit is dominated by core-INVALID multiple-object /
missing-price failures, which are no-ops under BOTH rules), so the re-score is
near-exact; episodes are flagged when they branch.
"""
import json, os
import game_lib as G


def run_free_pass(engine, role, episodes, out_path):
    """Play free decoding under strict credit; record per-turn raw + both
    interpretations. Save one record per episode."""
    states = []
    for ep in episodes:
        states.append({
            "seed": ep["seed"], "idx": ep["idx"], "v_b": ep["v_b"],
            "c_s": ep["c_s"], "floor": G.seller_floor(ep["c_s"]),
            "zopa": ep["zopa"], "eff_room": ep.get("eff_room", 0),
            "stubborn": G.stubborn_schedule(ep["seed"], G.T_ROUNDS),
            "ask": G.seller_open(ep["c_s"]), "history": [], "turns": [],
            "done": False,
        })
    for t in range(1, G.T_ROUNDS + 1):
        live = [s for s in states if not s["done"]]
        if not live:
            break
        prompts = [engine.render(role, s["v_b"], s["history"], s["ask"], t)
                   for s in live]
        seeds = [s["seed"] * 100 + t for s in live]
        raws = engine.generate(prompts, role, guided=False, seeds=seeds)
        for s, raw in zip(live, raws):
            ex = G.extract_buyer_action(raw)
            rec = {"round": t, "ask_shown": s["ask"], "raw": raw,
                   "parse_success": ex["parse_success"],
                   "strict_action": ex["strict_action"],
                   "lenient_action": ex["lenient_action"],
                   "price": ex["price"]}
            s["turns"].append(rec)
            # advance the realized trajectory under STRICT credit
            a, p = ex["strict_action"], ex["price"]
            if a not in G.BUYER_ACTIONS:           # no-op (fail / hold / reject)
                if t >= G.T_ROUNDS:
                    s["done"] = True
                continue
            if a == "accept":
                s["done"] = True
            elif a == "walk":
                s["done"] = True
            elif a == "offer":
                if p is None:
                    if t >= G.T_ROUNDS:
                        s["done"] = True
                    continue
                stub = s["stubborn"][t - 1]
                kind, val = G.seller_respond(s["ask"], p, s["c_s"], stub)
                s["history"].append({"who": "buyer", "price": p})
                if kind == "accept":
                    s["done"] = True
                else:
                    s["ask"] = val
                    s["history"].append({"who": "seller", "ask": val})
                    if t >= G.T_ROUNDS:
                        s["done"] = True
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for s in states:
            f.write(json.dumps(s) + "\n")
    return states


def replay(rec, rule):
    """Re-simulate the seller against the recorded per-turn actions under
    `rule` in {'strict','lenient'}. Returns a state dict compatible with
    signature.episode_components, plus 'branched' and 'any_strict_fail' flags."""
    c_s = rec["c_s"]
    ask = G.seller_open(c_s)
    stubborn = rec["stubborn"]
    state = {"idx": rec["idx"], "v_b": rec["v_b"], "c_s": c_s,
             "floor": rec["floor"], "zopa": rec["zopa"],
             "outcome": None, "deal_price": None, "rounds_used": 0,
             "turns": []}
    branched = False
    any_strict_fail = False
    done = False
    for tr in rec["turns"]:
        t = tr["round"]
        state["rounds_used"] = t
        act = tr["strict_action"] if rule == "strict" else tr["lenient_action"]
        price = tr["price"]
        if not tr["parse_success"]:
            any_strict_fail = True
        # branch detection: strict-fail but lenient-valid
        if (tr["strict_action"] not in G.BUYER_ACTIONS
                and tr["lenient_action"] in G.BUYER_ACTIONS):
            if rule == "lenient":
                branched = True
        out = {"round": t, "ask_shown": ask, "raw": tr["raw"],
               "action": act, "price": price,
               "parse_success": (act is not None), "effect": None}
        if act not in G.BUYER_ACTIONS:
            out["effect"] = "noop"
            state["turns"].append(out)
            if t >= G.T_ROUNDS:
                done = True; state["outcome"] = "no_deal_timeout"
            continue
        if act == "accept":
            out["effect"] = "deal"; state["deal_price"] = ask
            done = True; state["outcome"] = "deal"
        elif act == "walk":
            out["effect"] = "walk"; done = True; state["outcome"] = "walk"
        elif act == "offer":
            if price is None:
                out["effect"] = "noop"; state["turns"].append(out)
                if t >= G.T_ROUNDS:
                    done = True; state["outcome"] = "no_deal_timeout"
                continue
            stub = stubborn[t - 1] if t - 1 < len(stubborn) else False
            kind, val = G.seller_respond(ask, price, c_s, stub)
            if kind == "accept":
                out["effect"] = "deal"; state["deal_price"] = price
                done = True; state["outcome"] = "deal"
            else:
                out["effect"] = f"seller_counter:{val}"; ask = val
                if t >= G.T_ROUNDS:
                    done = True; state["outcome"] = "no_deal_timeout"
        state["turns"].append(out)
        if done:
            break
    if not done and state["outcome"] is None:
        state["outcome"] = "no_deal_timeout"
    state["branched"] = branched
    state["any_strict_fail"] = any_strict_fail
    # all-strict-parsed (for matched subset): every turn that was generated
    # strict-parsed
    state["all_strict_ok"] = all(tr["parse_success"] for tr in rec["turns"])
    return state
