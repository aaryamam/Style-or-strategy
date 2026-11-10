"""T12 — is the fitted accept-timing sensitivity (T1) surplus-aligned with D?

Decides one sentence: are the accept-timing coefficients T1 found in the aligned
finals GRADIENT-ALIGNED with seller D's best response, or ANTI-best-response (the
accept-margin analog of section 8's caving check against c*=+0.070)?

Accept-policy family (extends sellerD_adaptation_value.simulate_d's oracle accept
branch to a HARDNESS-DEPENDENT threshold): with the flat-optimal concession policy
fixed, the buyer accepts the standing ask iff its position
    pos = (ask - floor)/(v_b - floor)  <=  at(h) = at0 + at_adapt * h
where h is revealed hardness so far (op-a hold_rate / op-b conc_granted, exactly the
Test-4/T1 constructors). at_adapt>0 == "accept more readily as h rises".

============================ UNIT MAPPING (derived) ============================
T1 logistic accept boundary (P=0.5):
    b0 + b_hard*h + b_ask*ask_frac + b_turn*turn + b_room*room_remaining = 0,
    room_remaining = (v_b-ask)/(v_b-floor) = 1 - pos.
Solve the room_remaining threshold r*(h) = -(b0 + b_hard*h + ...)/b_room; accept iff
room_remaining > r*  <=>  pos < 1 - r* =: at(h). So at(h) = 1 - r*(h) and
    d at / d h = b_hard / b_room   =>   **at_adapt = b_hard / b_room**.
(Sign of at_adapt = sign of b_hard when b_room>0, as fitted.) CI endpoints of
at_adapt use b_hard at its stored CI over b_room at the point estimate — a
linearization noted in the report.
==============================================================================

Alignment: sign(at_adapt_fitted) vs sign(at_adapt_D*), where at_adapt_D* is D's
surplus-optimal accept sensitivity (grid, at0 fixed at the blind optimum). Surplus
value = surplus(fitted at_adapt) - surplus(hardness-blind accept policy), all else
fixed, vs D's scripted rule, in alpha*-gate units. Zero model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import game_lib as G
import sellerD_adaptation_value as AV
from seller_exp import seller_open, seller_floor, stub_schedule, seller_respond, SELLERS
import t1_accept_timing as T1
import _prov, _emit

SEL_D = SELLERS["D"]
FEAS = AV.FEAS
FINALS = ["instruct", "z_dpo", "t_rlvr"]
BASES = ["base", "z_base", "t_base"]
CKPTS = BASES + FINALS
OPS = {"opa_hold_rate": "a", "opb_conc_granted": "b"}
HZ = {"forced_k4": (4, "results/forced_horizon"),
      "natural_T8": (1, "runs/seller_exp")}
AT0_GRID = [i / 20 for i in range(21)]                       # 0..1
AT_ADAPT_GRID = [-1.5, -1.0, -0.7, -0.5, -0.3, -0.15, -0.07, 0.0,
                 0.07, 0.15, 0.3, 0.5, 0.7, 1.0, 1.5]


def simulate_accept(ep, alpha0, cbase, op, mar, at0, at_adapt):
    """Flat concession + hardness-dependent buyer accept threshold. Returns
    captured surplus (v_b - price) or 0."""
    v_b, c_s = ep["v_b"], ep["c_s"]
    floor = seller_floor(c_s); room = v_b - floor
    if room <= 0:
        return 0.0
    ask = seller_open(c_s); ask0 = ask
    sched = stub_schedule(ep["seed"], SEL_D["stub"])
    hist = []; resp = []; last = None
    for t in range(1, G.T_ROUNDS + 1):
        if t > 1:                                    # accept check on standing ask
            if op == "a":
                h = (sum(r[2] for r in resp) / len(resp)) if resp else 0.0
            else:
                h = ((ask0 - ask) / (ask0 - floor)) if ask0 > floor else 0.0
            at = at0 + at_adapt * h
            pos = (ask - floor) / room
            if pos <= at:
                return float(v_b - ask)
        if t == 1:
            p = max(floor, min(v_b, int(round(floor + alpha0 * room))))
        else:
            p = min(v_b, max(last, last + int(round(max(0.0, cbase) * (v_b - last)))))
        kind, val = seller_respond(SEL_D, ask, p, c_s, sched[t - 1], hist,
                                   min_accept_round=mar, round_idx=t)
        last = p; hist.append({"who": "buyer", "price": p})
        if kind == "accept":
            return float(v_b - p)
        old = ask; ask = val; resp.append((old, val, 1 if val >= old else 0))
    return 0.0


def mean_surplus(alpha0, cbase, op, mar, at0, at_adapt):
    return AV._mean([simulate_accept(ep, alpha0, cbase, op, mar, at0, at_adapt)
                     for ep in FEAS])


def simulate_fitted(ep, alpha0, cbase, op, mar, b, b_hard):
    """Buyer accepts the standing ask iff the FITTED T1 logistic index > 0
    (index = b0 + b_hard*h + b_ask*ask_frac + b_turn*turn + b_room*room_rem).
    b = (b0, b_ask, b_turn, b_room); b_hard passed separately so the hardness term
    can be zeroed for the 'hardness-blind, all else fixed' baseline. This simulates
    the ACTUAL fitted accept rule vs D (no threshold-family extrapolation)."""
    b0, b_ask, b_turn, b_room = b
    v_b, c_s = ep["v_b"], ep["c_s"]
    floor = seller_floor(c_s); room = v_b - floor
    if room <= 0:
        return 0.0
    ask = seller_open(c_s); ask0 = ask
    sched = stub_schedule(ep["seed"], SEL_D["stub"])
    hist = []; resp = []; last = None
    for t in range(1, G.T_ROUNDS + 1):
        if t > 1:
            if op == "a":
                h = (sum(r[2] for r in resp) / len(resp)) if resp else 0.0
            else:
                h = ((ask0 - ask) / (ask0 - floor)) if ask0 > floor else 0.0
            ask_frac = ask / v_b
            room_rem = (v_b - ask) / room
            idx = b0 + b_hard * h + b_ask * ask_frac + b_turn * float(t) + b_room * room_rem
            if idx > 0:
                return float(v_b - ask)
        if t == 1:
            p = max(floor, min(v_b, int(round(floor + alpha0 * room))))
        else:
            p = min(v_b, max(last, last + int(round(max(0.0, cbase) * (v_b - last)))))
        kind, val = seller_respond(SEL_D, ask, p, c_s, sched[t - 1], hist,
                                   min_accept_round=mar, round_idx=t)
        last = p; hist.append({"who": "buyer", "price": p})
        if kind == "accept":
            return float(v_b - p)
        old = ask; ask = val; resp.append((old, val, 1 if val >= old else 0))
    return 0.0


def mean_fitted(alpha0, cbase, op, mar, b, b_hard):
    return AV._mean([simulate_fitted(ep, alpha0, cbase, op, mar, b, b_hard)
                     for ep in FEAS])


def blind_optimum(alpha0, cbase, op, mar):
    best = (-1.0, None)
    for at0 in AT0_GRID:
        s = mean_surplus(alpha0, cbase, op, mar, at0, 0.0)
        if s > best[0]:
            best = (s, at0)
    return best[0], best[1]


GRAD_NULL_EPS = 0.02        # material best-response value threshold (pts)


def d_gradient(alpha0, cbase, op, mar, at0, s_blind):
    """D's surplus-optimal accept sensitivity (grid) + material best-response value.
    The direction is 'null' unless conditioning accept on hardness materially beats
    the blind threshold (curve max - S_blind >= GRAD_NULL_EPS)."""
    curve = {a: mean_surplus(alpha0, cbase, op, mar, at0, a) for a in AT_ADAPT_GRID}
    argmax = max(curve, key=curve.get)
    gopt_gain = curve[argmax] - s_blind
    s_pos = mean_surplus(alpha0, cbase, op, mar, at0, 0.05)
    s_neg = mean_surplus(alpha0, cbase, op, mar, at0, -0.05)
    local = s_pos - s_neg
    if gopt_gain < GRAD_NULL_EPS:
        sign = "zero"           # accept-timing has no material best-response value
    else:
        sign = "pos" if argmax > 0 else "neg" if argmax < 0 else "zero"
    return {"argmax_at_adapt": argmax, "sign": sign, "local_grad": local,
            "gopt_gain": gopt_gain, "curve": {str(k): v for k, v in curve.items()}}


# mean achievable-room fraction of value, for the ask_frac<->pos change-of-variable
RBAR = AV._mean([(e["v_b"] - seller_floor(e["c_s"])) / e["v_b"] for e in FEAS])


def refit_coeffs(DIR, ck, hidx):
    """Return full fitted vector (b0, b_hard, b_ask, b_turn, b_room) from the T1
    logistic on seller D. X cols: [int, hard, ask_frac, turn, room]."""
    pe = T1.load_ep(DIR, "D", ck)
    X, y, hcol = T1._design(pe, hidx)
    b = T1.logit_fit(X, y)
    return tuple(float(x) for x in b)              # (b0,b_hard,b_ask,b_turn,b_room)


def anchor():
    print("ANCHOR — reproduce c*=+0.070 (op-b concession) and oracle-flat 3.24/4.32:")
    s_ad, p_ad, _ = AV.grid_search("b", AV.C_ADAPT, 1)
    cstar = p_ad[2]
    ok_c = abs(cstar - 0.07) < 1e-9
    print(f"  c* op-b natural = {cstar:+.3f}  (target +0.070)  {'OK' if ok_c else 'MISMATCH'}")
    orc_n = AV.oracle_mean(1); orc_f = AV.oracle_mean(4)
    a = json.load(open("out/sellerD_adaptation_value.json"))
    tgt_n = a["horizons"]["natural_T8"]["oracle"]["s_oracle_minus_s_flat"]
    tgt_f = a["horizons"]["forced_k4"]["oracle"]["s_oracle_minus_s_flat"]
    sfl_n = a["horizons"]["natural_T8"]["oracle"]["s_flat_best_for_oracle"]
    sfl_f = a["horizons"]["forced_k4"]["oracle"]["s_flat_best_for_oracle"]
    d_n = (orc_n - sfl_n); d_f = (orc_f - sfl_f)
    ok_n = abs(d_n - tgt_n) < 0.02; ok_f = abs(d_f - tgt_f) < 0.02
    print(f"  oracle-flat natural = {d_n:.3f} (target {tgt_n:.3f}) {'OK' if ok_n else 'MISMATCH'}")
    print(f"  oracle-flat k4      = {d_f:.3f} (target {tgt_f:.3f}) {'OK' if ok_f else 'MISMATCH'}")
    if not (ok_c and ok_n and ok_f):
        print("ANCHOR MISMATCH — stopping T12."); sys.exit(1)
    return {"c_star": cstar, "oracle_flat_natural": d_n, "oracle_flat_k4": d_f}


def run():
    anc = anchor()
    d = _emit.load_all()
    out = {}
    src = {"source_script": "analysis/revision/t12_accept_timing_value.py",
           "seed": 1234, "repo_fingerprint": _prov.FINGERPRINT,
           "source_data": "runs/seller_exp, results/forced_horizon, "
                          "out/sellerD_adaptation_value.json, revision_numbers.json"}
    a = json.load(open("out/sellerD_adaptation_value.json"))
    CEIL = {"forced_k4": 4.318, "natural_T8": 3.239}

    grad = {}; results = {}
    for hlabel, (mar, DIR) in HZ.items():
        for opname, op in OPS.items():
            alpha0, cbase, _ = a["horizons"][hlabel][
                "a_hold_rate" if op == "a" else "b_conc_granted"]["params_flat"]
            s_blind, at0star = blind_optimum(alpha0, cbase, op, mar)
            gr = d_gradient(alpha0, cbase, op, mar, at0star, s_blind)
            grad[(hlabel, opname)] = dict(gr, at0star=at0star, s_blind=s_blind,
                                          alpha0=alpha0, cbase=cbase)
            print(f"\n[{hlabel}/{opname}] blind at0*={at0star:.2f} S_blind={s_blind:.3f} "
                  f"| D gradient {gr['sign']} (argmax at_adapt={gr['argmax_at_adapt']:+.2f}, "
                  f"best-response gain over blind={gr['gopt_gain']:+.4f} pts, "
                  f"local={gr['local_grad']:+.4f})")
            gsign = gr["sign"]
            for ck in CKPTS:
                b0, b_hard, b_ask, b_turn, b_room = refit_coeffs(DIR, ck, 1 if op == "a" else 2)
                bvec = (b0, b_ask, b_turn, b_room)
                key = f"accept_timing.fh.{ck}.{opname}.D" if hlabel == "forced_k4" \
                    else f"accept_timing.seller.{ck}.{opname}.D"
                cell = d.get(key, {})
                ci = cell.get("ci", [float("nan"), float("nan")])
                coef = cell.get("value", {}).get("coef", b_hard)
                # Direction: position-threshold mapping folding both ask coefficients,
                # at_adapt = b_hard/(b_room - b_ask*RBAR); sign(at_adapt)=sign(b_hard)
                # when the denom (ask-margin identification) is well-signed.
                denom = b_room - b_ask * RBAR
                reliable = denom > 0.05
                at_pt = coef / denom if reliable else float("nan")
                # Surplus VALUE: simulate the ACTUAL fitted logistic accept rule vs D
                # (no threshold extrapolation), baseline = same rule with b_hard=0
                # ('hardness-blind, all else fixed'). Evaluate at point + CI endpoints.
                s_blind_rule = mean_fitted(alpha0, cbase, op, mar, bvec, 0.0)
                def val(bh):
                    if bh != bh:
                        return float("nan")
                    return mean_fitted(alpha0, cbase, op, mar, bvec, bh) - s_blind_rule
                dv, dlo, dhi = val(coef), val(ci[0]), val(ci[1])
                spans0 = (ci[0] == ci[0] and ci[0] <= 0 <= ci[1])
                bh_sign = "pos" if coef > 0 else "neg" if coef < 0 else "zero"
                if gsign == "zero":
                    align_pt = "null_gradient"     # accept-timing surplus-neutral vs D
                elif spans0:
                    align_pt = "null"
                elif not reliable:
                    align_pt = "unreliable_map"
                else:
                    align_pt = "aligned" if bh_sign == gsign else "opposed"
                results[f"{hlabel}.{ck}.{opname}"] = {
                    "b_hard": coef, "b_ask": b_ask, "b_room": b_room, "denom": denom,
                    "reliable": reliable, "at_adapt": at_pt,
                    "alignment": align_pt, "surplus_delta": dv,
                    "delta_at_ci_lo": dlo, "delta_at_ci_hi": dhi,
                    "s_blind_rule": s_blind_rule, "spans0": spans0, "b_hard_ci": ci}
                print(f"    {ck:9} b_hard={coef:+.3f} at_adapt={at_pt:+.3f} "
                      f"align={align_pt:14} dS(fitted-rule)={dv:+.4f} "
                      f"[{dlo:+.4f},{dhi:+.4f}]")

    # emit per-ckpt (primary = forced_k4)
    for hlabel, (mar, DIR) in HZ.items():
        for ck in CKPTS:
            for opname in OPS:
                r = results[f"{hlabel}.{ck}.{opname}"]
                out[f"accept_timing_value.{hlabel}.{ck}.{opname}"] = {
                    "value": {"alignment": r["alignment"],
                              "surplus_delta": r["surplus_delta"],
                              "delta_at_ci_lo": r["delta_at_ci_lo"],
                              "delta_at_ci_hi": r["delta_at_ci_hi"],
                              "at_adapt": r["at_adapt"], "b_room": r["b_room"]},
                    **src}
    out["accept_timing_value.d_gradient"] = {"value":
        {f"{h}.{o}": {"sign": grad[(h, o)]["sign"],
                      "argmax_at_adapt": grad[(h, o)]["argmax_at_adapt"],
                      "local_grad": grad[(h, o)]["local_grad"],
                      "at0star": grad[(h, o)]["at0star"],
                      "s_blind": grad[(h, o)]["s_blind"]}
         for h in HZ for o in OPS}, **src}
    out["accept_timing_value.anchor"] = {"value": anc, **src}

    # ---- overall direction verdict (primary = forced_k4 finals) ----
    # A cell is classifiable only if it is powered (b_hard CI excludes 0), the
    # mapping is reliable (denom>0.05), AND D's gradient is non-null on that op.
    final_cells = [(ck, opname, results[f"forced_k4.{ck}.{opname}"])
                   for ck in FINALS for opname in OPS]
    powered = [c for c in final_cells if not c[2]["spans0"]]
    classifiable = [c for c in powered
                    if c[2]["reliable"] and c[2]["alignment"] in ("aligned", "opposed")]
    if not classifiable:
        # No classifiable cell: is it because D's gradient is null on every op?
        gnull = all(grad[("forced_k4", o)]["sign"] == "zero" for o in OPS)
        verdict = "indeterminate"
        verdict_reason = ("D accept-timing gradient is null on both ops "
                          "(accept-timing has no best-response value vs D) "
                          if gnull else
                          "no powered+reliably-mapped final cell with a defined gradient")
    elif all(c[2]["alignment"] == "opposed" for c in classifiable):
        verdict = "anti_best_response"; verdict_reason = "all classifiable cells opposed"
    elif all(c[2]["alignment"] == "aligned" for c in classifiable):
        verdict = "gradient_aligned"; verdict_reason = "all classifiable cells aligned"
    else:
        verdict = "mixed"; verdict_reason = "classifiable cells split aligned/opposed"
    # surplus-value summary over powered finals (faithful fitted-rule delta)
    pr = [c for c in powered if c[2]["surplus_delta"] == c[2]["surplus_delta"]]
    max_gain = max((c[2]["surplus_delta"] for c in pr), default=float("nan"))
    min_gain = min((c[2]["surplus_delta"] for c in pr), default=float("nan"))
    # Sanity bound: the VALUE (gain) of accept-timing adaptation must be <= the
    # 3.2-4.3 pt channel ceiling. (Losses from a maladaptive rule are bounded by
    # -S_blind, not the ceiling; we check the gain side.)
    gains = [results[f"{h}.{ck}.{o}"]["surplus_delta"]
             for h in HZ for ck in CKPTS for o in OPS
             if results[f"{h}.{ck}.{o}"]["surplus_delta"] ==
             results[f"{h}.{ck}.{o}"]["surplus_delta"]]
    ceil_ok = max(gains) <= max(CEIL.values()) + 1e-6
    out["accept_timing_value.verdict"] = {"value": verdict, "reason": verdict_reason,
        "powered_reliable_final_cells": [(c[0], c[1], c[2]["alignment"],
                                          round(c[2]["surplus_delta"], 4)) for c in pr],
        "max_surplus_gain_pts": max_gain, "min_surplus_gain_pts": min_gain,
        "d_gradient_signs_k4": {o: grad[("forced_k4", o)]["sign"] for o in OPS},
        "d_gradient_gain_k4": {o: grad[("forced_k4", o)]["gopt_gain"] for o in OPS},
        "sanity_ceiling_ok": ceil_ok, **src}
    _emit.merge(out)
    cur = _emit.load_all()
    cur.setdefault("verdicts", {})["accept_timing_direction"] = verdict
    json.dump(cur, open(_emit.OUT, "w"), indent=1)

    print(f"\n=== T12 VERDICT: accept_timing_direction = {verdict} ===")
    print("reason:", verdict_reason)
    print("D k4 gradient signs:", {o: grad[("forced_k4", o)]["sign"] for o in OPS},
          "| best-response gain over blind:",
          {o: round(grad[("forced_k4", o)]["gopt_gain"], 4) for o in OPS})
    print("powered+reliable final k4 cells:",
          [(c[0], c[1], c[2]["alignment"], round(c[2]["surplus_delta"], 4)) for c in pr])
    print(f"surplus gain over blind: max={max_gain:+.4f} min={min_gain:+.4f} pts "
          f"| sanity ceiling ok = {ceil_ok}")
    return out


if __name__ == "__main__":
    run()
