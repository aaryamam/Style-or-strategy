"""sellerD_adaptation_value.py — value-of-adaptation benchmark for seller D.
No model inference. Writes out/sellerD_adaptation_value.json.
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import game_lib as G
from seller_exp import (SELLERS, seller_open, seller_floor, stub_schedule,
                        seller_respond, optimal_opening)

SEL_D = SELLERS["D"]
GRID = G.make_grid(500)
FEAS = [e for e in GRID if e['eff_room'] > 0]
N = len(FEAS)
MEAN_ROOM = sum(e['eff_room'] for e in FEAS) / N


def simulate_d(ep, alpha_0, c_base, c_adapt, op, min_accept_round=1, accept_thresh=None):
    """Parametric buyer policy against seller D.

    Opening: floor + alpha_0 * eff_room.
    Each subsequent round: offer += max(0, c_base + c_adapt * h) * remaining_room.
    h='a' -> hold_rate; h='b' -> conc_granted (matching metric_concession operationalizations).
    accept_thresh (oracle only): if current ask <= floor + accept_thresh*eff_room, accept ask.
    Returns captured surplus (0 if no deal).
    """
    v_b, c_s = ep['v_b'], ep['c_s']
    floor = seller_floor(c_s)
    eff_room = v_b - floor
    ask = seller_open(c_s)
    ask_0 = ask
    sched = stub_schedule(ep['seed'], SEL_D['stub'])
    buyer_hist = []          # {who:buyer, price:p} — for _aggression_signal
    seller_resp = []         # (old_ask, new_ask, is_hold) — for hardness signals
    last_offer = None

    for t in range(1, G.T_ROUNDS + 1):
        # Oracle accept: buyer explicitly accepts current ask if favorable
        if accept_thresh is not None and t > 1:
            if ask <= floor + accept_thresh * eff_room:
                return float(v_b - ask)

        # Compute buyer's offer
        if t == 1:
            p = max(floor, min(v_b, int(round(floor + alpha_0 * eff_room))))
        else:
            if op == 'a':
                h = (sum(r[2] for r in seller_resp) / len(seller_resp)
                     if seller_resp else 0.0)
            else:  # 'b'
                h = ((ask_0 - ask) / (ask_0 - floor) if ask_0 > floor else 0.0)
            remaining = v_b - last_offer
            conc_frac = max(0.0, c_base + c_adapt * h)
            concession = int(round(conc_frac * remaining))
            p = min(v_b, max(last_offer, last_offer + concession))

        # seller_respond called BEFORE appending current offer (mirrors frozen loop)
        kind, val = seller_respond(SEL_D, ask, p, c_s, sched[t - 1], buyer_hist,
                                   min_accept_round=min_accept_round, round_idx=t)
        last_offer = p
        buyer_hist.append({"who": "buyer", "price": p})

        if kind == 'accept':
            return float(v_b - p)
        old_ask = ask
        ask = val
        seller_resp.append((old_ask, val, 1 if val >= old_ask else 0))

    return 0.0


# Grid for Steps 2-3
ALPHA   = [i / 20 for i in range(21)]                          # 0.00 .. 1.00 (21)
C_BASE  = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12,
           0.15, 0.20, 0.25, 0.30, 0.40, 0.50]                # 12
C_ADAPT = [-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.15, -0.07,
            0.0,
            0.07, 0.15, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]        # 17


def _mean(vals):
    return sum(vals) / len(vals)


def grid_search(op, c_adapt_vals, min_accept_round):
    best_s, best_p, best_wa = -1.0, None, None
    for a in ALPHA:
        for cb in C_BASE:
            for ca in c_adapt_vals:
                surs = [simulate_d(ep, a, cb, ca, op, min_accept_round) for ep in FEAS]
                s = _mean(surs)
                if s > best_s:
                    best_s = s
                    best_wa = sum(1 for x in surs if x == 0) / N
                    best_p = (a, cb, ca)
    return best_s, best_p, best_wa


def oracle_mean(min_accept_round, n_per_ep=3000, seed=42):
    """Per-episode random search: buyer has oracle knowledge of episode + D's rule."""
    rng = random.Random(seed)
    total = 0.0
    for ep in FEAS:
        best = 0.0
        for _ in range(n_per_ep):
            a  = rng.uniform(0, 1)
            cb = rng.uniform(0, 0.6)
            ca = rng.uniform(-4.0, 4.0)
            at = rng.uniform(0, 1)      # accept_thresh
            op = rng.choice(['a', 'b'])
            s  = simulate_d(ep, a, cb, ca, op, min_accept_round, accept_thresh=at)
            if s > best:
                best = s
        total += best
    return total / N


def run_horizon(min_accept_round, label):
    print(f"\n{'='*64}")
    print(f"HORIZON {label}  (min_accept_round={min_accept_round})")
    res = {}

    for op_name, op in [('a_hold_rate', 'a'), ('b_conc_granted', 'b')]:
        print(f"\n  op={op_name}")
        t0 = time.time()
        s_flat, p_flat, wa_flat = grid_search(op, [0.0], min_accept_round)
        print(f"  FLAT   S={s_flat:.4f}  wa={wa_flat:.3f}  "
              f"alpha={p_flat[0]:.2f} c_base={p_flat[1]:.3f}  [{time.time()-t0:.1f}s]")

        t0 = time.time()
        s_adapt, p_adapt, wa_adapt = grid_search(op, C_ADAPT, min_accept_round)
        c_star = p_adapt[2]
        print(f"  ADAPT  S={s_adapt:.4f}  wa={wa_adapt:.3f}  "
              f"alpha={p_adapt[0]:.2f} c_base={p_adapt[1]:.3f} c*={c_star:.3f}  [{time.time()-t0:.1f}s]")

        delta = s_adapt - s_flat
        sign_str = ("concede LESS to harder — correct adaptive direction, OPPOSITE to base-model caving"
                    if c_star < 0 else
                    "concede MORE to harder — SAME direction as base-model caving" if c_star > 0
                    else "zero — flat policy is optimal in this restricted class")
        print(f"  Delta_S={delta:.4f}  ({delta/s_flat*100:.2f}% of S_flat, "
              f"{delta/MEAN_ROOM*100:.2f}% of mean_room)")
        print(f"  c* sign: {'NEG' if c_star < 0 else 'POS' if c_star > 0 else 'ZERO'}  -> {sign_str}")

        res[op_name] = dict(
            s_flat=s_flat, s_adapt=s_adapt, delta_s=delta,
            delta_s_frac_flat=delta/s_flat, delta_s_frac_room=delta/MEAN_ROOM,
            c_star=c_star, c_star_sign='neg' if c_star<0 else 'pos' if c_star>0 else 'zero',
            wa_flat=wa_flat, wa_adapt=wa_adapt,
            params_flat=list(p_flat), params_adapt=list(p_adapt),
        )

    t0 = time.time()
    s_oracle = oracle_mean(min_accept_round)
    s_flat_best = max(res['a_hold_rate']['s_flat'], res['b_conc_granted']['s_flat'])
    print(f"\n  ORACLE  S={s_oracle:.4f}  oracle-S_flat={s_oracle-s_flat_best:.4f}  [{time.time()-t0:.1f}s]")
    res['oracle'] = dict(s_oracle=s_oracle,
                         s_oracle_minus_s_flat=s_oracle - s_flat_best,
                         s_flat_best_for_oracle=s_flat_best)
    return res


def main():
    out = {}

    # === Step 1: validation gate ===
    print("=== STEP 1: VALIDATION GATE ===")
    targets = {'A': (0.45, 11.7), 'B': (0.05, 22.2), 'C': (0.75, 2.5)}
    gate = {}
    passed = True
    for k in ['A', 'B', 'C']:
        a, s, _, nf = optimal_opening(k, n=500)
        ta, ts = targets[k]
        ok = abs(a - ta) < 0.01 and abs(s - ts) < 0.5
        status = "PASS" if ok else "FAIL"
        if not ok:
            passed = False
        print(f"  {k}: alpha={a:.3f} (tgt {ta}) surplus={s:.4f} (tgt {ts})  [{status}]")
        gate[k] = dict(alpha=a, surplus=s, target_alpha=ta, target_surplus=ts, status=status)
    out['step1_validation'] = gate
    if not passed:
        print("GATE FAIL — aborting"); return
    print(f"  Gate PASS. Feasible N={N}, mean_eff_room={MEAN_ROOM:.2f}\n")

    # === Steps 2-3-4 per horizon ===
    out['horizons'] = {}
    for label, mar in [('natural_T8', 1), ('forced_k4', 4)]:
        out['horizons'][label] = run_horizon(mar, label)

    out['meta'] = dict(
        grid_n=len(GRID), feasible_n=N, mean_eff_room=MEAN_ROOM,
        alpha_grid=ALPHA, c_base_grid=C_BASE, c_adapt_grid=C_ADAPT,
        oracle_n_per_ep=3000, oracle_rng_seed=42,
        policy_form=(
            "p_1 = floor + alpha_0*(v_b-floor); "
            "p_t = p_{t-1} + max(0, (c_base + c_adapt*h_t) * (v_b - p_{t-1})); "
            "offers are integers, monotonically non-decreasing, capped at v_b"),
        h_a="hold_rate = fraction of prior seller counters that were holds (Test-4 op-a)",
        h_b="conc_granted = (ask_0 - ask_t)/(ask_0 - floor) cumulative seller concession (Test-4 op-b)",
        flat_def="c_adapt fixed at 0",
        delta_s_note="Restricted parametric form: LOWER BOUND on Test-4 observable-channel value of adaptation",
        oracle_note="Per-episode random search n=3000; buyer has oracle knowledge of episode params + D rule + stub_schedule. Mean of per-episode optima = UPPER BOUND on any agent's mean surplus.",
    )

    _root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(_root, 'out'), exist_ok=True)
    with open(os.path.join(_root, 'out', 'sellerD_adaptation_value.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("\nWrote out/sellerD_adaptation_value.json")

    # === Console summary ===
    print("\n" + "="*64)
    print("SUMMARY")
    for label in ['natural_T8', 'forced_k4']:
        r = out['horizons'][label]
        print(f"\nHorizon {label}:")
        for opn in ['a_hold_rate', 'b_conc_granted']:
            d = r[opn]
            print(f"  {opn}: S_flat={d['s_flat']:.4f}  S_adapt={d['s_adapt']:.4f}  "
                  f"Delta_S={d['delta_s']:.4f} ({d['delta_s_frac_flat']*100:.2f}% of flat)  "
                  f"c*={d['c_star']:.3f} ({d['c_star_sign']})")
        orr = r['oracle']
        print(f"  Oracle: S={orr['s_oracle']:.4f}  "
              f"Oracle-S_flat={orr['s_oracle_minus_s_flat']:.4f}")


if __name__ == '__main__':
    main()
