"""T2 — Test-4 MDE -> excluded-surplus conversion.

For each powered Test-4 seller-D cell (all 6 checkpoints; Tulu included), take the
exclusion bound on the concession-adaptation slope beta_max = max(|CI_lo|,|CI_hi|),
then evaluate a concession-adaptation buyer policy parameterized by that slope with
the existing seller-D best-response module (pure game arithmetic), and report
surplus(beta_max) - surplus(0) in alpha*-gate surplus units, both horizons.

======================= UNIT MAPPING (derived, not eyeballed) =======================
Test-4 regression response (metric_concession.episode_obs):
    concession = (p_t - p_{t-1}) / room,   room = v_b - floor   [fraction of TOTAL room]
    slope beta = d(concession)/d(hardness),  hardness h in [0,1] (op-a hold_rate or
    op-b conc_granted).
Best-response policy (sellerD_adaptation_value.simulate_d):
    p_t = p_{t-1} + max(0, conc_frac * (v_b - p_{t-1})),  conc_frac = c_base + c_adapt*h
  => realized concession as fraction of total room:
    (p_t-p_{t-1})/room = conc_frac * (v_b - p_{t-1})/room = conc_frac * rho,
       rho = (v_b - p_{t-1})/(v_b - floor)  (remaining-room fraction).
  => d[(p_t-p_{t-1})/room]/dh = c_adapt * rho.
Matching the regression slope: beta = c_adapt * rho_bar   =>   c_adapt = beta / rho_bar,
where rho_bar is the mean remaining-room fraction at the flat policy's concession
steps (computed empirically below from the flat-optimal trajectory). Both ops share
this mapping (h in [0,1] in both). We evaluate c_adapt = +/- beta_max/rho_bar and take
the surplus-maximizing sign as the (largest defensible) excluded value.
=====================================================================================

Reuses sellerD_adaptation_value.simulate_d/FEAS/params and metric_concession/
fh_analyze for the beta(D) CIs. No model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import game_lib as G
import sellerD_adaptation_value as AV
from seller_exp import seller_open, seller_floor, stub_schedule, seller_respond, SELLERS
import metric_concession as M
import fh_analyze as FH
import _prov, _emit

SEL_D = SELLERS["D"]
FEAS = AV.FEAS
CKPTS = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]
OPS = {"opa": ("a", 1), "opb": ("b", 2)}
HORIZONS = {"natural_T8": 1, "forced_k4": 4}


def flat_surplus(alpha0, cbase, op, mar):
    return AV._mean([AV.simulate_d(ep, alpha0, cbase, 0.0, op, mar) for ep in FEAS])


def adapt_surplus(alpha0, cbase, cadapt, op, mar):
    return AV._mean([AV.simulate_d(ep, alpha0, cbase, cadapt, op, mar) for ep in FEAS])


def rho_bar(alpha0, cbase, op, mar):
    """mean remaining-room fraction rho at flat-policy concession steps."""
    rhos = []
    for ep in FEAS:
        v_b, c_s = ep["v_b"], ep["c_s"]
        floor = seller_floor(c_s); room = v_b - floor
        ask = seller_open(c_s); ask0 = ask
        sched = stub_schedule(ep["seed"], SEL_D["stub"])
        hist = []; resp = []; last = None
        for t in range(1, G.T_ROUNDS + 1):
            if t == 1:
                p = max(floor, min(v_b, int(round(floor + alpha0 * room))))
            else:
                if op == "a":
                    h = (sum(r[2] for r in resp) / len(resp)) if resp else 0.0
                else:
                    h = ((ask0 - ask) / (ask0 - floor)) if ask0 > floor else 0.0
                remaining = v_b - last
                rhos.append(remaining / room)   # remaining-room fraction pre-concession
                conc = max(0.0, cbase) * remaining
                p = min(v_b, max(last, last + int(round(conc))))
            kind, val = seller_respond(SEL_D, ask, p, c_s, sched[t - 1], hist,
                                       min_accept_round=mar, round_idx=t)
            last = p; hist.append({"who": "buyer", "price": p})
            if kind == "accept":
                break
            old = ask; ask = val; resp.append((old, val, 1 if val >= old else 0))
    return AV._mean(rhos) if rhos else float("nan")


def beta_ci(ck, op_hidx, forced):
    DIR = "results/forced_horizon" if forced else "runs/seller_exp"
    b, lo, hi, nep, nob = FH.cell(DIR, ck, "D", op_hidx)
    return b, lo, hi, nob


def best_flat_and_adapt(op, mar, c_val):
    """max over (alpha_0, c_base) of surplus at fixed c_adapt=c_val. This is
    S*(c_val); S*(0) = best flat surplus."""
    best = -1.0
    for a in AV.ALPHA:
        for cb in AV.C_BASE:
            s = adapt_surplus(a, cb, c_val, op, mar)
            if s > best:
                best = s
    return best


def run():
    out = {}
    src = {"source_script": "analysis/revision/t2_mde_surplus.py",
           "seed": 1234, "repo_fingerprint": _prov.FINGERPRINT,
           "source_data": "runs/seller_exp, results/forced_horizon, "
                          "out/sellerD_adaptation_value.json"}
    cells = {}
    binding = {"points": 0.0, "cell": None, "pct": 0.0}
    # Precompute S*(v) = best-(alpha,c_base) surplus at each candidate slope v,
    # per (op, horizon).  excluded(c_bound) = max_{|v|<=c_bound} S*(v) - S*(0).
    C_CANDS = sorted(set(AV.C_ADAPT + [0.0]))
    print("Precomputing S*(v) per (op,horizon)...")
    SSTAR = {}
    RHO = {}
    for hlabel, mar in HORIZONS.items():
        for opname, (op, hidx) in OPS.items():
            SSTAR[(hlabel, opname)] = {v: best_flat_and_adapt(op, mar, v)
                                       for v in C_CANDS}
            # rho_bar from the best-flat policy (argmax at v=0)
            best0 = max(((AV._mean([AV.simulate_d(ep, a, cb, 0.0, op, mar)
                                    for ep in FEAS]), a, cb)
                         for a in AV.ALPHA for cb in AV.C_BASE))
            RHO[(hlabel, opname)] = rho_bar(best0[1], best0[2], op, mar)
            print(f"  {hlabel}/{opname}: S*(0)={SSTAR[(hlabel,opname)][0.0]:.4f} "
                  f"rho_bar={RHO[(hlabel,opname)]:.3f}")

    print("\nT2 — Test-4 MDE -> excluded surplus (seller D):")
    for hlabel, mar in HORIZONS.items():
        for opname, (op, hidx) in OPS.items():
            sstar = SSTAR[(hlabel, opname)]
            s_flat = sstar[0.0]
            rb = RHO[(hlabel, opname)]
            for ck in CKPTS:
                forced = (mar == 4)
                b, lo, hi, nob = beta_ci(ck, hidx, forced)
                if b != b or lo != lo:
                    continue
                beta_max = max(abs(lo), abs(hi))
                c_map = beta_max / rb if rb and rb == rb else float("nan")
                in_range = [v for v in C_CANDS if abs(v) <= c_map]
                s_best = max(sstar[v] for v in in_range) if in_range else s_flat
                argc = max(in_range, key=lambda v: sstar[v]) if in_range else 0.0
                excluded = max(0.0, s_best - s_flat)
                pct = excluded / s_flat * 100 if s_flat else float("nan")
                key = f"{hlabel}.{ck}.{opname}"
                cells[key] = {"beta_D": b, "ci": [lo, hi], "beta_max": beta_max,
                              "rho_bar": rb, "c_adapt_bound": c_map,
                              "c_adapt_argmax": argc,
                              "s_flat": s_flat, "excluded_points": excluded,
                              "excluded_pct_of_flat": pct, "n_obs": nob}
                if excluded > binding["points"]:
                    binding = {"points": excluded, "pct": pct, "cell": key,
                               "beta_max": beta_max}
                print(f"  {key:26} beta={b:+.3f}[{lo:+.3f},{hi:+.3f}] bmax={beta_max:.3f} "
                      f"rho={rb:.3f} c_bound={c_map:.3f} excl={excluded:.4f}pts "
                      f"({pct:.2f}%) n={nob}")

    out["pnMDEsurplus"] = {"value": binding["points"], "notes":
        f"largest defensible excluded surplus (points), cell={binding['cell']}, "
        f"beta_max={binding.get('beta_max')}; sanity anchor <= ~0.18 pts", **src}
    out["pnMDEpct"] = {"value": binding.get("pct"),
        "notes": f"as % of flat buyer surplus; cell={binding['cell']}", **src}
    out["mde_surplus.cells"] = {"value": cells, **src}
    _emit.merge(out)
    print(f"\nBINDING excluded bound: {binding['points']:.4f} pts "
          f"({binding.get('pct'):.2f}%) at {binding['cell']}")
    print("sanity: paper caps channel at ~0.18 pts (<=1.8%) ->",
          "OK" if binding['points'] <= 0.20 else "CHECK UNIT MAPPING")
    return out


if __name__ == "__main__":
    run()
