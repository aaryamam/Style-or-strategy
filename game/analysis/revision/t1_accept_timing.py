"""T1 — Accept-timing adaptation ("Test 5").  NEW analysis; needs per-turn logs.

Unit of analysis: every buyer decision point at which a standing seller ask
exists (accept is available) = every recorded turn. accept = 1 iff the buyer's
action is 'accept' (explicitly taking the standing ask).

Model (logistic, per checkpoint):
    accept ~ hardness + ask_frac + turn_index + room_remaining
  hardness   : revealed seller hardness so far, BOTH Test-4 operationalizations
               op-a = cumulative hold rate (fraction of prior seller counters
                      that were holds), op-b = cumulative concession granted
                      (ask0-ask_t)/(ask0-floor)   [reuses the Test-4 constructors]
  ask_frac   : standing ask / v_b            (buyer-observable)
  room_remaining : (v_b - ask_shown)/(v_b - floor)  (analyst-side; ANALYST DECISION
               below: 'achievable room the buyer would still capture by accepting
               now', normalized by total achievable room v_b-floor)
  turn_index : round

Episode-cluster bootstrap CIs (resample episodes; B=2000, fixed seed 1234).
Contrasts per checkpoint x op: (i) seller D (primary), (ii) pooled across sellers
with seller fixed effects, (iii) beta(B)-beta(C) difference. Same on forced-horizon.
Per-cell usable accept-decision counts. Displacement Delta(final-base) per family.

ANALYST DECISIONS (logged): (1) accept binary excludes offer-sealed deals (an
offer that clears the bar is an 'offer', not an accept-timing choice). (2)
room_remaining normalization as above. (3) logistic fit by IRLS with tiny L2
ridge (1e-4) for numerical stability under quasi-separation. (4) round-1 decision
points are included (hardness=0 baseline); turn_index controls for stage.
No model inference.
"""
import json, os, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import _prov, _emit

SEED = 1234
B = 2000
SELLERS = ["A", "B", "C", "D"]
FAM = {"OLMo": ("base", "instruct"), "Zephyr": ("z_base", "z_dpo"),
       "Tulu": ("t_base", "t_rlvr")}
CKPTS = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]
RIDGE = 1e-4


def decision_points(state):
    """Per-episode rows: (accept, hold_rate, conc_granted, turn, room_remaining,
    ask_frac). One row per recorded turn (standing ask present)."""
    v_b, floor = state["v_b"], state["floor"]
    room = v_b - floor
    if room <= 0:
        return []
    turns = state["turns"]
    ask0 = turns[0]["ask_shown"] if turns else None
    rows = []
    holds = []           # prior seller-response hold flags
    for t in turns:
        ask = t["ask_shown"]
        hold_rate = (sum(holds) / len(holds)) if holds else 0.0
        conc = ((ask0 - ask) / (ask0 - floor)) if (ask0 and ask0 > floor) else 0.0
        room_remaining = (v_b - ask) / room
        ask_frac = ask / v_b if v_b else 0.0
        accept = 1 if t["action"] == "accept" else 0
        rows.append((accept, hold_rate, conc, float(t["round"]),
                     room_remaining, ask_frac))
        # update holds from THIS turn's seller response (affects later points)
        eff = t.get("effect")
        if isinstance(eff, str) and eff.startswith("seller_counter:"):
            new_ask = int(eff.split(":")[1])
            holds.append(1 if new_ask >= ask else 0)
    return rows


def load_ep(DIR, seller, ck):
    rows = [json.loads(l)["state"] for l in open(f"{DIR}/{seller}_{ck}.jsonl")]
    return [decision_points(s) for s in rows]     # list of per-episode row-lists


def _design(eplist, hard_idx, seller_fe=None):
    """Flatten episodes to X, y. hard_idx: 1=hold_rate, 2=conc_granted.
    seller_fe: optional list (parallel to eplist) of seller labels -> add dummies.
    Returns X (with intercept), y, and column index of hardness (0-based in X)."""
    X, y, sfe = [], [], []
    for k, ep in enumerate(eplist):
        for r in ep:
            X.append([1.0, r[hard_idx], r[5], r[3], r[4]])  # int, hard, ask_frac, turn, room
            y.append(r[0])
            if seller_fe is not None:
                sfe.append(seller_fe[k])
    X = np.array(X, float); y = np.array(y, float)
    if seller_fe is not None and len(sfe):
        cats = sorted(set(sfe))[1:]     # drop first as reference
        for c in cats:
            X = np.column_stack([X, np.array([1.0 if s == c else 0.0 for s in sfe])])
    return X, y, 1   # hardness is column 1


def logit_fit(X, y, ridge=RIDGE, iters=50):
    n, p = X.shape
    beta = np.zeros(p)
    R = ridge * np.eye(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        W = np.clip(W, 1e-8, None)
        grad = X.T @ (y - mu) - ridge * beta
        H = X.T @ (X * W[:, None]) + R
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def fit_hardness(eplist, hard_idx, seller_fe=None):
    if sum(len(e) for e in eplist) < 10:
        return float("nan")
    X, y, hcol = _design(eplist, hard_idx, seller_fe)
    if y.sum() < 1 or y.sum() > len(y) - 1:
        # no variation in accept -> hardness coef undefined/degenerate
        return float("nan")
    b = logit_fit(X, y)
    return float(b[hcol])


def boot_ci(eplist, hard_idx, seller_fe=None, b=B, seed=SEED):
    rng = np.random.RandomState(seed)
    m = len(eplist)
    vals = []
    for _ in range(b):
        idx = rng.randint(0, m, m)
        sub = [eplist[i] for i in idx]
        subfe = [seller_fe[i] for i in idx] if seller_fe is not None else None
        v = fit_hardness(sub, hard_idx, subfe)
        if v == v:
            vals.append(v)
    if len(vals) < 20:
        return (float("nan"), float("nan"))
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def counts(eplist):
    n_dp = sum(len(e) for e in eplist)
    n_acc = sum(r[0] for e in eplist for r in e)
    return n_dp, n_acc


def contrast_BC(peB, peC, hard_idx, b=B, seed=SEED):
    rng = np.random.RandomState(seed)
    m = min(len(peB), len(peC))
    pt = fit_hardness(peC, hard_idx) - fit_hardness(peB, hard_idx)
    ds = []
    for _ in range(b):
        idx = rng.randint(0, m, m)
        bB = fit_hardness([peB[i] for i in idx], hard_idx)
        bC = fit_hardness([peC[i] for i in idx], hard_idx)
        if bB == bB and bC == bC:
            ds.append(bC - bB)
    if len(ds) < 20:
        return pt, float("nan"), float("nan")
    ds.sort()
    return pt, ds[int(.025 * len(ds))], ds[int(.975 * len(ds))]


def displacement(data, DIR, hard_idx, seller):
    """Delta(final-base) hardness coef with paired episode bootstrap, per family."""
    res = {}
    rng = np.random.RandomState(SEED)
    for fam, (bk, fk) in FAM.items():
        peb, pef = data[(seller, bk)], data[(seller, fk)]
        pt = fit_hardness(pef, hard_idx) - fit_hardness(peb, hard_idx)
        m = min(len(peb), len(pef))
        ds = []
        for _ in range(B):
            idx = rng.randint(0, m, m)
            vb = fit_hardness([peb[i] for i in idx], hard_idx)
            vf = fit_hardness([pef[i] for i in idx], hard_idx)
            if vb == vb and vf == vf:
                ds.append(vf - vb)
        ds.sort()
        ci = ([ds[int(.025*len(ds))], ds[int(.975*len(ds))]]
              if len(ds) >= 20 else [float("nan"), float("nan")])
        res[fam] = {"delta": pt, "ci": ci}
    return res


OPS = {"opa_hold_rate": 1, "opb_conc_granted": 2}


def run_horizon(DIR, tag):
    data = {}
    for ck in CKPTS:
        for s in SELLERS:
            data[(s, ck)] = load_ep(DIR, s, ck)
    table = {}
    fires = []
    all_spanD = True
    for ck in CKPTS:
        for opname, hidx in OPS.items():
            # (i) seller D
            peD = data[("D", ck)]
            bD = fit_hardness(peD, hidx)
            loD, hiD = boot_ci(peD, hidx)
            ndp, nacc = counts(peD)
            spans0 = not (loD == loD and (loD > 0 or hiD < 0))
            if not spans0:
                fires.append(f"{ck}.{opname}.D")
            all_spanD = all_spanD and spans0
            # (ii) pooled + seller FE
            pooled = [data[(s, ck)][i] for s in SELLERS for i in range(len(data[(s, ck)]))]
            fe = [s for s in SELLERS for _ in range(len(data[(s, ck)]))]
            bP = fit_hardness(pooled, hidx, fe)
            loP, hiP = boot_ci(pooled, hidx, fe)
            # (iii) B vs C
            ptBC, loBC, hiBC = contrast_BC(data[("B", ck)], data[("C", ck)], hidx)
            table[f"{ck}.{opname}"] = {
                "D": {"coef": bD, "ci": [loD, hiD], "n_dp": ndp, "n_acc": nacc,
                      "spans0": spans0},
                "pooled_FE": {"coef": bP, "ci": [loP, hiP]},
                "BvsC": {"diff": ptBC, "ci": [loBC, hiBC]},
            }
    disp = {op: displacement(data, DIR, hidx, "D") for op, hidx in OPS.items()}
    return table, fires, all_spanD, disp, data


def run():
    out = {}
    src = {"source_script": "analysis/revision/t1_accept_timing.py",
           "seed": SEED, "repo_fingerprint": _prov.FINGERPRINT}
    print("=== T1 accept-timing: seller_exp (natural horizon) ===")
    tbl_s, fires_s, span_s, disp_s, _ = run_horizon("runs/seller_exp", "seller")
    print("=== T1 accept-timing: forced-horizon (k=4) ===")
    tbl_f, fires_f, span_f, disp_f, _ = run_horizon("results/forced_horizon", "fh")

    for horizon, tbl, sd in [("seller", tbl_s, "runs/seller_exp"),
                             ("fh", tbl_f, "results/forced_horizon")]:
        for cell, v in tbl.items():
            ck, op = cell.split(".")
            base = f"accept_timing.{horizon}.{ck}.{op}"
            out[f"{base}.D"] = {"value": {"coef": v["D"]["coef"]},
                "ci": v["D"]["ci"], "n": v["D"]["n_dp"],
                "source_data": f"{sd}/D_{ck}.jsonl",
                "notes": f"accepts={v['D']['n_acc']}; spans0={v['D']['spans0']}", **src}
            out[f"{base}.pooledFE"] = {"value": {"coef": v["pooled_FE"]["coef"]},
                "ci": v["pooled_FE"]["ci"], "source_data": f"{sd}/*_{ck}.jsonl", **src}
            out[f"{base}.BvsC"] = {"value": {"diff": v["BvsC"]["diff"]},
                "ci": v["BvsC"]["ci"], "source_data": f"{sd}/[BC]_{ck}.jsonl", **src}
    out["accept_timing.displacement.seller"] = {"value": disp_s, **src}
    out["accept_timing.displacement.fh"] = {"value": disp_f, **src}

    # accept-timing value share (oracle - flat) from best-response module, for context
    adv = json.load(open("out/sellerD_adaptation_value.json"))
    out["accept_timing.value_share"] = {"value": {
        "natural_oracle_minus_flat": adv["horizons"]["natural_T8"]["oracle"]["s_oracle_minus_s_flat"],
        "forced_oracle_minus_flat": adv["horizons"]["forced_k4"]["oracle"]["s_oracle_minus_s_flat"]},
        "source_script": "sellerD_adaptation_value.py",
        "source_data": "out/sellerD_adaptation_value.json",
        "notes": "3.2-4.3 pt opponent-modeling/accept-timing value share (oracle-flat)",
        "repo_fingerprint": _prov.FINGERPRINT}

    verdict = "null" if (span_s and span_f) else "adaptation"
    out["accept_timing.verdict"] = {"value": verdict,
        "firing_cells": {"seller": fires_s, "fh": fires_f},
        "notes": "null iff every per-ckpt hardness CI vs seller D spans 0 under "
                 "both ops, both horizons", **src}
    _emit.merge(out)
    cur = _emit.load_all()
    cur.setdefault("verdicts", {})["accept_timing"] = verdict
    json.dump(cur, open(_emit.OUT, "w"), indent=1)

    print("\n--- seller-D hardness coef (natural horizon) ---")
    for cell, v in tbl_s.items():
        d = v["D"]
        print(f"  {cell:24} coef={d['coef']:+.3f} CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] "
              f"n_dp={d['n_dp']} acc={d['n_acc']} {'FIRES' if not d['spans0'] else ''}")
    print("\n--- seller-D hardness coef (forced horizon) ---")
    for cell, v in tbl_f.items():
        d = v["D"]
        print(f"  {cell:24} coef={d['coef']:+.3f} CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] "
              f"n_dp={d['n_dp']} acc={d['n_acc']} {'FIRES' if not d['spans0'] else ''}")
    print("\nVERDICT:", verdict, "| firing seller:", fires_s, "| firing fh:", fires_f)
    print("displacement (natural, seller D):", disp_s)
    return out


if __name__ == "__main__":
    run()
