"""T10 — RLVR forfeit cost: realized and latent.

Harness failure default delta (quoted from code): a strict parse-FAIL buyer turn
becomes a NO-OP (game_lib.extract_buyer_action -> strict_action=None; seller_exp/
freepass advance with effect 'noop', ask unchanged). The turn is forfeited; if it
is the final round the episode ends 'no_deal_timeout' -> captured surplus 0. So
delta = forfeited turn (no action credited); the episode-level default on a
forfeit-forced no-deal is zero captured surplus.

(a) Realized forfeit-regime cost, all four-condition checkpoints: strict-vs-
    constrained gap in captured surplus AND surplus_efficiency, per checkpoint —
    the deployed cost of a forfeit harness. captured surplus = v_b-deal_price on
    deals, 0 on no-deal (forfeit-forced), over feasible episodes.
(b) Latent counterfactual exposure, t_rlvr: at realized DECISIVE states from the
    Appendix-B probe (pool 'late' = turn>=6, top achievable room; t_rlvr's own
    reached=90 / decisive=6), the absolute parse-failure rate of FORCED AGGRESSIVE
    actions (hold_justified, offer_conditional), pooled over generation seeds
    1234+777, times achievable room (eff_room) at each state. Reported as EXPOSURE
    CONDITIONAL ON THE POLICY TAKING THE AGGRESSIVE ACTION (not realized cost).
    Contrast: same on olmo_base decisive states.
No model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import study as ST
import _prov, _emit

AGG = {"hold_justified", "offer_conditional"}
CKPTS = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]
FAM = {"OLMo": ("base", "instruct"), "Zephyr": ("z_base", "z_dpo"),
       "Tulu": ("t_base", "t_rlvr")}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def surplus_stats(recs):
    cap, eff = [], []
    for r in recs:
        s = r["state"]
        room = s["v_b"] - s["floor"]
        if room <= 0:
            continue
        if s["outcome"] == "deal" and s["deal_price"] is not None:
            c = s["v_b"] - s["deal_price"]
        else:
            c = 0.0
        cap.append(c); eff.append(c / room)
    return mean(cap), mean(eff), len(cap)


def parse_record(ck):
    rows = [json.loads(l) for l in open(f"runs/n500/passA_{ck}.jsonl")]
    tot = sum(len(r["turns"]) for r in rows)
    ok = sum(1 for r in rows for t in r["turns"] if t["parse_success"])
    return ok, tot


def realized():
    res = {}
    for ck in CKPTS:
        strict = ST.cond_records(ck, "strict")
        constr = ST.load_constrained(ck)
        s_cap, s_eff, n = surplus_stats(strict)
        c_cap, c_eff, _ = surplus_stats(constr)
        ok, tot = parse_record(ck)
        res[ck] = {"strict_surplus": s_cap, "constrained_surplus": c_cap,
                   "gap_surplus": c_cap - s_cap,
                   "strict_eff": s_eff, "constrained_eff": c_eff,
                   "gap_eff": c_eff - s_eff, "n_feasible": n,
                   "parse_record": f"{ok}/{tot}",
                   "parse_fail_turns": tot - ok}
    return res


def load_probe(*names):
    rows = []
    for n in names:
        p = f"results/realized_robust/{n}.jsonl"
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return rows


def exposure(rows, pool):
    """abs parse-fail rate of forced-aggressive actions in `pool`, and
    room-weighted exposure = mean/total of (1-parse_success)*eff_room."""
    agg = [r for r in rows if r["action_type"] in AGG and r["pool"] == pool]
    if not agg:
        return None
    fails = [0 if r["parse_success"] else 1 for r in agg]
    exp = [(0 if r["parse_success"] else 1) * r["eff_room"] for r in agg]
    return {"n_aggressive_gen": len(agg),
            "abs_parse_fail_rate": mean(fails),
            "mean_eff_room": mean([r["eff_room"] for r in agg]),
            "mean_exposure_room": mean(exp),
            "total_exposure_room": sum(exp)}


def run():
    out = {}
    src = {"source_script": "analysis/revision/t10_rlvr_forfeit.py",
           "repo_fingerprint": _prov.FINGERPRINT,
           "notes": "delta = forfeited (no-op) turn; no-deal default captured surplus 0"}

    # (a) realized
    rz = realized()
    for ck, d in rz.items():
        out[f"rlvr_realized.{ck}"] = {
            "value": {"strict_surplus": d["strict_surplus"],
                      "constrained_surplus": d["constrained_surplus"],
                      "gap_surplus": d["gap_surplus"],
                      "strict_eff": d["strict_eff"],
                      "constrained_eff": d["constrained_eff"],
                      "gap_eff": d["gap_eff"]},
            "n": d["n_feasible"],
            "source_data": f"runs/n500/{{passA,constrained}}_{ck}.jsonl",
            "notes": f"parse_record {d['parse_record']}", **src}
    # headline RLVR keys (t_rlvr)
    t = rz["t_rlvr"]
    out["pnRLVRSurplus"] = {"value": t["gap_surplus"], "n": t["n_feasible"],
        "source_data": "runs/n500/{passA,constrained}_t_rlvr.jsonl",
        "notes": f"strict-vs-constrained captured-surplus gap; parse {t['parse_record']}",
        **src}
    out["pnRLVRPct"] = {"value": (t["gap_surplus"] / t["constrained_surplus"] * 100
                                  if t["constrained_surplus"] else 0.0),
        "notes": "gap as % of constrained captured surplus", **src}
    base = rz["base"]
    out["pnRLVRBaselinePct"] = {"value": (base["gap_surplus"] / base["constrained_surplus"] * 100
                                          if base["constrained_surplus"] else 0.0),
        "notes": "OLMo base forfeit cost as % of constrained surplus (contrast)", **src}

    # (b) latent exposure — t_rlvr (shared decisive, both seeds) + own; contrast olmo_base
    t_shared = load_probe("probe_t_rlvr_seed1234", "probe_t_rlvr_seed777")
    t_own = load_probe("probe_t_rlvr_own_seed1234")
    o_shared = load_probe("probe_olmo_base_seed1234", "probe_olmo_base_seed777")
    latent = {
        "t_rlvr_shared_decisive_pooledseeds": exposure(t_shared, "late"),
        "t_rlvr_own_decisive": exposure(t_own, "late"),
        "t_rlvr_own_reached": exposure(t_own, "late_loose"),
        "olmo_base_shared_decisive_pooledseeds": exposure(o_shared, "late"),
    }
    out["rlvr_latent_exposure"] = {"value": latent,
        "source_data": "results/realized_robust/probe_{t_rlvr,olmo_base}_*.jsonl",
        "seed": "1234+777",
        "notes": ("EXPOSURE CONDITIONAL ON THE POLICY TAKING THE AGGRESSIVE ACTION "
                  "(hold_justified/offer_conditional); NOT realized cost. "
                  "room-weighted = parse_fail x eff_room."), **src}
    _emit.merge(out)

    print("T10(a) realized forfeit cost (strict vs constrained captured surplus):")
    print(f"  {'ckpt':10} {'strict':>8} {'constr':>8} {'gap':>8} "
          f"{'s_eff':>7} {'c_eff':>7} {'d_eff':>7}  parse")
    for ck, d in rz.items():
        print(f"  {ck:10} {d['strict_surplus']:8.3f} {d['constrained_surplus']:8.3f} "
              f"{d['gap_surplus']:8.3f} {d['strict_eff']:7.3f} {d['constrained_eff']:7.3f} "
              f"{d['gap_eff']:7.3f}  {d['parse_record']}")
    print("\nT10(b) latent counterfactual exposure (aggressive-forced, decisive):")
    for k, v in latent.items():
        if v is None:
            print(f"  {k}: (no rows)"); continue
        print(f"  {k}: n={v['n_aggressive_gen']} fail_rate={v['abs_parse_fail_rate']:.3f} "
              f"mean_room={v['mean_eff_room']:.2f} mean_exp={v['mean_exposure_room']:.3f} "
              f"total_exp={v['total_exposure_room']:.1f}")
    return out


if __name__ == "__main__":
    run()
