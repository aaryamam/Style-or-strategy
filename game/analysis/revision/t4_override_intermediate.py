"""T4 — Override/rescue rows for intermediate checkpoints.
For every intermediate checkpoint with an existing paired free(passA)/constrained
run in runs/n500, run the frozen policy-shift diagnostic (policy_shift.analyze)
and emit rows in exact Table-2 format. Then compute the per-step exposure summary:
override rate at each endpoint of every consecutive attribution step.

Available paired (passA + constrained) checkpoints in runs/n500:
  OLMo   : base, sft, dpo, instruct  (intermediates: olmo_sft, olmo_dpo)
  Zephyr : z_base, z_sft, z_dpo      (intermediate: z_sft)
  Tulu   : t_base, t_rlvr            (NO intermediate passA -> '-')
Reuses policy_shift.analyze verbatim (extends FILEMAP only). No model inference.
"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import policy_shift as PS
import _prov, _emit

# extend FILEMAP with intermediate checkpoints (file basenames in runs/n500)
PS.FILEMAP.update({"olmo_sft": "sft", "olmo_dpo": "dpo"})  # z_sft already maps via name
PS.FILEMAP.setdefault("z_sft", "z_sft")

INTERMEDIATES = ["olmo_sft", "olmo_dpo", "z_sft"]
# attribution steps by family, in order (ckpt display names)
STEPS = {
    "OLMo": [("olmo_base", "olmo_sft"), ("olmo_sft", "olmo_dpo"),
             ("olmo_dpo", "olmo_inst")],
    "Zephyr": [("z_base", "z_sft"), ("z_sft", "z_dpo")],
    "Tulu": [("t_base", "t_sft"), ("t_sft", "t_dpo"), ("t_dpo", "t_rlvr")],
}
# ensure the endpoints resolve; add t_sft/t_dpo as intentionally-absent
PS.FILEMAP.setdefault("t_sft", "t_sft")
PS.FILEMAP.setdefault("t_dpo", "t_dpo")


def row(ck):
    r = PS.analyze(ck)
    if "MISSING" in r:
        return None
    return {"override": r["override"], "rescue": r["rescue"],
            "override_agg": r["override_agg"], "agg_n": r["agg_n"],
            "override_pass": r["override_pass"], "pas_n": r["pas_n"],
            "r1_override": r["r1_override"], "r1_n": r["r1_n"],
            "n_paired": r["n_paired"], "n_ctx": r["n_ctx"],
            "n_free_parsed": r["n_free_parsed"], "tv": r["tv"]}


def run():
    out = {}
    src = {"source_script": "analysis/revision/t4_override_intermediate.py",
           "source_data": "runs/n500/{passA,constrained}_*.jsonl",
           "repo_fingerprint": _prov.FINGERPRINT,
           "notes": "policy_shift.analyze, context-matched grammar override"}
    rows = {}
    for ck in INTERMEDIATES:
        r = row(ck)
        rows[ck] = r
        if r is None:
            print(f"  {ck}: MISSING")
            out[f"override.{ck}"] = {"value": "-", "notes": "no passA free run",
                                     **src}
            continue
        out[f"override.{ck}"] = {"value": r, **src}
        print(f"  {ck}: override={r['override']:.3f} rescue={r['rescue']:.3f} "
              f"ovr|aggr={r['override_agg']:.3f}({r['agg_n']}) "
              f"ovr|pass={r['override_pass']:.3f}({r['pas_n']}) "
              f"r1={r['r1_override']:.3f}({r['r1_n']}) "
              f"N={r['n_paired']}/{r['n_ctx']}/{r['n_free_parsed']}")

    # endpoints that have paired runs -> override rate (reuse Table-2 base/final too)
    ALL_CK = ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_inst",
              "z_base", "z_sft", "z_dpo", "t_base", "t_rlvr"]
    ovr_rate = {}
    for ck in ALL_CK:
        r = PS.analyze(ck)
        ovr_rate[ck] = None if "MISSING" in r else r["override"]

    exposure = {}
    for fam, steps in STEPS.items():
        exposure[fam] = []
        for a, b in steps:
            exposure[fam].append({
                "step": f"{a}->{b}",
                "override_start": ovr_rate.get(a, None),
                "override_end": ovr_rate.get(b, None),
                "note": ("" if ovr_rate.get(a) is not None and ovr_rate.get(b) is not None
                         else "endpoint(s) lack passA free run (–)")})
    out["override.per_step_exposure"] = {"value": exposure, **src}
    _emit.merge(out)
    print("\nPer-step exposure (override rate at endpoints):")
    for fam, steps in exposure.items():
        for s in steps:
            print(f"  {fam:7} {s['step']:22} "
                  f"start={s['override_start']} end={s['override_end']} {s['note']}")
    return out


if __name__ == "__main__":
    run()
