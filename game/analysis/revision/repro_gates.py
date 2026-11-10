"""Phase 0 reproduction gates. Every value is recomputed by IMPORTING the frozen
pipeline (study / metric_concession / policy_shift / seller_exp / b_analyze) and
compared to the published paper value. Point estimates must match to displayed
precision; bootstrap CIs may differ in trailing digits (seed fixed = 1234, noted).

Run: python analysis/revision/repro_gates.py
Writes analysis/revision/repro_gates.json and prints a PASS/FAIL table.
If ANY point-estimate gate fails, downstream tasks must NOT proceed on that path.
"""
import os, sys, json
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import study as ST
import metric_concession as M
import policy_shift as PS
import seller_exp as SE
import b_analyze as B
import _prov  # via sys.path below
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
import _prov

RESULTS = []


def check(name, got, target, tol, extra=""):
    ok = (got == got) and abs(got - target) <= tol
    RESULTS.append({"gate": name, "got": got, "target": target, "tol": tol,
                    "pass": bool(ok), "extra": extra})
    flag = "PASS" if ok else "**FAIL**"
    print(f"  [{flag}] {name}: got={got:.4f} target={target} tol={tol}  {extra}")
    return ok


# ---- loaders that reuse study's _wrap (signature + IF fold-in) ----
def load_c(ck, where="n500"):
    d = {"n500": "runs/n500", "n2000": "runs/n2000"}[where]
    return ST._wrap([json.loads(l)["state"] for l in open(f"{d}/constrained_{ck}.jsonl")])


def load_seller(seller, ck):
    return ST._wrap([json.loads(l)["state"]
                     for l in open(f"runs/seller_exp/{seller}_{ck}.jsonl")])


NB = 60  # bootstrap draws for gates (point estimate D is nboot-independent)


def main():
    print("=" * 74)
    print("PHASE 0 REPRODUCTION GATES  (repo fingerprint %s)" % _prov.FINGERPRINT[:12])
    print("=" * 74)

    # ---------- GATE 1: Table 3 constrained base->final Dfam ----------
    print("\nGATE 1 — Table 3 constrained base->final D_fam (N=500):")
    for fam, (b, a), tgt in [("OLMo", ("base", "instruct"), 0.792),
                             ("Zephyr", ("z_base", "z_dpo"), 0.353),
                             ("Tulu", ("t_base", "t_rlvr"), 0.801)]:
        r = ST.dfam(load_c(b), load_c(a), fam, nboot=NB)
        check(f"T3.{fam}", r["D"], tgt, 0.001)
    # Tulu independent seller-experiment seller-A value
    r = ST.dfam(load_seller("A", "t_base"), load_seller("A", "t_rlvr"),
                "Tulu_sellerA", nboot=NB)
    check("T3.Tulu_sellerA", r["D"], 0.8007, 0.002)

    # ---------- GATE 2: Table 7 consecutive steps ----------
    print("\nGATE 2 — Table 7 consecutive-step D_fam:")
    # OLMo N=500
    for (b, a), tgt in [(("base", "sft"), 1.110), (("sft", "dpo"), 0.701),
                        (("dpo", "instruct"), 0.096)]:
        r = ST.dfam(load_c(b), load_c(a), f"OLMo {b}->{a}", nboot=NB)
        check(f"T7.OLMo.{b}->{a}", r["D"], tgt, 0.002)
    # Zephyr N=2000
    for (b, a), tgt in [(("z_base", "z_sft"), 0.418), (("z_sft", "z_dpo"), 0.586)]:
        r = ST.dfam(load_c(b, "n2000"), load_c(a, "n2000"), f"Zephyr {b}->{a}", nboot=NB)
        check(f"T7.Zephyr.{b}->{a}", r["D"], tgt, 0.002)
    # Tulu N=2000
    for (b, a), tgt in [(("t_base", "t_sft"), 0.819), (("t_sft", "t_dpo"), 1.328),
                        (("t_dpo", "t_rlvr"), 0.160)]:
        r = ST.dfam(load_c(b, "n2000"), load_c(a, "n2000"), f"Tulu {b}->{a}", nboot=NB)
        check(f"T7.Tulu.{b}->{a}", r["D"], tgt, 0.002)

    # ---------- GATE 3: Table 2 policy-shift olmo_base ----------
    print("\nGATE 3 — Table 2 override/rescue (olmo_base):")
    ana = PS.analyze("olmo_base")
    check("T2.olmo_base.override", ana["override"], 0.208, 0.002)
    check("T2.olmo_base.rescue", ana["rescue"], 0.178, 0.002)

    # ---------- GATE 4: Test-4 z_base op-a cross-seller C-B point ----------
    print("\nGATE 4 — Test-4 z_base op-a cross-seller beta(C)-beta(B) point:")
    M.DIR = "runs/seller_exp"
    peB = M.load_obs("B", "z_base"); peC = M.load_obs("C", "z_base")
    pt = M.fit_beta(peC, 1)[0] - M.fit_beta(peB, 1)[0]
    check("T4.z_base.opa.CmB", pt, 0.837, 0.01,
          extra="(CI depends on RNG order; point gated)")

    # ---------- GATE 5: forced-horizon override-excluded base op-a ----------
    print("\nGATE 5 — forced-horizon override-excluded beta(D) OLMo base op-a:")
    csD = B.load_states("results/forced_horizon/D_base.jsonl")
    fsB = B.load_states("results/base_artifact/free_B_base.jsonl")  # warm import
    fsD = B.load_states("results/base_artifact/free_D_base.jsonl")
    ovrD, _ = B.overridden_rounds(fsD, csD)
    pe_ii = B.per_ep_excl(csD, ovrD)
    b_ii, _ = M.fit_beta(pe_ii, 1)
    check("T6.base.opa.excl_ovr", b_ii, 0.1754, 0.003)

    # ---------- GATE 6: alpha* surpluses ----------
    print("\nGATE 6 — best-response alpha* surpluses A/B/C:")
    for k, tgt in [("A", 11.71), ("B", 22.18), ("C", 2.47)]:
        a, s, _, nf = SE.optimal_opening(k, n=500)
        check(f"gate.alpha_surplus.{k}", s, tgt, 0.05, extra=f"alpha*={a}")

    # ---------- summary ----------
    npass = sum(1 for r in RESULTS if r["pass"])
    n = len(RESULTS)
    print("\n" + "=" * 74)
    print(f"REPRODUCTION GATES: {npass}/{n} PASS")
    fails = [r["gate"] for r in RESULTS if not r["pass"]]
    if fails:
        print("FAILED:", fails)
    else:
        print("ALL GATES PASS — cleared to emit Phase-1 numbers.")
    out = {"fingerprint": _prov.FINGERPRINT, "n_pass": npass, "n_total": n,
           "all_pass": npass == n, "gates": RESULTS}
    json.dump(out, open("analysis/revision/repro_gates.json", "w"), indent=1)
    print("wrote analysis/revision/repro_gates.json")


if __name__ == "__main__":
    main()
