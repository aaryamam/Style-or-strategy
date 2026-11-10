"""Single runner for the TMLR revision analysis pass.
Rebuilds provenance, runs the Phase-0 reproduction gates (aborts if any fail),
then executes every Phase-1 task module (T1-T10), merging into
analysis/revision/revision_numbers.json, and finally regenerates the LaTeX table
fragments. Zero model inference. CPU-only.

Usage:  python analysis/revision/run_all.py [--skip-gates] [--only T1,T3,...]
"""
import os, sys, json, argparse, importlib
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)

TASKS = ["t1_accept_timing", "t2_mde_surplus", "t3_permutation",
         "t4_override_intermediate", "t5_strict_attribution",
         "t6_sign_consistency", "t7_centered_if", "t8_olmo_gates",
         "t9_lenient", "t10_rlvr_forfeit", "t12_accept_timing_value"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    # provenance
    import _prov
    json.dump(_prov.STATE, open("analysis/revision/repo_state.json", "w"), indent=1)
    print("repo fingerprint:", _prov.FINGERPRINT[:16])

    if not args.skip_gates:
        import repro_gates
        repro_gates.main()
        g = json.load(open("analysis/revision/repro_gates.json"))
        if not g["all_pass"]:
            print("REPRODUCTION GATES FAILED — aborting Phase 1.")
            sys.exit(1)

    only = set(t.lower() for t in args.only.split(",") if t)
    for mod in TASKS:
        tag = mod.split("_")[0]           # t1, t2, ...
        if only and tag not in only and mod not in only:
            continue
        print("\n" + "=" * 74 + f"\nRUN {mod}\n" + "=" * 74)
        m = importlib.import_module(mod)
        m.run()

    # rebuild verdicts block (order-independent) + regenerate tables
    import _reconcile_verdicts
    _reconcile_verdicts.run()
    import make_tables
    make_tables.main()
    print("\nAll tasks complete. See analysis/revision/revision_numbers.json, "
          "REVISION_RESULTS.md, tables/.")


if __name__ == "__main__":
    main()
