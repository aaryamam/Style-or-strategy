"""Rebuild the top-level `verdicts` block from the canonical task keys, so it is
independent of task run-order (guards against a merge clobbering it). Idempotent."""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
import _emit


def run():
    d = _emit.load_all()
    v = d.get("verdicts", {})
    if "accept_timing.verdict" in d:
        v["accept_timing"] = d["accept_timing.verdict"]["value"]
    if "pnMDEsurplus" in d:
        v["mde_surplus_binding_pts"] = d["pnMDEsurplus"]["value"]
    if "pnCenteredLocus" in d:
        v["centered_if"] = d["pnCenteredLocus"]["verdict"]
    if "pnStrictLocusVerdict" in d:
        for fam, sv in d["pnStrictLocusVerdict"]["value"].items():
            v[f"strict_locus.{fam}"] = sv.get("ordering") or "not_computable"
    if "sign_consistency.verdicts" in d:
        v["sign_consistency"] = ("consistent"
            if d["sign_consistency.verdicts"].get("all_consistent") else "mixed")
    if "pnLenientCounts" in d:
        tot = d["pnLenientCounts"]["value"]["total"]
        v["lenient"] = "vacuous" if tot <= 5 else "substantial"
    if "accept_timing_value.verdict" in d:
        v["accept_timing_direction"] = d["accept_timing_value.verdict"]["value"]
    d["verdicts"] = v
    json.dump(d, open(_emit.OUT, "w"), indent=1)
    print("reconciled verdicts:", json.dumps(v))
    return v


if __name__ == "__main__":
    run()
