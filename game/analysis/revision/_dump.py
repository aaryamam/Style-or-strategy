"""Readable dump of revision_numbers.json for report assembly / eyeballing."""
import json, sys
d = json.load(open("analysis/revision/revision_numbers.json"))


def g(k):
    return d.get(k, {})


print("=== VERDICTS ===")
print(json.dumps(d.get("verdicts", {}), indent=1))
for k in ["accept_timing.verdict", "pnMDEsurplus", "pnMDEpct",
          "pnPermOLMo", "pnPermZephyr", "pnPermTulu",
          "pnRLVRFloorCompareOLMo", "pnRLVRFloorCompareTulu",
          "pnRLVRSurplus", "pnRLVRPct", "pnRLVRBaselinePct",
          "pnBootstrapRecompute", "pnLenientCounts"]:
    v = g(k)
    print(f"\n--- {k} ---")
    print(json.dumps({kk: v[kk] for kk in v if kk not in ("repo_fingerprint",)},
                     indent=1)[:1200])
