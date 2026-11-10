"""TASK A3 (CPU) — OWN-history probe sets for the two fired checkpoints. Draws
late-decisive histories from that checkpoint's OWN realized trajectories
(constrained — the user's spec — plus its own passA, both labelled), and own
early baseline. If a checkpoint rarely reaches decisive-late on-policy, the count
bounds the claim. Saves results/realized_robust/own_<ckpt>.json."""
import json, os
from collections import Counter
from task2_build_histories import render

# own realized sources per fired checkpoint (free passA + own constrained)
SRC = {
    "olmo_base": [("runs/n500/passA_base.jsonl", "passA", False),
                  ("runs/n500/constrained_base.jsonl", "constr_n500", True)],
    "t_rlvr": [("runs/n500/passA_t_rlvr.jsonl", "passA", False),
               ("runs/n2000/constrained_t_rlvr.jsonl", "constr_n2000", True),
               ("runs/n500/constrained_t_rlvr.jsonl", "constr_n500", True)],
}


def build(ck):
    early, late, loose, seen = [], [], [], set()
    for path, lab, wrapped in SRC[ck]:
        if not os.path.exists(path):
            continue
        for l in open(path):
            o = json.loads(l)
            st = o["state"] if wrapped else o
            ro = st.get("eff_room", 0)
            for r in sorted(set(t["round"] for t in st["turns"])):
                h = render(st, r)
                if not h or h in seen:
                    continue
                seen.add(h)
                rec = {"history": h, "eff_room": ro, "round": r, "src": lab}
                if r <= 2:
                    early.append(rec)
                elif r >= 6:                     # own-reached late (any eff_room)
                    loose.append(rec)
                    if ro >= 15:                 # own-decisive late (== original def)
                        late.append(rec)
    early = early[:90]
    late = sorted(late, key=lambda x: (-x["eff_room"], -x["round"]))[:90]
    loose = sorted(loose, key=lambda x: (-x["eff_room"], -x["round"]))[:90]
    return {"early": early, "late": late, "late_loose": loose}


def main():
    os.makedirs("results/realized_robust", exist_ok=True)
    for ck in SRC:
        d = build(ck)
        json.dump(d, open(f"results/realized_robust/own_{ck}.json", "w"), indent=1)
        print(f"{ck}: own early={len(d['early'])} own late-DECISIVE={len(d['late'])} "
              f"(src={dict(Counter(x['src'] for x in d['late']))}) "
              f"own late-REACHED(any eff_room)={len(d['late_loose'])} "
              f"(src={dict(Counter(x['src'] for x in d['late_loose']))})")


if __name__ == "__main__":
    main()
