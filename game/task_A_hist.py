"""TASK A (step 1, CPU) — robustness history pool for A1/A2. free-decode passA
ONLY. Reuses task2_build_histories.render() verbatim. The decisive passA late
universe is small (34 at round>=6 & eff_room>=15), so 3x MORE histories is
infeasible; instead we keep ALL passA round>=5 & eff_room>=10 late histories (72;
covers every A2 eff_room band {>=10,15-29,>=20} and both late cutoffs {5,6}) and
raise gens + add a 2nd seed downstream to lift effective N. early=90 baseline.
Saves results/realized_robust/hist_pool.json."""
import json, os, random
from collections import Counter
from task2_build_histories import render

random.seed(20240607)
PASSA = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]


def main():
    early, late, seen = [], [], set()
    for fk in PASSA:
        p = f"runs/n500/passA_{fk}.jsonl"
        if not os.path.exists(p):
            continue
        for l in open(p):
            st = json.loads(l); ro = st.get("eff_room", 0)
            for r in sorted(set(t["round"] for t in st["turns"])):
                h = render(st, r)
                if not h or h in seen:
                    continue
                seen.add(h)
                rec = {"history": h, "eff_room": ro, "round": r, "src": f"passA_{fk}"}
                if r <= 2:
                    early.append(rec)
                elif r >= 5 and ro >= 10:        # wide decisive-capable late pool
                    late.append(rec)
    random.shuffle(early)
    chosen = {"early": early[:90],
              "late": sorted(late, key=lambda x: (-x["eff_room"], -x["round"]))}
    os.makedirs("results/realized_robust", exist_ok=True)
    json.dump(chosen, open("results/realized_robust/hist_pool.json", "w"), indent=1)
    print("early:", len(chosen["early"]), "late(round>=5,eff_room>=10):", len(chosen["late"]))
    print(" late by source:", dict(Counter(x["src"] for x in chosen["late"])))
    L = chosen["late"]
    for cut in (5, 6):
        for band, lo, hi in [("ge10", 10, 999), ("15-29", 15, 29), ("ge20", 20, 999)]:
            print(f"  A1/A2 cell cutoff>={cut} eff_room[{band}]: "
                  f"{sum(1 for x in L if x['round']>=cut and lo<=x['eff_room']<=hi)}")


if __name__ == "__main__":
    main()
