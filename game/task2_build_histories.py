"""TASK 2 (step 1, CPU) — build a SHARED set of REALIZED histories for the
realized-history firing probe. Pools realized buyer-decision contexts from the
free-decode passA runs (primary) + constrained runs (to populate the scarce
late/decisive bin), bins by turn depth (early r1-2 / mid r3-5 / late r6-8),
prefers DECISIVE late histories (largest eff_room, deepest), renders them
seller-side to mirror the original probe's HISTORIES voice, and saves
results/realized_probe/histories.json. Deviation from per-checkpoint histories
is forced by scarcity (olmo_inst has 0 realized late contexts) and matches the
original probe's shared-history design — stated in the report.
"""
import json, os, random

random.seed(1234)
N_PER_BIN = 30
PASSA = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]
CONSTR_DIRS = ["runs/n500", "runs/n2000"]
CONSTR = ["base", "sft", "dpo", "instruct", "z_base", "z_sft", "z_dpo",
          "t_base", "t_sft", "t_dpo", "t_rlvr"]


def render(state, r):
    """Seller-side one-paragraph history as of the buyer's round-r decision."""
    turns = sorted(state["turns"], key=lambda t: t["round"])
    open_ask = turns[0]["ask_shown"]
    ask_r = next((t["ask_shown"] for t in turns if t["round"] == r), None)
    if ask_r is None:
        return None
    def act(t):
        return t.get("action", t.get("strict_action"))
    offers = [t["price"] for t in turns if t["round"] < r
              and act(t) == "offer" and t.get("price") is not None]
    last_offer = offers[-1] if offers else None
    if r <= 2:
        if last_offer is None:
            return (f"This is the opening of a price negotiation. You are asking "
                    f"{ask_r}. No counter-offer is on the table yet.")
        return (f"This is the opening of a price negotiation. You opened by asking "
                f"{open_ask}; the buyer's standing offer is {last_offer} and your "
                f"current ask is {ask_r}.")
    if r <= 5:
        return (f"The negotiation is underway, {r-1} rounds in. You opened at "
                f"{open_ask}; through the exchanges the buyer's standing offer is "
                f"now {last_offer} and your current ask is {ask_r}.")
    return (f"This is a late, high-stakes stage of a price negotiation, after "
            f"{r-1} rounds. You opened at {open_ask}; the buyer has pushed you "
            f"down over several exchanges to a standing offer of {last_offer}, and "
            f"your current ask is {ask_r}. The amount at stake is large.")


def collect():
    pools = {"early": [], "mid": [], "late": []}
    seen = set()

    def add(state, src):
        ro = state.get("eff_room", 0)
        rounds = sorted(set(t["round"] for t in state["turns"]))
        for r in rounds:
            b = "early" if r <= 2 else ("mid" if r <= 5 else "late")
            h = render(state, r)
            if not h:
                continue
            key = h
            if key in seen:
                continue
            seen.add(key)
            pools[b].append({"history": h, "eff_room": ro, "round": r, "src": src})

    for fk in PASSA:
        p = f"runs/n500/passA_{fk}.jsonl"
        if os.path.exists(p):
            for l in open(p):
                add(json.loads(l), f"passA_{fk}")
    for d in CONSTR_DIRS:
        for fk in CONSTR:
            p = f"{d}/constrained_{fk}.jsonl"
            if os.path.exists(p):
                for l in open(p):
                    add(json.loads(l)["state"], f"{d}/{fk}")
    return pools


def main():
    pools = collect()
    os.makedirs("results/realized_probe", exist_ok=True)
    chosen = {}
    for b in ["early", "mid", "late"]:
        cand = pools[b]
        if b == "late":
            cand = sorted(cand, key=lambda x: (-x["eff_room"], -x["round"]))  # decisive first
        else:
            random.shuffle(cand)
        chosen[b] = cand[:N_PER_BIN]
    json.dump(chosen, open("results/realized_probe/histories.json", "w"), indent=1)
    print("pool sizes:", {b: len(pools[b]) for b in pools})
    print("chosen per bin:", {b: len(chosen[b]) for b in chosen})
    for b in chosen:
        ro = [h["eff_room"] for h in chosen[b]]
        print(f"  {b}: eff_room min/max = {min(ro) if ro else 'NA'}/"
              f"{max(ro) if ro else 'NA'}; example: {chosen[b][0]['history'][:90]!r}")


if __name__ == "__main__":
    main()
