"""TASK 2 (step 3, CPU) — firing analysis on the realized-history probe.
Same firing condition as the frozen probe (probe_zt_analyze): eta = qF(agg)-
qF(pass); agg={hold_justified,offer_conditional}, pass={accept,offer_simple};
fire iff marginal eta<0 AND Deta=eta(late)-eta(early) <0 with CI excluding 0.
Side-by-side with the original synthetic-bin firing_test.json."""
import json, random
random.seed(1234)
AGG = {"hold_justified", "offer_conditional"}
PASS = {"accept", "offer_simple"}
CKPTS = ["olmo_base", "olmo_inst", "z_base", "z_dpo", "t_base", "t_rlvr"]
BINS = ["early", "mid", "late"]


def mean(b):
    return sum(b) / len(b) if b else float("nan")


def load(ck):
    import os
    p = f"results/realized_probe/realized_{ck}.jsonl"
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def bits(rows, binname, bucket):
    s = AGG if bucket == "agg" else PASS
    return [1 if r["parse_success"] else 0 for r in rows
            if r["history_bin"] == binname and r["action_type"] in s]


def boot_delta(ae, pe, al, pl, n=4000):
    out = []
    for _ in range(n):
        def m(x):
            return mean([x[random.randrange(len(x))] for _ in x]) if x else float("nan")
        out.append((m(al) - m(pl)) - (m(ae) - m(pe)))
    out.sort()
    return out[int(.025*n)], out[int(.975*n)]


def main():
    orig = json.load(open("out/firing_test.json"))["checkpoints"]
    out = ["# TASK 2 — realized-history firing probe\n",
           "Shared realized histories (results/realized_probe/histories.json; 30/bin, "
           "late = decisive eff_room 15-29), pooled from realized passA + constrained "
           "trajectories (forced by scarcity: olmo_inst has 0 realized late contexts). "
           "Probe harness, action set, parser, and firing condition reused verbatim.\n",
           "Checkpoint-set consistency note: the firing Table-4 has 6 rows; the "
           "probe_zt.py re-run script covers 4 (Zephyr/Tulu base+final) and OLMo "
           "base+inst were probed separately (probe_runs.jsonl) — hence the draft's "
           "'four free-decode checkpoints' vs Table-4's six. This realized re-run "
           "covers all 6.\n"]
    out.append("| checkpoint | marg_eta | eta@early | eta@mid | eta@late | "
               "Deta | Deta 95% CI | verdict | (orig synthetic Deta / verdict) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    verdicts = {}
    for ck in CKPTS:
        rows = load(ck)
        if rows is None:
            out.append(f"| {ck} | MISSING (no realized run) | | | | | | | |")
            verdicts[ck] = "MISSING"
            continue
        eta = {}
        for b in BINS:
            eta[b] = mean(bits(rows, b, "agg")) - mean(bits(rows, b, "pass"))
        agg_all = [x for b in BINS for x in bits(rows, b, "agg")]
        pas_all = [x for b in BINS for x in bits(rows, b, "pass")]
        marg = mean(agg_all) - mean(pas_all)
        deta = eta["late"] - eta["early"]
        dlo, dhi = boot_delta(bits(rows, "early", "agg"), bits(rows, "early", "pass"),
                              bits(rows, "late", "agg"), bits(rows, "late", "pass"))
        if marg >= 0:
            v = "wrong-direction"
        elif dhi < 0:
            v = "**FIRES**"
        elif dlo > 0:
            v = "inverted"
        else:
            v = "flat"
        verdicts[ck] = v
        o = orig.get(ck, {})
        od = o.get("delta", "NA"); ov = o.get("verdict", "NA")
        out.append(f"| {ck} | {marg!r} | {eta['early']!r} | {eta['mid']!r} | "
                   f"{eta['late']!r} | {deta!r} | [{dlo!r}, {dhi!r}] | {v} | "
                   f"({od} / {ov}) |")
    fires = [ck for ck, v in verdicts.items() if "FIRES" in str(v)]
    out.append(f"\n**Bottom line:** "
               + ("kernel STILL ABSENT on realized high-stakes histories for all "
                  "checkpoints (no fire)." if not fires else
                  f"FIRES on: {fires} — FINDING, investigate."))
    open("results/realized_probe/task2_report.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
