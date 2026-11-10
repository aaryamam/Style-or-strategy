"""Generate drop-in booktabs LaTeX tabular fragments from revision_numbers.json.
Writes analysis/revision/tables/{accept_timing,table2_extension_rows,
strict_locus,sign_consistency}.tex. Robust to missing keys (emits a % comment).
"""
import os, sys, json
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "analysis", "revision"))
os.chdir(_ROOT)
J = "analysis/revision/revision_numbers.json"
TDIR = "analysis/revision/tables"


def load():
    return json.load(open(J))


def f(x, n=3):
    try:
        return f"{x:.{n}f}"
    except Exception:
        return "--"


def ci(c, n=3):
    if not c or c[0] != c[0]:
        return "--"
    return f"[{c[0]:.{n}f}, {c[1]:.{n}f}]"


def accept_timing(d):
    CK = ["base", "instruct", "z_base", "z_dpo", "t_base", "t_rlvr"]
    L = [r"% T1 accept-timing: seller-D hardness coefficient (logistic), per checkpoint x op x horizon",
         r"\begin{tabular}{llrrr}", r"\toprule",
         r"checkpoint & op & \multicolumn{1}{c}{$\beta_{\text{hard}}$ (nat.)} & "
         r"\multicolumn{1}{c}{95\% CI} & \multicolumn{1}{c}{$\beta$ (k=4)} \\",
         r"\midrule"]
    for ck in CK:
        for op, oplab in [("opa_hold_rate", "a"), ("opb_conc_granted", "b")]:
            kn = d.get(f"accept_timing.seller.{ck}.{op}.D")
            kf = d.get(f"accept_timing.fh.{ck}.{op}.D")
            if not kn:
                continue
            cn = kn["value"]["coef"]; cif = kn.get("ci")
            cf = kf["value"]["coef"] if kf else float("nan")
            star = r"$^{*}$" if ("spans0=False" in (kn.get("notes") or "")) else ""
            fstar = r"$^{*}$" if (kf and "spans0=False" in (kf.get("notes") or "")) else ""
            cklab = ck.replace("_", r"\_")
            L.append(f"{cklab} & {oplab} & {f(cn)}{star} & {ci(cif)} & {f(cf)}{fstar} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table2_ext(d):
    rows = [("olmo\\_sft", "override.olmo_sft"), ("olmo\\_dpo", "override.olmo_dpo"),
            ("z\\_sft", "override.z_sft")]
    L = [r"% T4 Table-2 extension rows (intermediate checkpoints)",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"checkpoint & override & rescue & ovr$|$aggr (n) & ovr$|$pass (n) & "
         r"round-1 (n) & N pair/ctx/fp \\", r"\midrule"]
    for lab, key in rows:
        r = d.get(key, {}).get("value")
        if not isinstance(r, dict):
            L.append(f"{lab} & \\multicolumn{{6}}{{l}}{{-- (no free-decode run)}} \\\\")
            continue
        L.append(f"{lab} & {f(r['override'])} & {f(r['rescue'])} & "
                 f"{f(r['override_agg'])} ({r['agg_n']}) & "
                 f"{f(r['override_pass'])} ({r['pas_n']}) & "
                 f"{f(r['r1_override'])} ({r['r1_n']}) & "
                 f"{r['n_paired']}/{r['n_ctx']}/{r['n_free_parsed']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def strict_locus(d):
    L = [r"% T5 strict-regime stage attribution (N=500)",
         r"\begin{tabular}{llrr}", r"\toprule",
         r"family & step & \multicolumn{1}{c}{$D_{\text{fam}}$ (strict)} & "
         r"\multicolumn{1}{c}{95\% CI} \\", r"\midrule"]
    order = [("OLMo", ["base->sft", "sft->dpo", "dpo->instruct"]),
             ("Zephyr", ["z_base->z_sft", "z_sft->z_dpo"]),
             ("Tulu", ["t_base->t_rlvr"])]
    for fam, steps in order:
        for st in steps:
            k = d.get(f"strict_locus.{fam}.{st}")
            if not k:
                continue
            steplab = st.replace("_", r"\_").replace("->", r"$\to$")
            L.append(f"{fam} & {steplab} & "
                     f"{f(k['value']['d'])} & {ci(k.get('ci'))} \\\\")
        L.append(r"\midrule")
    if L[-1] == r"\midrule":
        L[-1] = r"\bottomrule"
    else:
        L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def sign_consistency(d):
    models = [("base", "OLMo"), ("z_base", "Mistral"), ("t_base", "Llama")]
    sets = [("i_all_constrained", "all-constrained"),
            ("ii_excl_override", "excl-override"),
            ("iii_free_parsed", "free-parsed$^{\\dagger}$")]
    L = [r"% T6 sign-consistency across turn sets: forced-horizon beta(D)",
         r"\begin{tabular}{lll rr r}", r"\toprule",
         r"model & op & set & \multicolumn{1}{c}{$\beta(D)$} & "
         r"\multicolumn{1}{c}{95\% CI} & sign \\", r"\midrule"]
    for ck, name in models:
        for op, oplab in [("opa_hold_rate", "a"), ("opb_conc_granted", "b")]:
            for sk, slab in sets:
                k = d.get(f"sign_consistency.{ck}.{op}.{sk}")
                if not k:
                    continue
                s = k["value"]["slope"]
                sign = "+" if s > 0 else ("$-$" if s < 0 else "0")
                L.append(f"{name} & {oplab} & {slab} & {f(s)} & {ci(k.get('ci'))} & {sign} \\\\")
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"% $^{\dagger}$ free-parsed column carries Defect-B selection bias."]
    return "\n".join(L)


def accept_timing_value(d):
    CK = [("base", "OLMo base"), ("instruct", "OLMo instruct"),
          ("z_base", "Mistral base"), ("z_dpo", "Zephyr DPO"),
          ("t_base", "Llama base"), ("t_rlvr", "Tulu RLVR")]
    gr = d.get("accept_timing_value.d_gradient", {}).get("value", {})
    ga = gr.get("forced_k4.opa_hold_rate", {}); gb = gr.get("forced_k4.opb_conc_granted", {})
    L = [r"% T12 accept-timing surplus value vs seller D (k=4; fitted-rule delta)",
         r"% D accept-timing best-response gain over blind: op-a "
         f"{f(ga.get('local_grad',0),4)}, op-b {f(gb.get('local_grad',0),4)} pts (both null).",
         r"\begin{tabular}{ll rr r}", r"\toprule",
         r"checkpoint & op & \multicolumn{1}{c}{$\Delta S$ (pts)} & "
         r"\multicolumn{1}{c}{CI-endpoint range} & alignment \\",
         r"\midrule"]
    for ck, lab in CK:
        for op, oplab in [("opa_hold_rate", "a"), ("opb_conc_granted", "b")]:
            k = d.get(f"accept_timing_value.forced_k4.{ck}.{op}")
            if not k:
                continue
            val = k["value"]
            rng = f"[{f(val['delta_at_ci_lo'])}, {f(val['delta_at_ci_hi'])}]"
            al = val["alignment"].replace("_", r"\_")
            L.append(f"{lab} & {oplab} & {f(val['surplus_delta'])} & {rng} & {al} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"% Verdict: accept-timing gradient NULL vs D (0.00 pt best-response value)"
          r" -> direction indeterminate; fitted sensitivity surplus-inert (op-a $\le$"
          r"+0.19 pt) to mildly maladaptive (op-b)."]
    return "\n".join(L)


def main():
    os.makedirs(TDIR, exist_ok=True)
    d = load()
    for name, fn in [("accept_timing", accept_timing),
                     ("table2_extension_rows", table2_ext),
                     ("strict_locus", strict_locus),
                     ("sign_consistency", sign_consistency),
                     ("accept_timing_value", accept_timing_value)]:
        open(f"{TDIR}/{name}.tex", "w").write(fn(d) + "\n")
        print("wrote", f"{TDIR}/{name}.tex")


if __name__ == "__main__":
    main()
