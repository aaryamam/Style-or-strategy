# Consolidated verification report — Tasks 0–3

*Read-only forensic (Task 0) + two experiments (Tasks 1–2) + one robustness computation (Task 3).
No frozen file (paper_data.md, paper/main.tex, existing result files) was modified. All new output
under results/. Models run sequentially on one A6000. Full precision. No paper edits, no fixes.*

Per-task detail files: `results/integrity_check/task0_forensic.txt`,
`results/forced_horizon/task1_report.md`, `results/realized_probe/task2_report.md`,
`results/loco_locus/task3_loco_locus.txt`.

---

## TASK 0 — the Tülu-0.801 coincidence (forensic)

Seller-A constrained D_fam recomputed via `study.dfam` (identical methodology, deterministic point
estimate) on the persisted seller-experiment generations vs the main runs:

| family | seller-A D_fam (full precision) | main D_fam (Table 3) | abs diff |
|---|---|---|---|
| OLMo | 0.8401585406838912 | 0.7923492407181946 | 0.047809299965696606 |
| Zephyr | 0.4056221140314698 | 0.3527084811133639 | 0.05291363291810591 |
| **Tülu** | **0.8006578759723876** | **0.8013268156995512** | **0.0006689397271636421** |

Source: `seller_analyze.py::test3` → `study.dfam` over `runs/seller_exp/A_{base,final}.jsonl`; the
value was printed to stdout (not persisted to JSON). Generation-identity check: the seller-A runs
are NOT a file copy of the main constrained runs (round-1 raw identical in only 487–492/500
episodes, not 500/500) — the `seller_exp.stub_schedule` uses a `+350` seed offset vs
`game_lib.stubborn_schedule`, plus vLLM batch-numerics, so they are independent generation passes.

**BOTTOM LINE (Task 0): INDEPENDENTLY COMPUTED, not copied.** Tülu seller-A = `0.8006578759723876`,
main = `0.8013268156995512` — differ by `0.000669`, both *round* to 0.801 (a 3-decimal rounding
coincidence). Tülu does NOT share the +0.05 offset seen on OLMo (+0.0478) and Zephyr (+0.0529); its
two independent realizations happen to land ~0.80 in both. The expectation of ~0.83–0.85 for Tülu
is NOT borne out — it is ~0.80.

---

## TASK 1 — forced-minimum-horizon concession test (k_min=4)

Additive default-off flag `min_accept_round` (frozen behavior bit-identical at default=1); seller's
accept-of-offer suppressed for buyer-offer rounds < 4 (open/concession/hold/walk unchanged). Ran the
FULL 6 checkpoints × 4 sellers, N=500, constrained, same seeds. Fed into the FROZEN Test-4 regression
(`metric_concession.py`) unchanged.

**Power achieved** (usable obs vs adaptive seller D): base 63→703, instruct 30→83, z_base 96→498,
z_dpo 96→328, t_base 197→401, t_rlvr 524→777. **All originally-starved cells now powered (≥50 obs).**

**β against adaptive seller D** (full precision in the detail file):

| ckpt | op-a hold_rate β(D) [95% CI] | non-null? | op-b conc_granted β(D) [95% CI] | non-null? |
|---|---|---|---|---|
| base | +0.18061 [+0.11415, +0.23920] | **YES** | −0.97952 [−1.14072, −0.82088] | **YES** |
| instruct | +0.03377 [−0.04661, +0.11313] | no | −0.22248 [−0.68976, +0.27827] | no |
| z_base | +0.12997 [+0.03215, +0.22813] | **YES** | −0.91535 [−1.22056, −0.63185] | **YES** |
| z_dpo | +0.00981 [−0.02025, +0.03979] | no | −0.09983 [−0.28500, +0.09140] | no |
| t_base | +0.07802 [+0.01050, +0.14744] | **YES** | −0.60558 [−0.84214, −0.33636] | **YES** |
| t_rlvr | −0.01065 [−0.10279, +0.07533] | no | −0.21666 [−0.55578, +0.15067] | no |

Cross-seller β(C)−β(B), op-a: base, z_base, z_dpo, t_base all DIFFER (CI excl 0); instruct, t_rlvr no.
op-b: z_dpo and t_rlvr DIFFER.

**BOTTOM LINE (Task 1): POWERED NOW — yes. NON-NULL β — yes, but split by training stage.** The three
**BASE** models (base, z_base, t_base) show **non-null β(D) in BOTH operationalizations** — *flagged
loudly*. Direction: op-a positive (concede MORE as the seller is observed to hold more) — the
anti-adaptive / state-tracking direction, not "punish the stubborn seller." The three **aligned
FINALS** (Instruct, z_dpo, t_rlvr) **remain NULL β(D) even when powered** — the original
no-adaptation verdict on the finals survives the power increase. (Cross-seller concession differences
appear for base/z_base/z_dpo/t_base on op-a — a broader signal than the original 2 weak Zephyr cells.)

---

## TASK 2 — realized-history firing probe

Shared realized histories (30/bin; late = decisive, eff_room 15–29) pooled from realized passA +
constrained trajectories (forced by scarcity: olmo_inst has 0 realized late contexts). Probe harness,
action set, parser, firing condition reused verbatim. Ran all 6 checkpoints. Fire iff marginal η<0
AND Δη=η(late)−η(early) < 0 with CI excluding 0.

| ckpt | marg η | η@early | η@mid | η@late | Δη | Δη 95% CI | verdict | synthetic (orig) |
|---|---|---|---|---|---|---|---|---|
| **olmo_base** | −0.53667 | −0.47333 | −0.54667 | −0.59000 | **−0.11667** | **[−0.20667, −0.02000]** | **FIRES** | flat (−0.017) |
| olmo_inst | −0.00444 | −0.01333 | 0.00000 | 0.00000 | +0.01333 | [+0.00333, +0.02667] | inverted | inverted (0.46) |
| z_base | −0.29111 | −0.21000 | −0.42667 | −0.23667 | −0.02667 | [−0.09333, +0.04333] | flat | flat (0.033) |
| z_dpo | −0.47333 | −0.45000 | −0.47000 | −0.50000 | −0.05000 | [−0.13333, +0.03667] | flat | flat (−0.01) |
| t_base | −0.57000 | −0.58333 | −0.56333 | −0.56333 | +0.02000 | [−0.06000, +0.10000] | flat | flat (0.0) |
| **t_rlvr** | −0.43333 | −0.38000 | −0.42667 | −0.49333 | **−0.11333** | **[−0.19333, −0.03667]** | **FIRES** | flat (0.0) |

**BOTTOM LINE (Task 2): the kernel-absent verdict does NOT survive on realized high-stakes histories
for 2 of 6 checkpoints — FIRES on `olmo_base` and `t_rlvr` (flagged loudly).** Both show monotonic
widening of the aggressive−passive parse gap into the decisive late bin (olmo_base η −0.473→−0.590;
t_rlvr −0.380→−0.493), with Δη CIs excluding 0 — whereas the frozen synthetic-bin probe rated both
PARTIAL-flat. The other four remain flat/inverted (kernel absent), consistent with the frozen record.
Caveats for the reader's judgment (not resolved here): Δη magnitudes are modest (~−0.11), olmo_base's
CI upper bound is −0.020 (marginal), and the realized histories are a shared, pooled, decisive-late
set (different regime from the synthetic bins).

---

## TASK 3 — leave-one-component-out + Euclidean locus robustness (CPU)

Recomputed the stage-attribution locus (largest CI-separated consecutive step) for each family under
each of the 5 LOCO 4-component signatures and a Euclidean distance over the full 5, reusing
`study.dfam` (swapping the component set). NBOOT=2000; OLMo N=500, Zephyr/Tülu N=2000.

Locus per variant (all 7 variants × 3 families): **OLMo = supervised (base→sft), Zephyr = preference
(z_sft→z_dpo), Tülu = preference (t_sft→t_dpo) in EVERY variant.** The argmax step never flips.
CI-separation weakens in a few low-margin variants (Zephyr loses CI-sep under drop-surplus,
drop-walkaway, and Euclidean; OLMo loses CI-sep under drop-haggling), but the identified locus is
unchanged in all of them.

**BOTTOM LINE (Task 3): locus ordering INVARIANT** — supervised/preference/preference holds across
all 5 LOCO variants and Euclidean; no variant flips the locus.

---

## One-line verdicts

- **Task 0:** INDEPENDENT computation, not a copy; true Tülu seller-A D_fam = `0.8006578759723876`
  (main `0.8013268156995512`) — 3-decimal rounding coincidence; Tülu shows no +0.05 offset.
- **Task 1:** Powered now (yes, all starved cells ≥50 obs). Non-null β(D): YES on the three BASE
  models (both ops, anti-adaptive direction) — flagged; aligned FINALS remain NULL even powered.
- **Task 2:** Kernel-absent does NOT fully survive — FIRES on `olmo_base` and `t_rlvr` over realized
  decisive-late histories (flagged); other four flat/inverted.
- **Task 3:** Locus ordering INVARIANT across all 5 LOCO + Euclidean (no flips).
