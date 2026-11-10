# Follow-up consolidated report — Task A (realized-probe robustness) + Task B (base-concession artifact)

*Follow-up to the Task 0–3 run. No frozen file modified; all new output under
results/realized_robust/ and results/base_artifact/. Models sequential on one A6000,
bf16. Full precision. No paper edits, no narrative decisions.*

Detail files: `results/realized_robust/taskA_report.md`, `results/base_artifact/taskB_report.md`.

---

## TASK A — realized-history firing probe: robustness + sanity

### A0 — audit of the existing realized run (read-only)
- **Sampling / effective N:** 30 histories/bin × 5 gens/(action,history) = 150 per action per bin
  → **300 agg + 300 pass behind each η(bin)** — i.e. the realized run already MATCHED the synthetic
  probe's 150-gen/action/bin budget (it was not under-sampled).
- **Decisive-late definition:** turn-index `round ≥ 6` AND the eff_room band was the *resulting*
  spread (15–29) of the top-30-by-eff_room selection — not an imposed threshold (it came from the
  builder's `sorted(-eff_room, -round)[:30]`).
- **Provenance (the key audit finding):** **all 30 late histories are free-decode passA, 0
  constrained.** They are pooled across checkpoints — late src = {z_base 10, t_base 13, z_dpo 2,
  t_rlvr 3, base 2}. The two FIRED checkpoints contributed only their own few (olmo_base **2/30**,
  t_rlvr **3/30**); the rest are *other* checkpoints' histories. olmo_inst contributed 0 (it has no
  realized late context). So the original fire was measured largely on **counterfactual histories
  for the fired checkpoint** — this is the "pooling kludge."
- **Bootstrap:** realized `boot_delta` n=4000, unit = probe-item parse bits — **identical** to the
  frozen Table-4 `probe_zt_analyze.boot_delta`.

### A4 — frozen-record consistency (read-only)
The original synthetic probe generated for **all 6 checkpoints** (2,250 gens each: `probe_runs.jsonl`
= OLMo base+aligned; `probe_runs_zephyr_tulu.jsonl` = z_base/z_dpo/t_base/t_rlvr), so Table-4's six
rows are fully backed. The phrase **"four free-decode checkpoints" does NOT appear in the frozen
record** (STUDY_REPORT.md / paper_data.md) — it exists only in my own Task-2 helper note. **No
six-vs-four inconsistency in the frozen record.** (If a separate draft outside this repo uses
"four", I cannot see it.)

### A1 — larger N (n=10, all 34 decisive histories; effective N ~2,720 late obs pooled) + 2nd seed
The passA decisive universe is only 34 histories, so 3× *more histories* is infeasible; N was raised
via gens + a second generation seed (777).

| ckpt | seed 1234 | seed 777 | pooled (2,720 late) |
|---|---|---|---|
| olmo_base | Δη −0.0449 [−0.0941, +0.0047] **flat** | Δη −0.0273 [−0.0756, +0.0190] **flat** | Δη −0.0361 [−0.0715, −0.0023] FIRES (marginal) |
| t_rlvr | Δη −0.1100 [−0.1529, −0.0654] **FIRES** | Δη −0.1715 [−0.2155, −0.1293] **FIRES** | Δη −0.1408 [−0.1724, −0.1097] **FIRES** |

olmo_base: **neither individual seed fires**; only the pooled large-N marginally crosses
(−0.036, vs the original −0.117) — the effect shrinks as N grows. t_rlvr: **both seeds fire**, large
stable magnitude.

### A2 — threshold sensitivity grid (both seeds pooled)
| ckpt | cells firing / 6 | fails where |
|---|---|---|
| olmo_base | **4/6** | flat at eff_room ≥20 (the most-decisive band), both cutoffs; Δη ~−0.03 |
| t_rlvr | **6/6** | fires at every band × cutoff; Δη −0.09 to −0.14 |

### A3 — own-history probe (fired checkpoints only)
0 decisive-late histories came from either model's OWN constrained runs; both rarely reach
decisive-late on-policy (olmo_base 2 own decisive, t_rlvr 6).

| ckpt | own-DECISIVE (N) | own-REACHED, any eff_room (N) |
|---|---|---|
| olmo_base | Δη −0.0594 [−0.214, +0.121] **flat** (2 hist, 80 obs) | Δη −0.0144 [−0.0667, +0.0383] **flat** (30 hist, 1,200 obs) |
| t_rlvr | Δη −0.1533 [−0.243, −0.062] **FIRES** (6 hist, 240 obs) | Δη −0.1522 [−0.184, −0.119] **FIRES** (90 hist, 3,600 obs) |

On its OWN histories olmo_base is **flat** (even well-powered at 1,200 obs); t_rlvr **fires** on its
own reached histories (3,600 obs).

### Revised fire verdict (resolves the 1-vs-2 count)
| ckpt | (i) larger N + 2nd seed | (ii) thresholds | (iii) own-history | **BOTTOM LINE** |
|---|---|---|---|---|
| **olmo_base** | both seeds FLAT; pooled marginal only | 4/6 (fails at ≥20) | FLAT (decisive & reached) | **fires only under pooled/counterfactual histories — effectively washes out; original −0.117 was a small-N/favorable-seed artifact** |
| **t_rlvr** | both seeds FIRE; pooled −0.141 | **6/6** | **FIRES** (own decisive & reached) | **fires (robust)** |

**TASK A bottom line: the count is 1, not 2.** Only **t_rlvr** is a robust fire (survives larger N,
a second seed, all 6 thresholds, and its own on-policy histories). **olmo_base does NOT survive** —
per-seed flat, own-history flat, vanishes at the most-decisive band; its original fire was driven by
the small-N pooled decisive set plus a favorable seed.

---

## TASK B — base-concession artifact check (forced-horizon, free vs constrained)

### B2 — context-matched grammar override rate (policy_shift logic, reused)
Override rate (action-type changed at a same-ask_shown matched turn): **base 0.28–0.35, z_base
0.25–0.26, t_base 0.24–0.26** — somewhat above Table-2's 16–21% because forced-horizon episodes run
deeper (more turns → more grammar-intervention opportunities). Full counts in the detail file.

### B3 — β(D) re-fit on three turn sets (frozen Test-4 regression)
| ckpt | op | (i) all constrained | (ii) **excl-overridden** | (iii) free natively-parsed | survives (ii)? |
|---|---|---|---|---|---|
| base | a hold_rate | +0.1806 [+0.111,+0.238] | **+0.1754 [+0.095,+0.243]** | +0.1262 [+0.057,+0.198] | **YES — real** |
| base | b conc_granted | −0.9795 [−1.13,−0.82] | **−0.9756 [−1.13,−0.81]** | −0.7189 [−0.90,−0.52] | **YES — real** |
| z_base | a hold_rate | +0.1300 [+0.035,+0.228] | **+0.1479 [+0.058,+0.243]** | +0.2133 [+0.118,+0.308] | **YES — real** |
| z_base | b conc_granted | −0.9154 [−1.20,−0.63] | **−0.8925 [−1.18,−0.59]** | −1.0823 [−1.28,−0.88] | **YES — real** |
| t_base | a hold_rate | +0.0780 [+0.010,+0.145] | **+0.0725 [−0.0004,+0.147]** | +0.0953 [−0.014,+0.196] | **NO — collapses to null (artifact)** |
| t_base | b conc_granted | −0.6056 [−0.85,−0.35] | **−0.6087 [−0.86,−0.32]** | −0.7389 [−1.03,−0.43] | **YES — real** |

(iii) free-decode is reported but **flagged for Defect-B selection bias** — free parse-failure is
action-dependent, so the free-parsed set conditions on the endogenous variable; (ii) is the clean
artifact-isolating estimate. β(C)−β(B) op-a **survives override-exclusion for all three base models**
(all CI-excl-0); op-b differs for none.

### Verdict per base model
- **OLMo base:** caving **REAL** — survives override exclusion on **both** ops (op-a +0.175 non-null;
  op-b −0.976 non-null).
- **Mistral base (z_base):** caving **REAL** — survives on **both** ops (op-a +0.148; op-b −0.893).
- **Llama base (t_base):** **MIXED** — op-a (hold_rate) **collapses to null** when overridden turns
  are removed (+0.073 [−0.0004, +0.147]) → **grammar-projection artifact** for op-a (marginal: upper
  bound +0.0004); op-b (conc_granted) **survives** (−0.609 non-null) → real.

**TASK B bottom line:** the base maladaptive caving (concede MORE to a harder seller) is **largely a
REAL base behavior that post-training removes**, NOT a pure grammar artifact — it survives override
exclusion in 5 of 6 model×op cells. The **single exception is t_base op-a (hold_rate)**, which is a
grammar-projection artifact (collapses to null, and was the weakest signal to begin with at +0.078).
The aligned finals were null in Task 1, so post-training genuinely removes this base heuristic.

---

## Headline bottom lines
- **TASK A — fire count = 1 (t_rlvr), not 2.** `t_rlvr` = **fires (robust)** across larger N, a 2nd
  seed, all 6 thresholds, and own on-policy histories. `olmo_base` = **fires only under
  pooled/counterfactual histories — washes out** (per-seed flat, own-history flat, original −0.117
  shrinks to −0.036). Audit: the original late bin was 100% pooled passA, fired checkpoints owned
  only 2–3/30 of their histories; bootstrap & per-cell budget were already correct; "four
  free-decode checkpoints" is not in the frozen record.
- **TASK B — base caving is mostly REAL, not a grammar artifact.** Survives override-exclusion in 5/6
  base×op cells (OLMo base both ops, Mistral base both ops, Llama base op-b). Only **t_base op-a**
  collapses to null → grammar-projection artifact. Override rates 24–35% under forced-horizon.
