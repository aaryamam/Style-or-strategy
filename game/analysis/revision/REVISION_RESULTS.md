# TMLR Revision Analysis — Results (REVISION_RESULTS.md)

Human-readable companion to `revision_numbers.json`. Every number is recomputed
from persisted logs by importing the frozen pipeline; **zero new model
inference**. Repo fingerprint (content-hash, not git — repo is not a checkout):
`ccf716193f90165c782a9c6209719d2c41821115086c99a5f0c9ef39ad6ecb5c`. Reproduction
gates: **19/19 PASS** (see `INVENTORY.md` §2, `repro_gates.json`). Run everything
with `python analysis/revision/run_all.py`.

## Verdict summary

| Task | Verdict / headline |
|---|---|
| T1 accept-timing | **`adaptation`** — fires in the *powered* (k=4) *aligned* checkpoints; base models null. See caveats. |
| T2 MDE→surplus | binding excluded bound **0.182 pts (1.79%)** — matches the paper's ≤1.8% cap. |
| T3 permutation | every displacement p≈0 **except OLMo dpo→instruct p=0.0046** (barely clears its floor). |
| T4 override intermediates | override **collapses to <0.011** after the base checkpoint. |
| T5 strict locus | OLMo & Zephyr loci **preserved**; Tülu chain **NO-GO** (no intermediate passA). |
| T6 sign-consistency | **consistent** across all 3 turn sets, all 6 model×op cells. |
| T7 centered IF | **`unchanged`**; bootstrap does **not** recompute IF/Σ (`pnBootstrapRecompute=NO`). |
| T8 OLMo gates | retrospective; β flips +0.181→−0.147, walkaway shared-null. |
| T9 lenient | **`vacuous`** (0 recovered turns). |
| T10 RLVR forfeit | realized cost large for base, ≈0 for t_rlvr; latent kernel **persists**. |
| T12 accept-timing value | **`indeterminate`** — D's accept-timing gradient is **null** (0.00 pt); the T1 signal is surplus-inert/stylistic. |

---

## T1 — Accept-timing adaptation ("Test 5")   *(new analysis)*

**Method.** Unit of analysis = every buyer decision point with a standing seller
ask (every recorded turn; accept available). `accept=1` iff the buyer's action is
`accept`. Per-checkpoint logistic regression (IRLS, tiny L2 ridge 1e-4)
`accept ~ hardness + ask_frac + turn_index + room_remaining`, with hardness under
both Test-4 operationalizations (op-a cumulative hold rate, op-b cumulative
concession granted — the exact Test-4 constructors). Episode-cluster bootstrap
(resample episodes, B=2000, seed 1234). Run on the seller-experiment episodes
(natural T=8) and the forced-horizon k=4 episodes (the powered variant for terse
checkpoints).

**Analyst decisions (logged).** (1) `accept` excludes offer-sealed deals (an offer
clearing the bar is an *offer*, not an accept-timing choice). (2) `ask_frac =
ask_shown/v_b` (buyer-observable); `room_remaining = (v_b−ask_shown)/(v_b−floor)`
(analyst-side achievable room the buyer would capture by accepting now). (3)
round-1 points included (hardness=0 baseline; `turn_index` controls stage). (4)
logistic fit by IRLS with ridge for quasi-separation.

**Primary contrast — seller-D hardness coefficient (accepts n in parens):**

| ckpt | op | β (natural) [CI] | β (k=4) [CI] | fires? |
|---|---|---|---|---|
| base | a | −0.134 [−4.58, 1.72] (24) | +0.034 [−0.52, 0.54] (206) | no |
| base | b | +3.889 [−8.82, 12.98] | −0.781 [−2.25, 0.61] | no |
| instruct | a | −0.403 [−1.30, 0.54] (131) | −0.386 [−0.90, 0.15] (354) | no |
| instruct | b | +6.238 [−0.53, 11.40] | **+5.698 [+2.28, +8.67]** | **k4** |
| z_base | a | +0.520 [−1.49, 1.90] (32) | −0.096 [−0.79, 0.51] (172) | no |
| z_base | b | +1.616 [−7.32, 9.06] | +0.617 [−1.11, 2.33] | no |
| z_dpo | a | **−4.746 [−24.8, −1.60]** (14) | **−2.877 [−4.16, −1.91]** (287) | **both** |
| z_dpo | b | +6.346 [−6.86, 33.7] | **+8.472 [+5.61, +12.4]** | **k4** |
| t_base | a | +0.340 [−0.41, 1.06] (81) | +0.194 [−0.30, 0.67] (279) | no |
| t_base | b | −0.569 [−5.01, 3.31] | −0.416 [−2.37, 1.52] | no |
| t_rlvr | a | +0.449 [−1.22, 1.84] (38) | **−0.786 [−1.56, −0.10]** (222) | **k4** |
| t_rlvr | b | −2.641 [−9.23, 3.92] | **+3.259 [+1.03, +5.70]** | **k4** |

**Branch verdict: `adaptation`** (not every per-checkpoint seller-D CI spans 0).
Firing cells: natural `z_dpo·op-a`; forced-horizon `instruct·op-b`, `z_dpo·op-a`,
`z_dpo·op-b`, `t_rlvr·op-a`, `t_rlvr·op-b`.

**Interpretation / caveats (important).**
- Firing is concentrated in the **forced-horizon (powered)** variant and in the
  **aligned** checkpoints (`instruct`, `z_dpo`, `t_rlvr`) — **no base model fires**
  under either op/horizon. This is a genuine departure from the Test-4 *concession*
  null (which found no seller-adaptation in the opening/concession channel): the
  *accept* decision in aligned models is history-sensitive.
- op-a and op-b fire with **opposite signs** for the same checkpoint (z_dpo, t_rlvr:
  op-a negative, op-b positive). These are economically **consistent** (op-a
  hold_rate and op-b conc_granted are inversely related): accept *less* when the
  seller has been holding, *more* when it has conceded — conditional on the standing
  ask level. This is opponent-sensitive accept timing.
- Natural-horizon cells are **underpowered** (14–131 accepts); their CIs are
  separation-inflated (z_dpo op-a natural CI spans [−24.8, −1.60]) and should not be
  over-read. The forced-horizon cells (207–354 accepts) are the reliable ones.
- op-b (conc_granted) is partly collinear with `ask_frac`/`room_remaining` (all
  functions of the current ask); its large magnitudes are inflated by that
  collinearity. op-a (hold_rate, a rate over prior rounds) is the cleaner axis, and
  it fires for z_dpo (both horizons) and t_rlvr (k=4).

**Secondary contrasts** are in `revision_numbers.json`
(`accept_timing.<horizon>.<ckpt>.<op>.{pooledFE,BvsC}`): pooled-across-sellers with
seller fixed effects, and the β(B)−β(C) cross-seller difference.

**Displacement Δ(final−base), seller-D hardness, bootstrap CI:**
- op-a: OLMo −0.269 [−2.26, 4.36] (spans 0); **Zephyr −5.266 [−24.2, −1.93]
  (excludes 0)**; Tülu +0.109 [−1.76, 1.62] (spans 0).
- op-b: all span 0.
So post-training *significantly* increased Zephyr's accept-timing responsiveness to
seller holds (op-a), consistent with the z_dpo firing above.

**Value context.** The opponent-modeling / accept-timing value share (best-response
oracle − flat, from `sellerD_adaptation_value.json`) is **3.24 pts (natural)** /
**4.32 pts (k=4)** — i.e. accept-timing/opponent-modeling is where surplus is left
on the table, so a firing accept-timing signal is economically meaningful, unlike
the ≤0.18-pt concession channel (T2).

---

## T2 — Test-4 MDE → excluded-surplus conversion   *(re-analysis)*

**Unit mapping (derived, not eyeballed).** Test-4 response is
`concession=(p_t−p_{t−1})/room`, `room=v_b−floor`, slope `β=d(concession)/dh`,
`h∈[0,1]`. The best-response policy uses
`p_t=p_{t−1}+max(0,(c_base+c_adapt·h))·(v_b−p_{t−1})`, so realized
`concession/room = conc_frac·ρ`, `ρ=(v_b−p_{t−1})/(v_b−floor)`, giving
`d(concession/room)/dh = c_adapt·ρ`. Matching: **`c_adapt = β/ρ̄`**, ρ̄ = mean
remaining-room fraction over the flat-optimal trajectory (**ρ̄=0.550 natural,
0.584 k=4**, computed empirically). Both ops share the mapping (h∈[0,1] in both).

**Estimand.** `β_max = max(|CI_lo|,|CI_hi|)` → slope bound `c_map=β_max/ρ̄`.
Excluded surplus = `max_{|c_adapt|≤c_map} S*(c_adapt) − S*(0)`, where `S*(v)` is the
best-(α₀,c_base) seller-D surplus at slope `v` (grid-searched with the frozen
best-response module; α*-gate surplus units). Flat (c_adapt=0) is in-range so the
excluded value is ≥0 and bounded by the module's global adaptation cap.

**Result.** Well-powered cells with tight CIs bound the channel to ≈0 (e.g.
`z_dpo·op-a·k4`, c_map=0.067 → **0.000 pts**). Loose-CI cells rise to the global cap
(natural op-b 0.128 pts; k=4 op-b 0.182 pts). **Binding (largest defensible) bound:
0.1823 pts = 1.79% of the flat buyer's surplus** (`forced_k4·base·op-b`,
β_max=1.129). This **matches the paper's own ≤~0.18-pt / ≤1.8% cap** — the unit
mapping is validated by landing on that anchor. Per-cell values in
`revision_numbers.json:mde_surplus.cells`. `pnMDEsurplus=0.182`, `pnMDEpct=1.79`.

---

## T3 — Paired permutation nulls + empirical floors   *(new computation)*

**Method.** On the shared constrained episode grid, swap the two checkpoints'
per-episode signatures within each paired episode with prob ½; recompute the **full**
D̂_fam pipeline per permutation — per-checkpoint means, the aggression_penalty IF
entries (recomputed per group), and pooled Σ (ridge 0.05). B=5000, seed 1234.
p = fraction of permuted D̂ ≥ observed. The numpy re-implementation reproduces
`study.dfam`'s point D to 1e-9 (sanity: OLMo base→final 0.792349 = 0.792349).

| comparison | N | D | p | null med | null p95 | unpaired √(2k/N) |
|---|---|---|---|---|---|---|
| OLMo base→instruct | 500 | 0.792 | **0.0000** | 0.125 | 0.204 | 0.141 |
| Zephyr z_base→z_dpo | 500 | 0.353 | **0.0000** | 0.137 | 0.220 | 0.141 |
| Tülu t_base→t_rlvr | 500 | 0.801 | **0.0000** | 0.137 | 0.220 | 0.141 |
| OLMo base→sft | 500 | 1.110 | 0.0000 | 0.137 | 0.225 | 0.141 |
| OLMo sft→dpo | 500 | 0.701 | 0.0000 | 0.121 | 0.196 | 0.141 |
| **OLMo dpo→instruct** | 500 | **0.096** | **0.0046** | 0.045 | 0.077 | 0.141 |
| Zephyr z_base→z_sft | 2000 | 0.418 | 0.0000 | 0.069 | 0.113 | 0.071 |
| Zephyr z_sft→z_dpo | 2000 | 0.586 | 0.0000 | 0.065 | 0.106 | 0.071 |
| Tülu t_base→t_sft | 2000 | 0.819 | 0.0000 | 0.072 | 0.117 | 0.071 |
| Tülu t_sft→t_dpo | 2000 | 1.328 | 0.0000 | 0.070 | 0.116 | 0.071 |
| **Tülu t_dpo→t_rlvr** | 2000 | **0.160** | **0.0000** | 0.031 | 0.050 | 0.071 |

**Empirical paired floor.** N=500: null median 0.045–0.137, p95 0.077–0.225.
N=2000: null median 0.031–0.072, p95 0.050–0.117. The paired null median tracks the
unpaired analytic approx (√(2k/N) = 0.141 @500, 0.071 @2000) closely — pairing on
the shared grid does **not** shrink the floor much here (`pnNullFloorFiveHundred`,
`pnNullFloorTwoK`).

**RLVR-step floor comparison (`pnRLVRFloorCompare*`).**
- **OLMo DPO→Instruct (0.096, N=500): p=0.0046, sits at the ~99.5th percentile of
  its own null (median 0.045, p95 0.077).** It is the weakest displacement — real,
  but only just above its floor.
- **Tülu DPO→RLVR (0.160, N=2000): p=0.0000, far above its null (median 0.031,
  p95 0.050).** RLVR moves the signature decisively for Tülu; the final OLMo step
  barely clears its floor.

---

## T4 — Override/rescue rows for intermediate checkpoints   *(pipeline re-run)*

Frozen policy-shift diagnostic (`policy_shift.analyze`) on the intermediate
checkpoints with an existing paired free(passA)/constrained run. Table-2 format:

| ckpt | override | rescue | ovr\|aggr (n) | ovr\|pass (n) | round-1 (n) | N pair/ctx/fp |
|---|---|---|---|---|---|---|
| olmo_sft | 0.007 | 0.008 | 0.018 (163) | 0.003 (388) | 0.008 (495) | 615/613/608 |
| olmo_dpo | 0.007 | 0.003 | 0.000 (116) | 0.008 (497) | 0.004 (499) | 726/726/724 |
| z_sft | 0.011 | 0.058 | 0.023 (88) | 0.007 (416) | 0.002 (477) | 711/705/664 |

Tülu intermediates (t_sft, t_dpo): "**–**" (no passA free run).

**Per-step exposure summary (override rate at each step's endpoints):**
- OLMo: base **0.208** → sft 0.0066 → dpo 0.0069 → instruct 0.0082.
- Zephyr: z_base **0.161** → z_sft 0.0105 → z_dpo 0.0014.
- Tülu: t_base **0.165** → (t_sft –) → (t_dpo –) → t_rlvr 0.0108.

**Read.** The grammar-override confound is a **base-model phenomenon**: override
collapses ~20–30× at the first post-training step and stays <1.1% thereafter. Every
consecutive attribution step *after* base is essentially override-free, so the
Table-7 loci are not exposed to the override confound.

---

## T5 — Strict-regime stage attribution   *(pipeline re-run under strict)*

Consecutive-step D_fam under strict scoring (freepass strict replay), per-step
pooled Σ, B=2000, seed 1234, N=500 (passA only exists at N=500). Coverage limited by
passA availability.

| family | step | D (strict) | 95% CI |
|---|---|---|---|
| OLMo | base→sft | 1.118 | [0.988, 1.284] |
| OLMo | sft→dpo | 0.710 | [0.614, 0.840] |
| OLMo | dpo→instruct | 0.093 | [0.056, 0.161] |
| Zephyr | z_base→z_sft | 0.433 | [0.325, 0.584] |
| Zephyr | z_sft→z_dpo | 0.569 | [0.472, 0.698] |
| Tülu | t_base→t_rlvr | 1.141 | [1.037, 1.260] | (base→final only) |

**Locus verdict (`pnStrictLocusVerdict`).**
- **OLMo**: strict locus = `base→sft` (supervised) = Table-7 constrained locus,
  **CI-separated**, ordering **preserved**.
- **Zephyr**: strict locus = `z_sft→z_dpo` (preference) = Table-7 locus, ordering
  **preserved**, but **CI-separation is lost at N=500 strict** (z_base→z_sft hi=0.584
  overlaps z_sft→z_dpo lo=0.472) — flag: the locus is preserved in point-ranking but
  the two Zephyr steps are not separated at this N under strict scoring.
- **Tülu**: **NO-GO** for the locus test — only base→final is computable under strict
  (t_sft/t_dpo have no free-decode pass). Reported as not_computable.

---

## T6 — Sign-consistency across turn sets   *(re-fit on forced-horizon logs)*

Forced-horizon (k=4) caving slope β(D) for 3 base models × 2 ops × 3 turn sets
(all-constrained / excl-override / free-parsed). All values reproduce the published
override-excluded β(D) (OLMo +0.175/−0.976, Mistral/Zephyr +0.148/−0.893,
Llama/Tülu +0.073/−0.609) — cross-check ✓. Full table in `tables/sign_consistency.tex`.

**Verdict: sign-consistent** across all three turn sets for **all 6 model×op cells**
(op-a always +, op-b always −). The free-parsed column (Defect-B selection bias,
flagged) preserves signs though magnitudes drift. `sign_consistency`=`consistent`.

---

## T7 — Centered influence function + bootstrap inspection   *(pipeline variant + code read)*

**(a)** Rebuilt aggression_penalty per-episode entries with the centered univariate
IF `a_i = β̂ + (o_i−ō)(p_i−p̄−β̂(o_i−ō))/s²_o` (vs `study._agg`'s multivariate
`[1,opening,v_b,c_s]` IF), rebuilt Σ, recomputed all base→final distances, all
Table-7 steps, and the per-family locus. Point-D deltas are small (all |ΔD|≤0.08;
e.g. OLMo base→final 0.792→0.856, Tülu t_dpo→t_rlvr 0.160→0.158). **Locus ordering
and CI-separation preserved for all three families** (OLMo→`base→sft`,
Zephyr→`z_sft→z_dpo`, Tülu→`t_sft→t_dpo`). **Verdict: `unchanged`.** (T7 uses the
loco_locus N convention — OLMo N=500, Zephyr/Tülu N=2000 — so its base→final row
differs from Table 3's N=500 Zephyr/Tülu values.)

**(b) `pnBootstrapRecompute = NO`.** `study.dfam` computes `pooled_stats` and
`Sinv=_inv(cov(...))` **once before** the bootstrap loop; the loop only resamples
indices and re-averages component means via `mvec`. The aggression_penalty IF
entries are fixed at load time by `study._wrap/_agg` and never recomputed per
resample. So IF entries and pooled Σ are held **fixed** from the full sample; only
the per-component means vary across resamples. (This is exactly the assumption T3's
permutation deliberately relaxes.)

---

## T8 — OLMo admissibility gates, retrospective   *(gate code on OLMo data)*

Same numeric gates as §6 Zephyr/Tülu, on OLMo constrained base/final. All labelled
`retrospective`.
- **β (aggression_penalty):** base **+0.181**, final **−0.147**; construct check:
  82/99 (0.83) of rejected-opening deals resolve *worse* than the accepted group —
  the penalty construct is valid (aggressive openings that get rejected end worse).
- **Variance (non-degeneracy):** every one of the 5 components is well above the
  near-constant threshold at both checkpoints (min var 0.042, walkaway) — **no
  degenerate axis**.
- **Walkaway non-nullity:** base→final standardized gap **+0.000**, raw rate
  0.044→0.044 — walkaway is a **shared null** for OLMo (as for Zephyr/Tülu).
`pnOLMoBeta/Var/Walk`.

---

## T9 — Lenient-recovery counts   *(count from scoring logs)*

Per checkpoint, turns where strict parsing forfeited but the lenient re-scorer
recovered a valid buyer action. **Total = 0** (OLMo 0, Zephyr 0, Tülu 0).
Strict vs lenient base→final D_fam (4 dp) are **identical** — OLMo 0.8629/0.8629,
Zephyr 0.6821/0.6821, Tülu 1.1415/1.1415 (gap 0.0000). **Verdict: `vacuous`.** This
confirms the freepass caveat: base format deficits are core-INVALID
(multiple-object / missing-price) failures that are no-ops under *both* strict and
lenient rules, so lenient recovery adds nothing. `pnLenientCounts`.

---

## T10 — RLVR forfeit cost: realized and latent   *(re-analysis)*

**δ (from code):** a strict parse-fail turn is a **no-op** (no buyer action
credited; ask unchanged); a forfeit-forced no-deal defaults to **zero captured
surplus** (`game_lib.extract_buyer_action` + `seller_exp`/`freepass`).

**(a) Realized forfeit-regime cost** — strict-vs-constrained captured surplus per
four-condition checkpoint (over feasible episodes):

| ckpt | strict | constrained | gap | Δ eff | parse record |
|---|---|---|---|---|---|
| base | −0.054 | 0.347 | **+0.401** | +0.013 | 757/924 |
| instruct | −0.374 | −0.384 | −0.010 | +0.001 | 729/729 |
| z_base | −0.581 | 1.345 | **+1.926** | +0.088 | 805/1123 |
| z_dpo | 2.554 | 2.571 | +0.017 | +0.001 | 771/847 |
| t_base | 0.635 | 1.286 | **+0.650** | +0.017 | 831/1142 |
| **t_rlvr** | 3.983 | 3.968 | **−0.015** | −0.002 | **1501/1501** |

The deployed cost of a forfeit harness is **large for base models** (base +0.40,
z_base +1.93, t_base +0.65 pts of captured surplus lost to forfeits) and **≈0 for
aligned/RLVR** (t_rlvr −0.015 with a perfect **1501/1501** parse record; instruct,
z_dpo ≈0). `pnRLVRSurplus=−0.015`, `pnRLVRPct≈−0.4%`; contrast
`pnRLVRBaselinePct≈116%` (base's forfeit cost exceeds its own constrained surplus —
forfeits more than double base's realized loss). *(Reported spec figure 1487/1487
for t_rlvr; the exact live count is 1501/1501 — zero forfeits either way.)*

**(b) Latent counterfactual exposure, t_rlvr** — at realized decisive states
(pool `late`; turn≥6, high achievable room), the absolute parse-failure rate of
**forced aggressive** actions (hold_justified, offer_conditional) pooled over
generation seeds 1234+777, × achievable room. *Labeled: exposure **conditional on
the policy taking the aggressive action**, not realized cost.*

| set | n gen | parse-fail rate | mean room | mean exposure (room·fail) |
|---|---|---|---|---|
| t_rlvr shared-decisive (seeds pooled) | 2880 | **0.445** | 19.47 | 8.69 |
| t_rlvr own-decisive | 120 | 0.500 | 17.83 | 8.92 |
| t_rlvr own-reached (n=90 hist) | 1800 | 0.499 | 1.19 | 0.59 |
| **olmo_base shared-decisive (contrast)** | 2880 | **0.693** | 19.47 | 13.45 |

**Read.** Even t_rlvr — which forfeits **nothing** on-policy (part a) — *would* fail
to parse **44.5%** of forced aggressive actions at decisive states, a mean exposure
of ~8.7 room-units per aggressive decision. The forfeit **kernel persists
structurally**; RLVR lowers it relative to base (69.3% / 13.4 room-units) but does
**not eliminate** it — it is masked because the RLVR policy does not *choose* those
aggressive actions on-policy. (Total-exposure sums are inflated by per-state
generation repeats; the interpretable quantities are the fail-rate and mean
per-decision exposure.) `rlvr_latent_exposure`.

---

## T12 — Accept-timing value: is the T1 sensitivity surplus-aligned with D?   *(addendum)*

**Question.** T1 found accept-timing *fires* in the aligned finals. Is that firing
**gradient-aligned** with seller D's best response (earned opponent-modeling) or
**anti-best-response** (the accept-margin analog of §8's caving check against
c*=+0.070)? Zero new inference — the existing best-response module pointed at the
accept decision.

**Anchor (reproduced, else stop).** `c* = +0.070` (op-b concession optimum) and the
accept-timing oracle−flat value **3.239 pt (natural) / 4.318 pt (k=4)** both
reproduce exactly from the module. ✓

**Accept-policy family + unit mapping (derived).** Extend the module's oracle accept
branch to a hardness-dependent threshold: accept iff `pos=(ask−floor)/room ≤ at(h) =
at0 + at_adapt·h`. From the T1 logistic accept boundary, folding both ask
coefficients, `at_adapt = b_hard/(b_room − b_ask·r̄)` (r̄ = mean achievable-room
fraction of value); `sign(at_adapt)=sign(b_hard)` where the ask-margin is
well-identified. **Surplus value** is computed by simulating the **actual fitted
logistic accept rule** vs D (accept iff the fitted index >0) against the same rule
with the hardness term zeroed ("hardness-blind, all else fixed") — *not* by
extrapolating the threshold family, which over-extrapolates the collinearity-inflated
op-b coefficient (an earlier version produced a spurious −11 pt via `at_adapt≈+5.2`;
the unit error was that linear extrapolation, fixed here by simulating the rule).

**D's accept-timing gradient is NULL.** Conditioning the accept decision on revealed
hardness yields **0.0000 pt** best-response gain over the blind-optimal threshold
(at0*=0.45), on **both** operationalizations, both horizons. The current standing ask
is a sufficient statistic for the accept decision — hardness history adds nothing.

**Surplus value of the fitted rule (k=4, ΔS = fitted − hardness-blind, α*-gate pts):**

| ckpt | op-a ΔS [CI range] | op-b ΔS [CI range] |
|---|---|---|
| instruct | +0.111 [+0.25, −0.02] | −0.310 [−0.21, −0.31] |
| z_dpo | +0.172 [+0.25, +0.10] | −0.342 [−0.24, −0.43] |
| t_rlvr | +0.195 [+0.58, +0.02] | −2.025 [−0.88, −2.45] |

Max gain over all powered finals **+0.195 pt** (op-a), min **−2.02 pt** (t_rlvr op-b);
sanity ceiling (≤3.2–4.3 pt) **OK**.

**Verdict: `accept_timing_direction = indeterminate`** — because D's accept-timing
gradient is null, there is *no best-response direction to align to*. The op-a hardness
term contributes a negligible **+0.11 to +0.19 pt**; the op-b term *subtracts* (down to
−2.0 pt for t_rlvr, the collinearity-inflated coefficient taken literally). Every cell
is `null_gradient` (surplus-neutral axis).

**Paper sentence — "installed but surplus-inert (stylistic), not gradient-aligned."**
The accept-timing sensitivity T1 detected in the aligned finals is *not earned
opponent-modeling*: it carries ≈0 best-response value against D (max +0.19 pt, on the
scale of the ≤0.18-pt concession channel from T2, and far below the 3.2–4.3-pt
accept-timing ceiling), and where the fitted op-b sensitivity is large it is mildly
*maladaptive*. This is behavioral-signature displacement, consistent with the
seller-experiment STYLE-DOMINATED verdict and §8's maladaptive-caving finding — the
accept-timing "firing" is a stylistic footprint, not strategy. (Analyst decisions:
surplus via faithful fitted-rule simulation not threshold extrapolation; direction sign
from `sign(b_hard)`; gradient declared null when its best-response gain over blind
< 0.02 pt.)

## Analyst decisions (consolidated)

1. **Provenance without git.** Repo is not a checkout; every value carries a
   content-hash `repo_fingerprint` in place of a git hash (`_prov.py`).
2. **T1 accept binary** excludes offer-sealed deals; `ask_frac=ask/v_b`,
   `room_remaining=(v_b−ask)/(v_b−floor)`; round-1 points kept; IRLS + ridge 1e-4.
3. **T2 unit mapping** `c_adapt=β/ρ̄`, ρ̄ empirical from the flat-optimal trajectory;
   excluded value = best-in-range surplus − best-flat surplus (≥0).
4. **T3** paired label-swap, full-pipeline recompute per permutation, B=5000; numpy
   pipeline validated to 1e-9 against `study.dfam`.
5. **T5/T4 NO-GO scopes** (Tülu intermediate free-decode) are reported as "–" /
   not_computable, never simulated.
6. **T7 base→final** rows use the loco_locus N convention (OLMo 500, Zephyr/Tülu
   2000), noted where it diverges from Table 3.
7. **T10 latent exposure** is explicitly conditional-on-aggressive-action, not
   realized cost; totals inflated by generation repeats (use fail-rate + mean).

## Surprises to flag for the paper edit

- **T1 fires** (accept-timing shows history-sensitivity) in the *aligned* models
  under the *powered* horizon — a real departure from the Test-4 concession-null,
  in the economically-meaningful channel (opponent-modeling worth 3.2–4.3 pts). Base
  models remain null. Interpret with the power/collinearity caveats above.
- **T3: OLMo dpo→instruct (0.096) barely clears its own permutation floor**
  (p=0.0046, ~99.5th pct), whereas **Tülu's RLVR step (0.160) is far above floor** —
  the two "small final steps" are not equivalent.
- **T5 Zephyr strict**: locus preserved in ranking but **CI-separation lost at N=500**.
- **T4**: the override confound is **entirely a base-model artifact** (collapses
  after the first post-training step) — a clean argument that attribution steps are
  not override-contaminated.
- **T10 latent kernel persists in t_rlvr** (44.5% counterfactual forfeit rate)
  despite zero realized forfeits — the forfeit mechanism is masked, not removed.
- **T12 resolves the T1 surprise: the accept-timing firing is stylistic, not
  strategic.** D's accept-timing gradient is **null** (0.00 pt best-response value —
  the current ask is a sufficient statistic), so the fitted sensitivity is
  surplus-inert (op-a ≤+0.19 pt) to mildly maladaptive (op-b), never a gain. T1
  "fires" statistically but the signal earns nothing against D — consistent with the
  STYLE-DOMINATED / maladaptive-caving story, not opponent-modeling.
