# Phase 0 — Inventory, Reproduction Gates, GO/NO-GO

Repo: `/path/to/game` (**not** a git checkout — `git rev-parse` fails).
Provenance is a content-hash fingerprint over the pipeline source + a
(path,size,mtime) manifest of every input data file: see `_prov.py` /
`repo_state.json`. Combined source fingerprint:
`ccf716193f90165c782a9c6209719d2c41821115086c99a5f0c9ef39ad6ecb5c`.
Every emitted value carries `repo_fingerprint` in place of a git hash.

**Zero new model inference was performed.** Every number below is recomputed from
persisted episode/probe logs by *importing* the frozen pipeline
(`study`, `signature`, `distance`, `metric_concession`, `policy_shift`,
`seller_exp`, `b_analyze`, `fh_analyze`, `loco_locus`, `freepass`,
`sellerD_adaptation_value`, `zephyr_gates`).

## 1. Data map

Per-turn episode records **do** persist everything T1/T4/T6/T10 need. A per-episode
record is `{"state": {...}}` (or, for passA, a flat state) with `turns[]` carrying
per-turn: `round`, `ask_shown` (standing seller ask), `action` (buyer action
type), `price`, `parse_success`, `effect` (`deal` / `walk` / `noop` /
`seller_counter:<new_ask>`), plus `strict_action`/`lenient_action` in passA. Accept
events = `action=="accept"` or an `offer` whose `effect=="deal"`. So standing asks,
buyer actions, accept events, and per-turn parse status are all recoverable.

| Artifact | Location | Format / N | Notes |
|---|---|---|---|
| Four-scoring-condition (base→final) | `runs/n500/{passA,constrained}_{base,sft,dpo,instruct,z_base,z_sft,z_dpo,t_base,t_rlvr}.jsonl` | N=500 | passA = free decode (strict+lenient per turn); constrained = grammar-forced. |
| Stage-attribution constrained | `runs/n500/constrained_*` (OLMo, N=500); `runs/n2000/constrained_{z_*,t_*}` (Zephyr/Tülu, N=2000) | 500 / 2000 | n2000 is **constrained-only** (no passA). |
| Seller experiment | `runs/seller_exp/{A,B,C,D}_{base,instruct,z_base,z_dpo,t_base,t_rlvr}.jsonl` | 6 ckpt × 4 sellers × 500 = 12,000 | per-turn asks/actions/effects persist. |
| Forced-min-horizon (k=4) | `results/forced_horizon/{A,B,C,D}_*.jsonl` | same 6×4×500 | `min_accept_round=4` variant. |
| Forced-horizon free (context-match) | `results/base_artifact/free_{B,C,D}_{base,z_base,t_base}.jsonl` | base models only | powers the B2 override / B3 free-parsed turn set. |
| Teacher-forced firing probe (synthetic) | `out/firing_test.json` | 6 ckpt | η per early/mid/late synthetic bin. |
| Firing probe realized-history | `results/realized_probe/realized_*.jsonl` (all 6); `results/realized_robust/probe_{olmo_base,t_rlvr}_{seed1234,seed777,own_seed1234}.jsonl` | seeds 1234 & 777 | `pool∈{early,late,late_loose}`, `action_type`, `eff_room`, `parse_success`. own_t_rlvr: 90 reached / 6 decisive. |
| Policy-shift paired runs | `runs/n500/{passA,constrained}_*` | see below | Table-2 diagnostic. |
| Best-response module | `out/sellerD_adaptation_value.json` (+ `sellerD_adaptation_value.py`) | — | α* surpluses 11.71/22.18/2.47 (A/B/C); oracle−flat 3.24 (T8) / 4.32 (k4). |
| Test-4 forced-horizon fits | recomputed by `b_analyze.py` / `fh_analyze.py` from the logs above | — | three turn sets fit on demand (T6). |
| Admissibility gates | `zephyr_gates.py`, `tulu_gates.py`; outputs `out/cov_{zephyr,olmo,tulu}.json` | — | reused for OLMo retrospective (T8). |

### Paired free/constrained coverage (resolves the Table-2 / Table-5 question)
Free decode (`passA`) exists in `runs/n500` for **OLMo {base, sft, dpo, instruct}**,
**Zephyr {z_base, z_sft, z_dpo}**, **Tülu {t_base, t_rlvr}**. Therefore:
- Table 2 base/final 6 checkpoints: **all present.**
- Table 5 free-decode for z_sft: **present** (and additionally olmo_sft, olmo_dpo).
- Tülu intermediates (t_sft, t_dpo): **no passA** → free-decode NO-GO (T5 Tülu chain, T4 Tülu intermediates).
- OLMo intermediates (olmo_sft, olmo_dpo): **passA present** → GO (resolved).

n2000 has **no** passA at any checkpoint (constrained-only).

### Harness parse-failure default δ (quoted from code)
`game_lib.extract_buyer_action`: a strict parse-**fail** turn yields
`strict_action=None`. In `seller_exp.run_vs_seller` / `freepass` such a turn is a
**no-op** (`effect="noop"`; the standing `ask` is unchanged, no buyer action is
credited); if it is the final round the episode ends `no_deal_timeout`. So **δ = a
forfeited (no-action) turn; the episode-level default on a forfeit-forced no-deal is
zero captured surplus.** (No auto-accept, no penalty price.)

## 2. Reproduction gates — ALL PASS (19/19)

`repro_gates.py` recomputes each published value by importing the frozen pipeline.
Full table in `repro_gates.json`. Point estimates match to displayed precision
(deterministic given stored logs + seed 1234; bootstrap/permutation trailing digits
may differ — seeds fixed and noted).

| Gate | Got | Target | Pass |
|---|---|---|---|
| T3 OLMo base→final | 0.7923 | 0.792 | ✓ |
| T3 Zephyr base→final | 0.3527 | 0.353 | ✓ |
| T3 Tülu base→final | 0.8013 | 0.801 | ✓ |
| T3 Tülu seller-A | 0.8007 | 0.8007 | ✓ |
| T7 OLMo base→sft | 1.1102 | 1.110 | ✓ |
| T7 OLMo sft→dpo | 0.7012 | 0.701 | ✓ |
| T7 OLMo dpo→instruct | 0.0965 | 0.096 | ✓ |
| T7 Zephyr z_base→z_sft | 0.4184 | 0.418 | ✓ |
| T7 Zephyr z_sft→z_dpo | 0.5860 | 0.586 | ✓ |
| T7 Tülu t_base→t_sft | 0.8192 | 0.819 | ✓ |
| T7 Tülu t_sft→t_dpo | 1.3277 | 1.328 | ✓ |
| T7 Tülu t_dpo→t_rlvr | 0.1604 | 0.160 | ✓ |
| T2 olmo_base override | 0.2083 | 0.208 | ✓ |
| T2 olmo_base rescue | 0.1778 | 0.178 | ✓ |
| Test-4 z_base op-a β(C)−β(B) | 0.8366 | 0.837 | ✓ |
| Forced-horizon β(D) OLMo base op-a (excl-ovr) | 0.1754 | 0.1754 | ✓ |
| α* surplus A | 11.714 | 11.71 | ✓ |
| α* surplus B | 22.180 | 22.18 | ✓ |
| α* surplus C | 2.473 | 2.47 | ✓ |

Additionally re-verified live this pass: the T3 numpy permutation pipeline
reproduces `study.dfam` point D to 1e-9 (OLMo base→final 0.792349 == 0.792349); T6
reproduces the published forced-horizon override-excluded β(D) for all three base
models both ops (OLMo +0.175/−0.976, Mistral/Zephyr +0.148/−0.893, Llama/Tülu
+0.073/−0.609). **Cleared to emit Phase-1 numbers.**

## 3. GO / NO-GO per task

| Task | Verdict | Blocking artifact / note |
|---|---|---|
| T1 Accept-timing (Test 5) | **GO** | seller_exp + forced_horizon per-turn logs present. |
| T2 MDE→excluded-surplus | **GO** | Test-4 CIs + `sellerD_adaptation_value` best-response module. |
| T3 Permutation nulls | **GO** | constrained n500/n2000 grids (paired by seeded idx). |
| T4 Override rows (intermediates) | **GO** (olmo_sft, olmo_dpo, z_sft) / **partial** | Tülu intermediates lack passA → "–". |
| T5 Strict stage attribution | **GO** OLMo full chain, Zephyr full chain; **base→final only** for Tülu | Tülu t_sft/t_dpo passA absent → intermediate strict steps NO-GO. |
| T6 Sign-consistency 3 turn sets | **GO** | forced_horizon + base_artifact free runs (base models). |
| T7 Centered IF + bootstrap inspect | **GO** | constrained grids; code reading. |
| T8 OLMo admissibility gates | **GO** | OLMo constrained base/final. |
| T9 Lenient-recovery counts | **GO** | passA strict+lenient per-turn interpretations. |
| T10 RLVR forfeit realized+latent | **GO** | four-condition runs + realized_robust probes (both seeds). |
| T11 Release bundle | optional | not run unless requested (anonymization scan only). |

No reproduction gate failed and no required data path is missing, so all GO items
proceed automatically. The only NO-GO scopes are Tülu-intermediate free-decode
analyses (T5 Tülu chain locus, T4 Tülu intermediate override), blocked by the
absence of `passA_t_sft`/`passA_t_dpo` (constrained-only by original design) — never
substituted by simulation.
