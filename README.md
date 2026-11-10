# Style or Strategy? Discriminating Post-Training Displacement from Its Confounds in LLM Price Negotiation

Official code, interaction logs, analysis suite, and reproduction artifacts for:

> **Style or Strategy?: Discriminating Post-Training Displacement from Its Confounds in LLM Price Negotiation**  
> Aaryaman Kalani, Dhruv Kumar, Yash Sinha  
> *Department of Computer Science, BITS Pilani*  
> Published in **Transactions on Machine Learning Research (TMLR)**

---

## Table of Contents

- [Overview](#overview)
- [The Three Confounds & Discriminators](#the-three-confounds--discriminators)
- [Summary of Key Findings](#summary-of-key-findings)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Reproducing Paper Results (CPU, Zero GPU Inference)](#reproducing-paper-results-cpu-zero-gpu-inference)
  - [1. Phase-0 Reproduction Gates](#1-phase-0-reproduction-gates)
  - [2. Master Analysis Runner](#2-master-analysis-runner)
  - [3. Running Individual Analytical Tasks](#3-running-individual-analytical-tasks)
- [Running Generation & Model Inference (GPU / vLLM)](#running-generation--model-inference-gpu--vllm)
  - [Evaluated Model Checkpoints](#evaluated-model-checkpoints)
  - [Running the Bilateral Negotiation Game](#running-the-bilateral-negotiation-game)
  - [Running the Format Probes](#running-the-format-probes)
- [Paper-to-Code Mapping](#paper-to-code-mapping)
- [Citation](#citation)
- [License & Contact](#license--contact)

---

## Overview

When an aligned language model (post-trained with SFT, DPO, or RLVR) behaves differently from its base model in interactive bargaining, it is tempting to credit post-training with instilling strategic capabilities. This paper argues that such an inference must be earned: measured base $\to$ aligned differences are confounded by:
1. **Output-format failures** that masquerade as strategic moves (e.g., a base model dropping a field or generating conversational commentary forfeits its turn);
2. **Differential failure kernels** that could concentrate selectively on decisive late turns beyond the reach of naive rescoring;
3. **Fixed stylistic anchoring** that can mimic strategic behavior against a single fixed counterpart.

We build one discriminating test per confound—each designed with a live deflationary outcome—and evaluate the battery across three open base $\to$ SFT $\to$ DPO ($\to$ RLVR) lineages: **OLMo-2-7B**, **Zephyr-7B (Mistral)**, and **Tülu-3-8B (Llama-3.1)**.

```
       Bilateral Game Setup                        Five-Component Signature                     Displacement Metric
  ┌───────────────────────────────┐           ┌───────────────────────────────────┐        ┌─────────────────────────────┐
  │ Buyer (LLM) vs. Seller        │           │ 1. opening_aggressiveness         │        │ Mahalanobis Distance:       │
  │ • Seller A (Hardened)         │  ──────>  │ 2. surplus_efficiency             │ ────>  │ D_fam = ||z_a - z_b||_{Σ^-1}│
  │ • Seller B (Concession-reward)│           │ 3. haggling indicator             │        │                             │
  │ • Seller C (Aggression-punish)│           │ 4. walkaway indicator             │        │ Measures intra-lineage      │
  │ • Seller D (Adaptive)         │           │ 5. aggression_penalty (via EIF)   │        │ displacement, not divergence│
  └───────────────────────────────┘           └───────────────────────────────────┘        └─────────────────────────────┘
```

---

## The Three Confounds & Discriminators

| Discriminator | Confound Separated | Method | Empirical Finding |
|---|---|---|---|
| **1. Format Wedge** | True behavior vs. output parse-artifact | Compares constrained decoding (vLLM + XGrammar JSON schema) against free decoding (strict and lenient parsing). | Format wedge is unequal across families (OLMo $\approx 8\%$, Tülu $\approx 30\%$, Zephyr $\approx 48\%$). Constrained decoding rewrites 16–21% of base actions (heavily aggressive moves) while touching $\le 1.1\%$ of aligned moves. |
| **2. Firing Test** | Irreducible differential failure kernel vs. uniform degradation | Teacher-forced probe testing parse success of aggressive vs. passive actions across turn depth (synthetic bins and realized decisive histories). | Synthetic probe shows zero selective kernel (flat $\Delta\eta$). Realized decisive histories show a genuine kernel **only** in Tülu-RLVR under free decoding; on-policy routing masks it. |
| **3. Strategy vs. Style** | Contingent behavior & effective strategy vs. fixed style | Counterpart battery: divergent-optimum sellers (A, B, C; $\alpha^* \in [0.05, 0.75]$) and adaptive seller D (modulates concessions on buyer aggression). | **Flat openings**: LLMs anchor at fixed opening prices regardless of seller optimum. **Null concession adaptation**: $\beta \approx 0$ against adaptive seller D under default evaluation prompt. |
| **Test 5: Detect-then-Audit** | Real installed contingency vs. surplus value | Logistic model of accept timing against revealed hardness, paired with a best-response value audit against Seller D. | Hardness-sensitive accept-timing is detected at forced horizon ($k=4$) in 5 of 6 aligned cells and 0 of 6 base cells, but audits **surplus-inert** ($\le +0.19$ points earned against a 3.2–4.3 point ceiling). |

---

## Summary of Key Findings

1. **Format-Controlled Displacement is Real**: Constrained decoding eliminates parse forfeits, leaving a statistically significant displacement within each family ($p < 2 \times 10^{-4}$ against paired-permutation nulls), but grammar projection asymmetrically rewrites base endpoints.
2. **Openings and Concessions are Non-Strategic by Default**: Models exhibit fixed per-checkpoint anchors across counterparty types. Concession behavior does not adapt to revealed seller hardness under the frozen evaluation prompt (elicitation probes show this is a default, not an incapacity).
3. **Accept-Timing Contingency is Surplus-Inert**: While post-training installs an accept-timing sensitivity to opponent hardness in aligned checkpoints at $k=4$, the standing ask is an exact sufficient statistic for Seller D's accept rule ($0.000$ gradient), making the contingency economically unrewarded.
4. **Stage Attribution & Interface Control**: When controlling for serialization (few-shot completion vs. chat templates), Tülu retains a robust preference-stage locus (SFT $\to$ DPO), Zephyr is indeterminate, and OLMo flips with serialization.
5. **Terminal RLVR Moves Behavior Least**: Terminal verifiable-reward RL (math/reasoning rewards) moves bilateral negotiation behavior less than every preceding post-training stage across both bases.

---

## Repository Structure

```text
.
├── README.md                           # This repository guide
├── supplementary_identification.pdf    # Identification theory monograph (proofs, theorems, bounds)
│
├── game/                               # Core negotiation game & analysis suite
│   ├── game_lib.py                     # Environment mechanics, JSON schemas, strict/lenient parsers
│   ├── runner.py                       # High-level episode execution harness
│   ├── engine_game.py                  # vLLM engine wrapper with guided JSON decoding
│   ├── signature.py                    # Five-component behavioral signature & EIF computation
│   ├── study.py                        # Mahalanobis D_fam metric, bootstrap CIs, permutation tests
│   ├── seller_exp.py                   # Sellers A, B, C, and adaptive Seller D definitions & runs
│   ├── sellerD_adaptation_value.py     # Best-response value module & oracle benchmarks
│   ├── policy_shift.py                 # Grammar override & rescue diagnostics (Table 6)
│   ├── metric_concession.py            # Test-4 concession-on-hardness regression
│   ├── b_analyze.py                    # Tests 1 & 4 analysis (oracle opening gap & concession)
│   ├── fh_gen.py / fh_analyze.py       # Forced-minimum-horizon (k=4) experiment & regression
│   ├── loco_locus.py                   # Leave-one-component-out locus sensitivity check
│   ├── zephyr_gates.py / tulu_gates.py # Admissibility gates for signature components
│   │
│   ├── runs/                           # 12,000+ frozen per-episode interaction logs
│   │   ├── n500/                       # Base->final runs (free passA & constrained, N=500)
│   │   ├── n2000/                      # Stage-attribution constrained runs (N=2000)
│   │   └── seller_exp/                 # 6 checkpoints × 4 sellers × N=500 episodes
│   │
│   ├── results/                        # Robustness and verification outputs
│   │   ├── forced_horizon/             # k=4 experiment logs (6 checkpoints × 4 sellers)
│   │   ├── base_artifact/              # Context-matched free runs for base models
│   │   ├── realized_probe/             # Realized-history firing probe generations
│   │   ├── realized_robust/            # Seeded robustness checks (seeds 1234 & 777)
│   │   ├── integrity_check/            # Forensic verification records
│   │   ├── CONSOLIDATED_REPORT.md      # Detailed verification report (Tasks 0-3)
│   │   └── FOLLOWUP_REPORT.md          # Follow-up report (Tasks A & B)
│   │
│   ├── out/                            # Precalculated summaries, covariances, and value caches
│   │   ├── seller_exp_results.json     # Test 1 & 4 compiled statistics
│   │   ├── sellerD_adaptation_value.json# Oracle benchmarks and best-response surfaces
│   │   ├── firing_test.json            # Synthetic teacher-forced probe results
│   │   └── cov_{olmo,zephyr,tulu}.json # Within-family covariance matrices
│   │
│   └── analysis/revision/              # Master reproducibility suite & LaTeX generators
│       ├── run_all.py                  # Master CLI runner executing Phase-0 and Phase-1
│       ├── repro_gates.py              # 19 Phase-0 reproduction gates (pre-flight checks)
│       ├── repro_gates.json            # Gate targets, tolerances, and verification status
│       ├── repo_state.json             # Content-hash provenance fingerprint manifest
│       ├── INVENTORY.md                # Comprehensive data artifact dictionary
│       ├── REVISION_RESULTS.md         # Full human-readable breakdown of numerical results
│       ├── revision_numbers.json       # Master JSON containing all computed statistics
│       ├── make_tables.py              # Drop-in LaTeX tabular generator
│       ├── tables/                     # Generated LaTeX tables matching the paper
│       └── t1_accept_timing.py .. t12_*# Individual task analysis scripts (T1-T12)
│
└── format_probe/                       # Synthetic and realized teacher-forced format probes
    ├── probe_lib.py                    # Probe prompts, counterfactual action injection
    ├── probe_zt.py                     # Synthetic probe execution across history bins
    ├── probe_zt_analyze.py             # Analysis of synthetic probe runs (Axis 1 & 2)
    ├── task2_realized_gen.py           # Realized-history probe runner on decisive turns
    └── out/                            # Raw probe generations (13,500+ turns)
```

---

## Installation & Setup

### Requirements
- **Python**: 3.10 or higher
- **Core Analysis (CPU-only)**: `numpy` (all reproduction tasks run out-of-the-box on CPU)
- **Model Inference (GPU-optional)**: `torch`, `vllm >= 0.6.0`, `xgrammar`, `transformers`

### Environment Setup
```bash
# Clone repository
git clone <repo-url>
cd <repo-name>

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install analysis dependencies
pip install numpy
```

---

## Reproducing Paper Results (CPU, Zero GPU Inference)

All per-turn and per-episode interaction logs (over 12,000 negotiation episodes and 13,500 format probe turns) are included in this repository. **No GPU or LLM inference is required to recompute all paper statistics, bootstrap confidence intervals, permutation nulls, and LaTeX tables.**

### 1. Phase-0 Reproduction Gates
Before running downstream analyses, execute the reproduction gates. These gates independently verify that the environment, data integrity, and core metrics match the frozen paper values exactly:

```bash
cd game
python3 analysis/revision/repro_gates.py
```

Expected output:
```text
==========================================================================
REPRODUCTION GATES: 19/19 PASS
ALL GATES PASS — cleared to emit Phase-1 numbers.
wrote analysis/revision/repro_gates.json
==========================================================================
```

### 2. Master Analysis Runner
To execute all analysis modules, compute bootstrap distributions, update `revision_numbers.json`, and regenerate all paper LaTeX tables:

```bash
cd game
python3 analysis/revision/run_all.py
```

This master runner executes:
1. **Provenance Verification**: Validates the SHA-256 fingerprint (`repo_state.json`).
2. **Phase-0 Reproduction Gates**: Aborts if any of the 19 gates fail.
3. **Phase-1 Task Modules** (`T1` through `T12`): Recomputes regressions, permutation distributions, and value audits.
4. **Table Generation**: Outputs publication-ready LaTeX fragments in `game/analysis/revision/tables/`.

### 3. Running Individual Analytical Tasks
You can also run specific tasks using the `--only` flag:

```bash
cd game

# Run Accept-Timing (Test 5) logistic models:
python3 analysis/revision/run_all.py --skip-gates --only t1

# Run Value Audit of Accept-Timing against Seller D:
python3 analysis/revision/run_all.py --skip-gates --only t12

# Run Paired Permutation Nulls for D_fam (Table 11):
python3 analysis/revision/run_all.py --skip-gates --only t3

# Run Consecutive-Step Strict Attribution (Table 12):
python3 analysis/revision/run_all.py --skip-gates --only t5

# Run Forced-Horizon Sign Consistency Check (Table 14):
python3 analysis/revision/run_all.py --skip-gates --only t6
```

#### Task Reference Table

| Task Flag | Script | Paper Section / Table | Description |
|---|---|---|---|
| `t1` | `t1_accept_timing.py` | Section 5, Table 8 | Logistic models of accept-timing on revealed hardness (natural vs. $k=4$). |
| `t2` | `t2_mde_surplus.py` | Section 5, Appendix G | Minimum detectable effect (MDE) and excluded surplus bounds ($0.182$ pts / $1.79\%$). |
| `t3` | `t3_permutation.py` | Appendix D, Table 11 | Exact paired-permutation null calibration for $D_{\text{fam}}$ ($B=5000$). |
| `t4` | `t4_override_intermediate.py`| Section 6, Table 6 | Grammar override exposure across intermediate stages (SFT, DPO). |
| `t5` | `t5_strict_attribution.py` | Appendix E, Table 12 | Consecutive-step attribution under strict (forfeit) decoding. |
| `t6` | `t6_sign_consistency.py` | Section 8, Table 14 | Base caving slopes across 3 turn sets (constrained, override-excluded, free). |
| `t7` | `t7_centered_if.py` | Appendix H | Verification of centered empirical influence function (EIF) and bootstrap. |
| `t8` | `t8_olmo_gates.py` | Section 6, Appendix B.3| Retrospective admissibility gates for the in-sample OLMo family. |
| `t9` | `t9_lenient.py` | Section 3, Appendix D | Lenient re-scorer recovery audit (confirms 0 recoverable turns). |
| `t10` | `t10_rlvr_forfeit.py` | Section 7, Appendix B.4| RLVR masked-kernel audit: on-policy realized vs. counterfactual late turns. |
| `t12` | `t12_accept_timing_value.py`| Section 5, Table 9, Fig 3 | Best-response value audit of fitted accept rules against Seller D. |

---

## Running Generation & Model Inference (GPU / vLLM)

If you wish to run new model evaluations or collect new trajectories from open weights, use the inference scripts. Inference requires a CUDA-enabled GPU (tested on NVIDIA RTX A6000 48GB, `bfloat16`).

### Evaluated Model Checkpoints

All model lineages are publicly available on Hugging Face:

| Family | Stage | Checkpoint Hugging Face Identifier | Default Serialization |
|---|---|---|---|
| **OLMo-2-7B** | Base | `allenai/OLMo-2-1124-7B` | 3-shot completion prompt |
| | SFT | `allenai/OLMo-2-1124-7B-SFT` | Chat template (system + user) |
| | DPO | `allenai/OLMo-2-1124-7B-DPO` | Chat template |
| | Instruct (RLVR) | `allenai/OLMo-2-1124-7B-Instruct` | Chat template |
| **Zephyr-7B** | Base | `mistralai/Mistral-7B-v0.1` | 3-shot completion prompt |
| | SFT | `alignment-handbook/zephyr-7b-sft-full` | Chat template |
| | DPO | `HuggingFaceH4/zephyr-7b-beta` | Chat template |
| **Tülu-3-8B** | Base | `NousResearch/Meta-Llama-3.1-8B` (un-gated mirror) | 3-shot completion prompt |
| | SFT | `allenai/Llama-3.1-Tulu-3-8B-SFT` | Chat template |
| | DPO | `allenai/Llama-3.1-Tulu-3-8B-DPO` | Chat template |
| | RLVR | `allenai/Llama-3.1-Tulu-3-8B` | Chat template |

### Running the Bilateral Negotiation Game

Negotiate $N=500$ episodes against a specific seller under constrained JSON decoding:

```bash
cd game
python3 seller_exp.py \
    --role allenai/OLMo-2-1124-7B-Instruct \
    --seller D \
    --n 500 \
    --backend vllm \
    --gpu_mem 0.85
```

Available sellers:
- `A`: Hardened frozen seller (opens high, concedes 15%/round, holds 35%, split-the-difference accept bar).
- `B`: Concession-rewarding seller (optimal fixed opening $\alpha^* = 0.05$).
- `C`: Aggression-punishing seller (optimal fixed opening $\alpha^* = 0.75$).
- `D`: Adaptive seller (adjusts concession rate based on buyer's historical aggression).

To run the forced-minimum-horizon experiment ($k=4$):
```bash
python3 fh_gen.py --role allenai/OLMo-2-1124-7B-Instruct --seller D --n 500
```

### Running the Format Probes

Execute the synthetic teacher-forced format probe across early, mid, and late history bins:
```bash
cd format_probe
python3 probe_zt.py --role allenai/OLMo-2-1124-7B --backend vllm
python3 probe_zt_analyze.py
```

Execute the realized-history format probe on decisive late game states ($round \ge 6$):
```bash
python3 task2_realized_gen.py --role allenai/Llama-3.1-Tulu-3-8B --backend vllm
```

---

## Paper-to-Code Mapping

| Paper Item | Description | Code / Data Artifact |
|---|---|---|
| **Table 1** | Overview of the three discriminators and results | Summarized in `README.md` and [REVISION_RESULTS.md](game/analysis/revision/REVISION_RESULTS.md) |
| **Table 2** | Base $\to$ final $D_{\text{fam}}$ across 4 scoring regimes & format wedges | `game/study.py`, `game/analysis/revision/tables/table2_extension_rows.tex` |
| **Table 3** | Synthetic firing test ($\eta$ and $\Delta\eta$ by history bin) | `format_probe/probe_zt_analyze.py`, `game/out/firing_test.json` |
| **Table 4** | Stage attribution under default protocol ($D_{\text{fam}}$ consecutive steps) | `game/study.py`, `game/analysis/revision/t3_permutation.py` |
| **Table 5** | Interface control: stage locus under matched serializations | `game/analysis/revision/t5_strict_attribution.py`, `game/loco_locus.py` |
| **Table 6** | Policy-shift diagnostic (grammar override & rescue rates) | `game/policy_shift.py`, `game/analysis/revision/t4_override_intermediate.py` |
| **Table 7 & Figure 2** | Test 1: Oracle opening gap regressions against $\alpha^*$ | `game/b_analyze.py`, `game/out/seller_exp_results.json` |
| **Table 8 & Figure 3 (Left)** | Test 5 Detection: Accept-decision hardness coefficients | `game/analysis/revision/t1_accept_timing.py`, `game/analysis/revision/tables/accept_timing.tex` |
| **Table 9 & Figure 3 (Right)**| Test 5 Audit: Best-response gradient and fitted $\Delta S$ | `game/analysis/revision/t12_accept_timing_value.py`, `game/sellerD_adaptation_value.py` |
| **Table 10 & Figure 4** | Constrained per-component standardized displacement | `game/signature.py`, `game/out/components_pooled.json` |
| **Table 11** | Null calibration of $D_{\text{fam}}$ via paired permutation tests | `game/analysis/revision/t3_permutation.py` |
| **Table 12** | Strict-regime (forfeit) consecutive-step locus | `game/analysis/revision/t5_strict_attribution.py`, `game/analysis/revision/tables/strict_locus.tex` |
| **Table 13** | Firing test Axis 2: Logistic slope of parse failure on turn index | `format_probe/probe_zt_analyze.py` |
| **Table 14** | Forced-horizon base caving slopes across 3 turn sets | `game/fh_analyze.py`, `game/analysis/revision/t6_sign_consistency.py` |
| **Table 15** | Weight ($W$), serialization ($I$), and interaction ($X$) decomposition | `game/analysis/revision/REVISION_RESULTS.md` |
| **Table 16** | Non-adaptive scripted-buyer anchors for $D_{\text{fam}}$ | `game/study.py` (Scripted-buyer validation) |
| **Appendix C & PDF** | Identification theory under misclassification and censoring | [supplementary_identification.pdf](supplementary_identification.pdf) |

---

## Citation

If you find this work, codebase, or interaction logs helpful in your research, please cite our paper:

```bibtex
@article{kalani2026style,
  title   = {Style or Strategy?: Discriminating Post-Training Displacement from Its Confounds in {LLM} Price Negotiation},
  author  = {Aaryaman Kalani and Dhruv Kumar and Yash Sinha},
  journal = {Transactions on Machine Learning Research},
  year    = {2026}
}
```

---

## License & Contact

- **Code License**: MIT License. See [LICENSE](LICENSE) for details.
- **Data & Logs**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Questions & Issues**: For questions regarding reproducing results or the methodology, please open an issue on GitHub or contact the authors:
  - Aaryaman Kalani: `f20220488@pilani.bits-pilani.ac.in`
  - Dhruv Kumar: `dhruv.kumar@pilani.bits-pilani.ac.in`
  - Yash Sinha: `yash.sinha@pilani.bits-pilani.ac.in`
