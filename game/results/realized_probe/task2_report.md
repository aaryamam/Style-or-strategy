# TASK 2 — realized-history firing probe

Shared realized histories (results/realized_probe/histories.json; 30/bin, late = decisive eff_room 15-29), pooled from realized passA + constrained trajectories (forced by scarcity: olmo_inst has 0 realized late contexts). Probe harness, action set, parser, and firing condition reused verbatim.

Checkpoint-set consistency note: the firing Table-4 has 6 rows; the probe_zt.py re-run script covers 4 (Zephyr/Tulu base+final) and OLMo base+inst were probed separately (probe_runs.jsonl) — hence the draft's 'four free-decode checkpoints' vs Table-4's six. This realized re-run covers all 6.

| checkpoint | marg_eta | eta@early | eta@mid | eta@late | Deta | Deta 95% CI | verdict | (orig synthetic Deta / verdict) |
|---|---|---|---|---|---|---|---|---|
| olmo_base | -0.5366666666666666 | -0.47333333333333333 | -0.5466666666666666 | -0.59 | -0.11666666666666664 | [-0.2066666666666666, -0.020000000000000018] | **FIRES** | (-0.017 / PARTIAL-flat) |
| olmo_inst | -0.004444444444444473 | -0.013333333333333308 | 0.0 | 0.0 | 0.013333333333333308 | [0.0033333333333332993, 0.026666666666666616] | inverted | (0.46 / PARTIAL-inverted) |
| z_base | -0.2911111111111111 | -0.20999999999999996 | -0.42666666666666664 | -0.23666666666666658 | -0.026666666666666616 | [-0.09333333333333338, 0.043333333333333335] | flat | (0.033 / PARTIAL-flat) |
| z_dpo | -0.4733333333333334 | -0.44999999999999996 | -0.47 | -0.5 | -0.050000000000000044 | [-0.13333333333333341, 0.036666666666666625] | flat | (-0.01 / PARTIAL-flat) |
| t_base | -0.5700000000000001 | -0.5833333333333333 | -0.5633333333333332 | -0.5633333333333334 | 0.019999999999999907 | [-0.05999999999999994, 0.09999999999999998] | flat | (0.0 / PARTIAL-flat) |
| t_rlvr | -0.43333333333333335 | -0.38 | -0.42666666666666664 | -0.4933333333333333 | -0.11333333333333329 | [-0.19333333333333336, -0.036666666666666625] | **FIRES** | (0.0 / PARTIAL-flat) |

**Bottom line:** FIRES on: ['olmo_base', 't_rlvr'] — FINDING, investigate.
