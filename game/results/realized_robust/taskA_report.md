# TASK A — realized-probe robustness (olmo_base, t_rlvr)

## A1 — larger N (n=10, all 34 decisive histories) + second seed
Decisive late = pool 'late' ∩ round≥6 ∩ eff_room∈[15,29] (original def). The passA decisive universe is only 34, so 3× MORE histories is infeasible; effective N raised via n=10 + 2 seeds.

### olmo_base
- seed 1234: marg=-0.5556451612903226 η@early=-0.5433333333333333 η@late=-0.5882352941176471 Δη=-0.044901960784313744 CI=[-0.0940849673202615,0.004705882352941115] N_late=1360 → **flat**
- seed 777: marg=-0.5963709677419355 η@early=-0.588888888888889 η@late=-0.6161764705882353 Δη=-0.02728758169934631 CI=[-0.07558823529411751,0.01898692810457525] N_late=1360 → **flat**
- **pooled (both seeds, ~2720 late obs)**: marg=-0.576008064516129 η@early=-0.566111111111111 η@late=-0.6022058823529411 Δη=-0.03609477124183014 CI=[-0.071454248366013,-0.002287581699346397] N_late=2720 → **FIRES**

### t_rlvr
- seed 1234: marg=-0.41572580645161294 η@early=-0.38555555555555554 η@late=-0.49558823529411766 Δη=-0.11003267973856212 CI=[-0.15294117647058825,-0.06542483660130716] N_late=1360 → **FIRES**
- seed 777: marg=-0.3681451612903226 η@early=-0.3211111111111111 η@late=-0.49264705882352944 Δη=-0.1715359477124183 CI=[-0.21552287581699348,-0.12934640522875818] N_late=1360 → **FIRES**
- **pooled (both seeds, ~2720 late obs)**: marg=-0.39193548387096777 η@early=-0.3533333333333334 η@late=-0.49411764705882355 Δη=-0.14078431372549016 CI=[-0.17238562091503262,-0.10972222222222228] N_late=2720 → **FIRES**

## A2 — threshold sensitivity grid (both seeds pooled)
| ckpt | cutoff | eff_room band | Δη | 95% CI | N_late | verdict |
|---|---|---|---|---|---|---|
| olmo_base | ≥5 | ≥10 | -0.03249999999999997 | [-0.060138888888888964,-0.006458333333333233] | 5760 | FIRES |
| olmo_base | ≥5 | 15-29 | -0.03318466353677629 | [-0.060606416275430286,-0.005876369327073472] | 5680 | FIRES |
| olmo_base | ≥5 | ≥20 | -0.0030555555555557 | [-0.04111111111111121,0.03388888888888886] | 2400 | flat |
| olmo_base | ≥6 | ≥10 | -0.03609477124183014 | [-0.0704575163398693,-0.0019281045751633474] | 2720 | FIRES |
| olmo_base | ≥6 | 15-29 | -0.03609477124183014 | [-0.06934640522875823,-0.0022058823529411686] | 2720 | FIRES |
| olmo_base | ≥6 | ≥20 | -0.015555555555555545 | [-0.06388888888888888,0.033055555555555616] | 1200 | flat |
| t_rlvr | ≥5 | ≥10 | -0.09180555555555547 | [-0.11624999999999996,-0.06770833333333337] | 5760 | FIRES |
| t_rlvr | ≥5 | 15-29 | -0.09173708920187784 | [-0.11611893583724575,-0.06762910798122068] | 5680 | FIRES |
| t_rlvr | ≥5 | ≥20 | -0.09499999999999997 | [-0.1280555555555556,-0.06277777777777771] | 2400 | FIRES |
| t_rlvr | ≥6 | ≥10 | -0.14078431372549016 | [-0.17052287581699344,-0.1108006535947712] | 2720 | FIRES |
| t_rlvr | ≥6 | 15-29 | -0.14078431372549016 | [-0.17236928104575155,-0.11153594771241826] | 2720 | FIRES |
| t_rlvr | ≥6 | ≥20 | -0.1333333333333333 | [-0.17500000000000004,-0.09111111111111103] | 1200 | FIRES |

## A3 — own-history probe (fired checkpoints only)
own-DECISIVE = own realized round≥6 ∩ eff_room≥15 (scarce); own-REACHED = own realized round≥6 any eff_room (what it actually reaches on-policy). 0 decisive-late came from either model's OWN CONSTRAINED runs.

### olmo_base  (own early=90, own decisive-late=2, own reached-late=30)
- own-DECISIVE (2 hist): marg=-0.5668478260869565 η@early=-0.5655555555555556 η@late=-0.625 Δη=-0.05944444444444441 CI=[-0.2138888888888889,0.12055555555555564] N_late=80 → **flat**
- own-REACHED (30 hist): marg=-0.5691666666666667 η@early=-0.5655555555555556 η@late=-0.5800000000000001 Δη=-0.014444444444444482 CI=[-0.06666666666666665,0.03833333333333322] N_late=1200 → **flat**

### t_rlvr  (own early=90, own decisive-late=6, own reached-late=90)
- own-DECISIVE (6 hist): marg=-0.35624999999999996 η@early=-0.3466666666666667 η@late=-0.5 Δη=-0.15333333333333332 CI=[-0.24277777777777776,-0.06222222222222229] N_late=240 → **FIRES**
- own-REACHED (90 hist): marg=-0.4227777777777778 η@early=-0.3466666666666667 η@late=-0.49888888888888894 Δη=-0.15222222222222226 CI=[-0.18388888888888888,-0.11888888888888893] N_late=3600 → **FIRES**

## Revised fire verdict
| ckpt | (i) larger N+2nd seed | (ii) threshold grid | (iii) own-history | BOTTOM LINE |
|---|---|---|---|---|
| olmo_base | pooled=FIRES (s1=flat,s2=flat) | 4/6 cells fire | decisive=flat(N=80), reached=flat(N=1200) | **fires only under pooled/counterfactual histories** |
| t_rlvr | pooled=FIRES (s1=FIRES,s2=FIRES) | 6/6 cells fire | decisive=FIRES(N=240), reached=FIRES(N=3600) | **fires (robust)** |
