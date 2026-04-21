# M3 Table 4 Summary

Computed from the raw `logs/` and `logs/device_logs/` artifacts in this repo.

## Ranked Completed Candidates

| Family | Config | Runs | Conv. round | Time to 90 [s] | Total energy [J] | Global acc [%] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sword_mu09_m10 | fedprox, champion_sword, lr=0.01, mu=0.9, beta=10.0, epochs=1/1 | exp198/run1, exp409/run2, exp409/run3 | 7 | 79.27 | 69.46 | 91.21 +/- 0.91 |
| refined_fedavg_m10 | fedavg, refined_simplecnn_small_4_8_deep, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp164/run1, exp410/run2, exp410/run3 | 7 | 79.52 | 66.79 | 91.24 +/- 0.70 |
| sword_lr009 | fedprox, champion_sword, lr=0.009, mu=1.0, beta=10.0, epochs=1/1 | exp197/run1, exp406/run2, exp406/run3 | 7 | 79.68 | 69.97 | 91.29 +/- 1.07 |
| sword_mu11 | fedprox, champion_sword, lr=0.01, mu=1.1, beta=10.0, epochs=1/1 | exp200/run1, exp407/run2, exp407/run3 | 8 | 91.40 | 68.74 | 91.74 +/- 0.85 |
| deeper_fedprox_m9 | fedprox, refined_simplecnn_small_4_8_deeper, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp400/run1, exp400/run2, exp400/run3 | 9 | 115.19 | 82.96 | 91.43 +/- 0.70 |

## Recommended Table 4 for Team Black

Runs used: exp198/run1, exp409/run2, exp409/run3

Aggregation method: `fedprox, champion_sword, lr=0.01, mu=0.9, beta=10.0, local_epochs=1/1`

| Metric | Non-IID |
| --- | ---: |
| Convergence time [#rounds] | 7 |
| Avg. time per communication round [s] | 11.16 |
| Total wall clock time [s] | 337.03 |
| Total wall clock time to reach 90.00% [s] | 79.27 |
| Global test accuracy [%] | 91.21 +/- 0.91 |
| RPI avg. energy consumption per round [J] | 41.72 |
| MC1 avg. energy consumption per round [J] | 27.74 |
| Total avg. energy consumption per communication round [J] | 69.46 |
| Avg. amount of communicated data per communication round [MB] | 0.089 |

## Recommended Table 4 for Team Green

Runs used: exp164/run1, exp410/run2, exp410/run3

Aggregation method: `fedavg, refined_simplecnn_small_4_8_deep, lr=0.01, mu=1.0, beta=10.0, local_epochs=1/1`

| Metric | Non-IID |
| --- | ---: |
| Convergence time [#rounds] | 7 |
| Avg. time per communication round [s] | 11.24 |
| Total wall clock time [s] | 339.78 |
| Total wall clock time to reach 90.00% [s] | 79.52 |
| Global test accuracy [%] | 91.24 +/- 0.70 |
| RPI avg. energy consumption per round [J] | 39.62 |
| MC1 avg. energy consumption per round [J] | 27.17 |
| Total avg. energy consumption per communication round [J] | 66.79 |
| Avg. amount of communicated data per communication round [MB] | 0.089 |

## Skipped Families

These candidates reused the right hyperparameters but did not produce three full 30-round runs,
so they were not considered official Table 4 finalists here.

| Family | Config | Runs | Reason |
| --- | --- | --- | --- |
| sword_m7 | fedprox, champion_sword, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp371/run1, exp369/run2, exp373/run3 | incomplete logs ([31, 31, 24]) |
| refined_m7 | fedprox, refined_simplecnn_small_4_8_deep, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp372/run1, exp370/run2, exp374/run3 | incomplete logs ([31, 31, 24]) |
| explorer_fedavg_m7 | fedavg, champion_explorer_v2, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp341/run1, exp375/run2, exp376/run3 | incomplete logs ([31, 31, 13]) |
| explorer_fedmax_m7 | fedmax, champion_explorer_v2, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp342/run1, exp377/run2, exp378/run3 | incomplete logs ([31, 31, 14]) |
| sword_baseline_m9 | fedprox, champion_sword, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp199/run1, exp402/run2, exp402/run3 | incomplete logs ([31, 31, 24]) |
| refined_baseline_m9 | fedprox, refined_simplecnn_small_4_8_deep, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp166/run1, exp403/run2, exp403/run3 | incomplete logs ([31, 31, 24]) |
| explorer_fedavg_m9 | fedavg, champion_explorer_v2, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp341/run1, exp404/run2, exp404/run3 | incomplete logs ([31, 31, 13]) |
| explorer_fedmax_m9 | fedmax, champion_explorer_v2, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp342/run1, exp405/run2, exp405/run3 | incomplete logs ([31, 31, 14]) |
| explorer_fedprox_m10 | fedprox, champion_explorer_v2, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp209/run1, exp408/run2, exp408/run3 | incomplete logs ([31, 31, 13]) |
| deeper_fedavg_m9 | fedavg, refined_simplecnn_small_4_8_deeper, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp399/run1, exp399/run2, exp399/run3 | incomplete logs ([31, 31, 16]) |
| groupnorm_m9 | fedprox, simplecnn_small_4_8_deep_groupnorm, lr=0.01, mu=1.0, beta=10.0, epochs=1/1 | exp401/run1, exp401/run2, exp401/run3 | incomplete logs ([31, 10, 20]) |
