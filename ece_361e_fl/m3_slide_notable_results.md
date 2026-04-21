# M3 Slide Notable Results

This file is slide-focused: each table shows the three individual runs plus an average row.
The average row is the arithmetic mean of the three run rows.

## Team Black

Most notable result config: `fedprox, champion_sword, lr=0.01, mu=0.9, beta=10.0, local_epochs=1/1`

| Slide row | Source run | Avg. time per communication round [s] | Total wall clock time to reach 90.00% [s] | RPI avg. energy consumption per round [J] | MC1 avg. energy consumption per round [J] | Total avg. energy consumption per communication round [J] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Run 1 | exp198 run1 | 10.27 | 72.65 | 38.97 | 22.75 | 61.73 |
| Run 2 | exp409 run2 | 11.57 | 36.48 | 43.42 | 30.77 | 74.19 |
| Run 3 | exp409 run3 | 11.65 | 143.07 | 42.78 | 29.69 | 72.47 |
| Average | arithmetic mean of the 3 runs | 11.16 | 84.07 | 41.72 | 27.74 | 69.46 |

Official Table 4 averaged-curve convergence: round `7`
 with `time to 90% = 79.27 s`.

## Team Green

Most notable result config: `fedavg, refined_simplecnn_small_4_8_deep, lr=0.01, mu=1.0, beta=10.0, local_epochs=1/1`

| Slide row | Source run | Avg. time per communication round [s] | Total wall clock time to reach 90.00% [s] | RPI avg. energy consumption per round [J] | MC1 avg. energy consumption per round [J] | Total avg. energy consumption per communication round [J] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Run 1 | exp164 run1 | 10.74 | 75.30 | 37.78 | 23.58 | 61.36 |
| Run 2 | exp410 run2 | 11.45 | 35.74 | 40.07 | 28.75 | 68.82 |
| Run 3 | exp410 run3 | 11.55 | 93.73 | 41.00 | 29.19 | 70.20 |
| Average | arithmetic mean of the 3 runs | 11.24 | 68.26 | 39.62 | 27.17 | 66.79 |

Official Table 4 averaged-curve convergence: round `7`
 with `time to 90% = 79.52 s`.

