### Table 4 (uci_pen_digits, perturbation profile calibrated) - Fidelity R/P/H of phi^MU. n/a = undefined because R=0

| seg | perturb | rf_R | rf_P | rf_H | knn_R | knn_P | knn_H | gru_R | gru_P | gru_H |
|---|---|---|---|---|---|---|---|---|---|---|
| mdl | gaussian | 0.000 | n/a | n/a | 0.000 | n/a | n/a | 0.450 | 0.689 | 0.544 |
| mdl | rotation | 0.200 | 0.593 | 0.299 | 0.250 | 0.583 | 0.350 | 0.600 | 0.619 | 0.610 |
| mdl | scaling | 0.050 | 0.500 | 0.091 | 0.050 | 0.500 | 0.091 | 0.250 | 0.583 | 0.350 |
| rdp | gaussian | 0.000 | n/a | n/a | 0.000 | n/a | n/a | 0.400 | 0.678 | 0.503 |
| rdp | rotation | 0.200 | 0.673 | 0.308 | 0.250 | 0.583 | 0.350 | 0.600 | 0.577 | 0.588 |
| rdp | scaling | 0.100 | 0.600 | 0.171 | 0.050 | 1.000 | 0.095 | 0.200 | 0.650 | 0.306 |
| window | gaussian | 0.000 | n/a | n/a | 0.000 | n/a | n/a | 0.300 | 0.656 | 0.412 |
| window | rotation | 0.100 | 0.167 | 0.125 | 0.050 | 1.000 | 0.095 | 0.400 | 0.435 | 0.417 |
| window | scaling | 0.000 | n/a | n/a | 0.050 | 0.167 | 0.077 | 0.200 | 0.573 | 0.297 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
