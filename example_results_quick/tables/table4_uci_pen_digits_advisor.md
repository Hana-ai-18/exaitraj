### Table 4 (uci_pen_digits, perturbation profile advisor) - Fidelity R/P/H of phi^MU. n/a = undefined because R=0

| seg | perturb | rf_R | rf_P | rf_H | knn_R | knn_P | knn_H | gru_R | gru_P | gru_H |
|---|---|---|---|---|---|---|---|---|---|---|
| mdl | gaussian | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.100 | 0.625 | 0.172 |
| mdl | rotation | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.150 | 0.713 | 0.248 |
| mdl | scaling | 0.050 | 0.500 | 0.091 | 0 | n/a | n/a | 0.150 | 0.678 | 0.246 |
| rdp | gaussian | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.100 | 0.625 | 0.172 |
| rdp | rotation | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.150 | 0.685 | 0.246 |
| rdp | scaling | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.150 | 0.392 | 0.217 |
| window | gaussian | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.100 | 0.292 | 0.149 |
| window | rotation | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.150 | 0.537 | 0.235 |
| window | scaling | 0.000 | n/a | n/a | 0 | n/a | n/a | 0.150 | 0.206 | 0.174 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
