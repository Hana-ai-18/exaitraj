### Table 15 - Failure analysis per (dataset, black box)

| dataset | bb | delta_lo | delta | usable | C2ST | eps_K | lp_feasible | cand | final | main_cause |
|---|---|---|---|---|---|---|---|---|---|---|
| uci_pen_digits | gru | -1.103 | -0.929 | 1 | 0.539 | 0.413 | 0.775 | 2.000 | 0.000 | no knowledge gap (95% CI of delta includes 0) |
| uci_pen_digits | knn | -0.075 | -0.015 | 1 | 0.539 | 0.413 | n/a | n/a | n/a | no knowledge gap (95% CI of delta includes 0) |
| uci_pen_digits | rf | -0.082 | -0.012 | 1 | 0.539 | 0.413 | n/a | n/a | n/a | no knowledge gap (95% CI of delta includes 0) |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
