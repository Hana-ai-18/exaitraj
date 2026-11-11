### Table 1 - Knowledge gap (log-loss; delta_nll > 0 means the black box beats the K-only model)

| dataset | bb | tier | usable | bacc_f | bacc_g | S_f | S_g | delta_nll | delta_sd | ci_lo | ci_hi | known_ratio | n_seeds | pass_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uci_pen_digits | gru | 0 | 1 | 0.666 | 0.858 | -1.096 | -0.419 | -0.677 | n/a | -0.846 | -0.513 | 1.561 | 1 | False |
| uci_pen_digits | gru | 1 | 1 | 0.666 | 0.904 | -1.096 | -0.306 | -0.790 | n/a | -0.967 | -0.657 | 1.655 | 1 | False |
| uci_pen_digits | gru | 2 | 1 | 0.666 | 0.954 | -1.096 | -0.167 | -0.929 | n/a | -1.103 | -0.805 | 1.770 | 1 | False |
| uci_pen_digits | knn | 0 | 1 | 0.938 | 0.858 | -0.182 | -0.419 | 0.237 | n/a | 0.156 | 0.321 | 0.888 | 1 | False |
| uci_pen_digits | knn | 1 | 1 | 0.938 | 0.904 | -0.182 | -0.306 | 0.123 | n/a | 0.060 | 0.192 | 0.942 | 1 | False |
| uci_pen_digits | knn | 2 | 1 | 0.938 | 0.954 | -0.182 | -0.167 | -0.015 | n/a | -0.075 | 0.031 | 1.007 | 1 | False |
| uci_pen_digits | rf | 0 | 1 | 0.950 | 0.858 | -0.180 | -0.419 | 0.240 | n/a | 0.158 | 0.331 | 0.887 | 1 | False |
| uci_pen_digits | rf | 1 | 1 | 0.950 | 0.904 | -0.180 | -0.306 | 0.126 | n/a | 0.051 | 0.198 | 0.941 | 1 | False |
| uci_pen_digits | rf | 2 | 1 | 0.950 | 0.954 | -0.180 | -0.167 | -0.012 | n/a | -0.082 | 0.037 | 1.006 | 1 | False |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
