### Table 2 - Twin quality (C2ST close to 0.5 and small eps_K are good; leak_gK = K-only model loss caused by the twin)

| dataset | gen | n | eps_K_median | eps_K_mean_best | eps_Kavg_median | accept_rate | C2ST | nn_ratio_median | valid_frac | diversity | leak_gK | usable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uci_pen_digits | flow | 115 | 0.413 | 0.805 | 0.078 | 0.009 | 0.539 | 1.107 | 1.000 | 0.505 | 0.084 | 1 |
| uci_pen_digits | gauss | 115 | 0.570 | 1.144 | 0.125 | 0.002 | 0.483 | 1.193 | 1.000 | 0.737 | 0.151 | 0 |
| uci_pen_digits | naive | 115 | 0.797 | 1.530 | 0.167 | 0.002 | 0.635 | 1.410 | 0.999 | 1.265 | 0.293 | 0 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
