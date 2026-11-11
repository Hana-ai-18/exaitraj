### Table 8 - Sanity checks (Spearman near 0 is good; the CRT on the label-shuffled model must NOT reject)

| dataset | method | spearman_label_shuffle | spearman_weight_random | crt_p_shuffled | crt_reject_shuffled | acc_shuffled |
|---|---|---|---|---|---|---|
| uci_pen_digits | trajconcept | 0.449 | 0.609 | 0.600 | 0 | 0.090 |
| uci_pen_digits | trajconcept_n | -0.049 | 0.331 | 0.600 | 0 | 0.090 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
