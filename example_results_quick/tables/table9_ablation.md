### Table 9 - Ablation on the planted world (uci_pen_digits/freq/b=4.0); rec_h = |corr| of the best concept with h*

| group | variant | p_crt | detect | MU | candidates | final | rec_h | eps_K | feasible | useful |
|---|---|---|---|---|---|---|---|---|---|---|
| extractor | L1 | 1 | 0 | 0.087 | 5 | 0.000 | 0.249 | 0.437 | 0.775 | 0.000 |
| extractor | L2 | 1 | 0 | 0.087 | 5 | 0.000 | 0.126 | 0.437 | 0.000 | 0.000 |
| gen | flow | 1 | 0 | 0.087 | 5 | 0.000 | 0.249 | 0.437 | 0.775 | 0.000 |
| gen | gauss | 1 | 0 | 0.146 | 3 | 0.000 | 0.166 | 0.581 | 0.875 | 0.000 |
| gen | naive | 1 | 0 | 0.301 | 4 | 0.000 | 0.161 | 0.813 | 0.900 | 0.000 |
| layer | first | 1 | 0 | 0.087 | 3 | 0.000 | 0.232 | 0.437 | 0.875 | 0.000 |
| layer | last | 1 | 0 | 0.087 | 5 | 0.000 | 0.249 | 0.437 | 0.775 | 0.000 |
| source | both | 1 | 0 | 0.087 | 5 | 0.000 | 0.249 | 0.437 | 0.775 | 0.000 |
| source | disagree | 1 | 0 | 0.087 | 0 | n/a | 0.000 | 0.437 | 0.000 | n/a |
| source | twin | 1 | 0 | 0.087 | 4 | 0.000 | 0.235 | 0.437 | 0.750 | 0.000 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
