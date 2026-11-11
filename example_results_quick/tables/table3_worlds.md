### Table 3 - Planted-answer worlds: detection rate (b>0 power, b=0 false-alarm rate) and recovery of the planted factor h*

| dataset | family | b | power_crt | power_ctrl | power_raw | power_gate | twin_ok | leak | rec_h | concept_prec | loc_err | MU | cand | final | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uci_pen_digits | freq | 0 | 0 | 0 | 1 | 0 | 1 | 0.035 | 0.241 | 0 | n/a | 0.162 | 6 | 0 | 1 |
| uci_pen_digits | freq | 1 | 0 | 0 | 1 | 0 | 1 | 0.015 | 0.065 | 0 | n/a | 0.143 | 3 | 0 | 1 |
| uci_pen_digits | freq | 2 | 0 | 0 | 0 | 0 | 1 | 0.028 | 0.048 | 0 | n/a | 0.113 | 5 | 0 | 1 |
| uci_pen_digits | freq | 4 | 0 | 0 | 0 | 0 | 1 | 0.032 | 0.249 | 0 | n/a | 0.087 | 5 | 0 | 1 |
| uci_pen_digits | where | 0 | 0 | 0 | 1 | 0 | 1 | 0.035 | 0.158 | 0 | 0.217 | 0.162 | 6 | 0 | 1 |
| uci_pen_digits | where | 1 | 0 | 0 | 1 | 0 | 1 | 0.021 | 0.121 | 0 | 0.229 | 0.112 | 6 | 0 | 1 |
| uci_pen_digits | where | 2 | 0 | 0 | 0 | 0 | 1 | 0.025 | 0.114 | 0 | 0.230 | 0.077 | 3 | 0 | 1 |
| uci_pen_digits | where | 4 | 0 | 0 | 0 | 0 | 1 | 0.006 | 0.057 | 0 | 0.222 | 0.094 | 3 | 0 | 1 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
