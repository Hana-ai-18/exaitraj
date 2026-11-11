### Table 4 summary (profile calibrated): mean over 3 segmentations x 3 perturbations. Pinfo_minus_rand = precision above a random map; n/a = undefined because R=0

| dataset | bb | method | R | P | H | Pinfo_minus_rand | frac_informative |
|---|---|---|---|---|---|---|---|
| uci_pen_digits | gru | phi^MU | 0.378 | 0.607 | 0.447 | 0.122 | 0.378 |
| uci_pen_digits | knn | phi^MU | 0.078 | 0.639 | 0.176 | 0.196 | 0.078 |
| uci_pen_digits | rf | phi^MU | 0.072 | 0.507 | 0.199 | 0.051 | 0.072 |
| uci_pen_digits | gru | phi^MU normalised | 0.378 | 0.584 | 0.444 | 0.100 | 0.378 |
| uci_pen_digits | knn | phi^MU normalised | 0.078 | 0.736 | 0.167 | 0.293 | 0.078 |
| uci_pen_digits | rf | phi^MU normalised | 0.072 | 0.529 | 0.193 | 0.073 | 0.072 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
