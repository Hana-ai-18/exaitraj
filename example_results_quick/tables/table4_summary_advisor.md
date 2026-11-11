### Table 4 summary (profile advisor): mean over 3 segmentations x 3 perturbations. Pinfo_minus_rand = precision above a random map; n/a = undefined because R=0

| dataset | bb | method | R | P | H | Pinfo_minus_rand | frac_informative |
|---|---|---|---|---|---|---|---|
| uci_pen_digits | gru | phi^MU | 0.133 | 0.528 | 0.207 | 0.075 | 0.133 |
| uci_pen_digits | knn | phi^MU | 0.000 | n/a | n/a | n/a | 0.000 |
| uci_pen_digits | rf | phi^MU | 0.006 | 0.500 | 0.091 | -0.021 | 0.006 |
| uci_pen_digits | gru | phi^MU normalised | 0.133 | 0.495 | 0.204 | 0.042 | 0.133 |
| uci_pen_digits | knn | phi^MU normalised | 0.000 | n/a | n/a | n/a | 0.000 |
| uci_pen_digits | rf | phi^MU normalised | 0.006 | 0.250 | 0.083 | -0.271 | 0.006 |

*n/a = not defined (it is NOT zero): P and H need R > 0; delta_sd needs >= 2 seeds; concept columns exist only for neural networks. Numbers that are all integers are shown without decimals; 1/0 columns such as usable are fractions of runs.*
