# Đối chiếu công thức: code ↔ thước đo/giao thức của paper XAI4Traj (Le, Bui, Vo, Neurocomputing 658, 2025) ↔ đề xuất TrajConcept

> **XAI4Traj không còn được chạy như một phương pháp.** Từ phiên bản này chỉ giữ *thước đo* Fidelity (Eq.7–10), phép nhiễu (Eq.2–4) và 3 cách chia đoạn, áp dụng cho bản đồ φ^MU của TrajConcept. Mục A bên dưới là bằng chứng rằng thước đo/phép nhiễu của ta khớp bản gốc; các dòng về surrogate/trọng số chỉ để tham khảo.

Mỗi dòng ghi **vị trí code**, **trạng thái** và **bằng chứng kiểm tra**. Trạng thái: ✔ khớp và có unit test; ⚠ khớp có điều kiện (ghi rõ); ✘ khác paper (cố ý, ghi rõ).
Test chạy bằng `python tests/test_core.py && python tests/test_pipeline.py`.

## A. Công thức gốc của XAI4Traj: paper ↔ repo chính thức ↔ code của ta

Nguồn đối chiếu: **PDF gốc** (`Z_Published_Paper.pdf`, Neurocomputing 658, 131691) và **repo chính thức** `DAIR-Group/XAI4Traj` (đã clone và chạy so sánh số học, xem `tests/test_vs_official.py`).
Quy ước: ✔ khớp có test số học · ⚠ khác có chủ đích/cần quyết định · ✘ phát hiện sai khác giữa paper và repo.

| Mục | Paper (văn bản) | Repo (code chạy ra kết quả) | Code của ta | TT |
|---|---|---|---|---|
| Eq.(2) nhiễu | p̃ = p + ε, ε ~ N(μ,(σs)²); **không nêu σ, s** | `normal(0, std=3) * scale=1.5` ⇒ độ lệch **4,5 đơn vị toạ độ thô** | `apply_op("noise")`; profile `advisor`: 4,5 đơn vị thô (khung thô) | ✔ test: cùng RNG ⇒ trùng khớp từng số |
| Eq.(3) co giãn | p̃ = s·p; **không nêu s** | s = **1,2** (mặc định), quanh **gốc (0,0) thô** | profile `advisor`: s=1,2, quanh gốc thô | ✔ test |
| Eq.(4) xoay | R(θ)p quanh gốc; **không nêu θ** | θ = **π/18 (10°)**, quanh **gốc (0,0) thô** | profile `advisor`: θ=π/18, quanh gốc thô | ✔ test |
| Eq.(5) trọng số | π = exp(−DTW_norm) | `exp(−|z|)`, z = (d−mean)/(std+1e-10) của DTW; std=0 ⇒ 1 | `dtw_weights(…,"abs")` (mặc định) | ✔ test. **Algorithm 1 dòng 14 ghi exp(−d²)**: mâu thuẫn nội bộ paper; có `weight_form="alg"`. *Bản cũ của ta dùng exp(−z) có dấu: SAI, đã bỏ.* |
| Eq.(6) surrogate | "Linear **Ridge** Regression" | **`LogisticRegression()`** (L2, C=1) trên **nhãn cứng** của black box, có `sample_weight`; độ quan trọng = \|hệ số hàng của lớp dự đoán\| | `surrogate="logreg"` (mặc định, như repo) hoặc `"ridge"` (Eq.6 dạng văn bản) | ✘ paper≠repo. Ta theo repo vì repo sinh ra bảng kết quả. Cần hỏi thầy. |
| Số mẫu nhiễu L | "L" (không nêu) | `min(2¹⁰, 2^T)` mẫu, bit ngẫu nhiên | `min(cfg.L_perturb, 2^T)`; full: 256 (repo: 1024) | ⚠ giảm vì thời gian |
| Gating "giải thích được" | Eq.(7) (PDF bị vỡ ký hiệu) | quỹ đạo giải thích được ⇔ surrogate thấy ≥ 2 lớp **và** có ≥ 1 đoạn lật nhãn khi nhiễu riêng đoạn đó. Nếu surrogate chỉ thấy 1 lớp ⇒ `change=0`, `precision=0.0` | `fidelity(...)`: θ=None ⇒ không giải thích được | ✔ **Eq.(7) đã chốt:** Recall = (số quỹ đạo giải thích được)/(số quỹ đạo cần giải thích). Phần tử tổng kép trong PDF là lỗi dàn trang; định nghĩa bằng lời trong paper khớp repo. |
| Eq.(8) Precision | P_X = 1/min(T,R) Σ 1{flip hạng i}·r_i/i | `ap_at_k` | `precision_X` | ✔ test: bằng `ap_at_k` của repo khi dùng đúng. ✘ **LỖI REPO:** `experiment.py` truyền `relevant_class` là *list* (`get_Y()`), nên `ap_at_k` luôn trả **1,0**. Bảng trong paper có P<1 nên số liệu in ra không dùng đường này (hoặc đã sửa). Đừng chạy repo nguyên bản để lấy P. |
| Eq.(9), (10) | trung bình P_X; H = 2PR/(P+R) | — | `fidelity` | ✔ (R=0 ⇒ P, H = NaN, đúng như paper) |
| Xếp hạng đoạn | theo \|θ\| giảm dần | `argsort(abs(coef))[::-1]` | `argsort(−θ)` với θ = \|hệ số\| | ✔ *Bản cũ xếp theo θ có dấu: đã sửa* |
| RDP | ε ∈ [0,005; 0,5] | thư viện `rdp` (ε tuyệt đối). Đoạn đầu chỉ chứa 1 điểm | `_rdp` (ε tương đối theo đường chéo khung bao) | ✔ test: điểm ngắt **trùng thư viện** trên 200 quỹ đạo. Khác biệt: đoạn đầu của repo chỉ 1 điểm, của ta gộp. (Lưu ý: `rdp` của repo gọi `np.cross` trên vector 2-D nên **lỗi với numpy ≥ 2**.) |
| MDL | "Lee et al. [33], L(H)+L(D\|H)" | **không phải TRACLUS**: chi phí = Σ khoảng cách vuông góc tới đường nối đầu–cuối, cắt khi > ε=0,8 (tuyệt đối) | `mdl` = đúng công thức của repo (ε tương đối); TRACLUS còn là `mdl_traclus` | ✔ test: kích thước đoạn **trùng khớp** trên 200 quỹ đạo |
| Cửa sổ trượt | Fig.3b: chia đều thành đoạn bằng nhau, cửa sổ = 5% số điểm | `step=5` cố định với cửa sổ = 5% độ dài ⇒ **đoạn chồng lấn hoặc hở**; ghép lại không còn là quỹ đạo gốc | phân hoạch đều (theo paper) | ⚠ theo paper, không theo repo. Cần hỏi thầy |
| DTW | DTW | `fastdtw` (xấp xỉ) | DTW chính xác (vector hoá) | ✔ test: sai số tương đối 0 trên L=40 |
| Thiết lập | 80/20; 7 bộ; RF, KNN, GRU, LSTM, Transformer, TrajFormer; 3 chia đoạn × 3 phép nhiễu | — | 70/30 + tách khám phá/xác nhận; TrajFormer bản **lite**; **lấy mẫu lại về độ dài cố định L** (paper giữ độ dài gốc) | ⚠ |

**Giao thức bảng của thầy (Bảng 2–8) mà E4 dùng để đánh giá φ^MU:** mỗi bộ dữ liệu một bảng; hàng = (chia đoạn ∈ {mdl, rdp, sliding}) × (nhiễu ∈ {gaussian, rotation, scaling}); cột = 6 black box × (R, P, H) — chỉ cho TrajConcept (`trajconcept`, `trajconcept_n`). Ta chạy **hai profile** tham số nhiễu:
- `advisor`: giá trị mặc định của repo (σ·s = 4,5; s = 1,2; θ = π/18) **trong toạ độ thô**, xoay/co giãn quanh gốc (0,0) thô (cần điểm đầu thô của từng quỹ đạo, lưu ở `TrajSet.origin`). Kiểm tra bằng test với `Perturbation` của repo trên toạ độ thô.
- `calibrated`: toạ độ chuẩn hoá (bước trung vị = 1), σ = 1, s = 1,5, θ = π/4, quanh điểm đầu: nhiễu đủ mạnh để R không suy biến.

⚠ **Paper không công bố σ, s, θ, L**; các giá trị `advisor` là mặc định trong repo, **chưa chắc là giá trị đã dùng để in Bảng 2–8**. Cần hỏi thầy.

## B. Thước đo hiệu chỉnh (đề xuất của chúng tôi, không có trong paper)

| Thước đo | Công thức | Code | Bằng chứng |
|---|---|---|---|
| Quỹ đạo "có thông tin" | 1 ≤ R ≤ T−1 | `fidelity` | `test_fidelity_R_P_H` (frac_informative) |
| Mốc ngẫu nhiên E[P_X] | (1/T) Σ_i [1+(i−1)(R−1)/(T−1)]/i | `expected_random_precision` | Suy ra: E[f_i r_i/i] = (R/T)(1+(i−1)(R−1)/(T−1))/i, chia cho R. **Kiểm bằng liệt kê toàn bộ hoán vị (T=5,R=2) và Monte Carlo 4·10⁴ lần** cho 4 cặp (T,R). |
| P_info − P_rand | trung bình P_X − E[P_X] trên quỹ đạo có thông tin | `fidelity` | như trên |
| Toán tử chung | lật nhãn khi ≥ 1/2 trong m=8 lần nhiễu Gauss / m=8 twin | `explain.flips_common` | — |
| AUC xóa/chèn theo ADE | x = ADE/ADE_full (xóa) hoặc 1−ADE/ADE_full (chèn), y = P(lớp gốc), toán tử xóa = nội suy đường thẳng | `explain.deletion_insertion` | `apply_op("straight")` được test cộng tuyến |
| Rò sang tri thức cũ (`leak_overlap`) | κ_j = |logit g_K(X) − logit g_K(X xóa đoạn j)|; leak = T·Σ_j φ̂_j κ̂_j (φ̂,κ̂ chuẩn hóa tổng 1; **1 = ngẫu nhiên**) | `experiments._leak` | — (chưa có test độc lập; xem mục D) |
| Khối lượng đúng | Σ_{j∈đoạn cài} φ̂_j / (tỉ lệ độ dài đoạn cài); 1 = ngẫu nhiên | `e4_worlds_local` | — |

## C. TrajConcept

| Đại lượng | Công thức | Code | Bằng chứng |
|---|---|---|---|
| Song ánh (p₁,β₁,ν,ω) | decode∘encode = id | `geometry.py` | `test_roundtrip`, sai số < 1e-4 (bước đứng hẳn bị sàn tốc độ 1e-4) |
| Cơ sở tuyến tính V | trực chuẩn; mức & xu hướng của ν, ω | `geometry.linear_basis` | `test_basis_orthonormal_and_K_linear_in_span`: mục K tuyến tính là hàm chính xác của a = Vu |
| Thẻ K khả vi ↔ K thật | 28 cột trùng khớp | `softk.soft_K(soft=False)` | `test_softk_matches_numpy_K`: khớp tới 1e-4 trên cả 28 cột |
| Twin giữ K tuyến tính | a giữ nguyên (kể cả sau cắt ν) | `twins.TwinGenerator` | `test_twin_keeps_linear_K_exactly…` < 1e-5 |
| ε_K | max_j |K_j(X′)−K_j(X)|/σ_j, **K thật**, trung vị theo quỹ đạo (twin tốt nhất trong pool) | `twins.twins`, `pipeline.twin_quality` | báo cả `eps_Kavg_median` (trung bình theo j) |
| MU-share | E_X Var_b g(X′_b) / Var_X g(X), g = log-odds lớp dự đoán | `pipeline.mu_and_crt` | — |
| CRT thô | p = (1+#{T_b ≥ T₀})/(B+1), T = mean log p_f(y|X) | `mu_and_crt` (`p_raw`) | B = số twin (min p = 1/(B+1); B=39 → 0,025) |
| **CRT đối chứng (chính)** | cùng công thức với T = S(f) − S(g_K) | `mu_and_crt` (`p_value`) | g_K không thể dùng thông tin ngoài K ⇒ mọi sụt giảm của g_K dưới twin là rò rỉ K (`leak_gK`); lấy hiệu để khử |
| Khoảng cách tri thức Δ_r | S(f) − S(g_{K_r}), S = −log-loss; bootstrap ghép cặp theo mẫu | `Cell.gate` | CI 95% |
| Tỉ lệ tri thức đã biết | (S_g − S_chance)/(S_f − S_chance) | `Cell.gate` | **> 1 khi f tệ hơn g_K** (không bị cắt, để lộ trường hợp này) |
| LP khái niệm | min ‖v‖₁ + C·Σξ, Dv + ξ ≥ 1, ξ ≥ 0 | `concepts.solve_concept` | HiGHS; báo tỉ lệ khả thi (ξ = 0) |
| Novelty Schut | min_k [err_h(k) − err_a(k)] > 0 | `concepts.schut_novelty` | NaN nếu < 50 mẫu 'máy đúng, g_K sai' |
| ρ, N_K | ‖P⊥v‖/‖v‖, 1−R²_CV(c|K) | `orth_ratio`, `functional_novelty` | — |
| Hữu ích U | −log-loss ngoại mẫu (cross-fit) của [K,c] trừ của [K], CI bootstrap | `concepts.usefulness` | tập xác nhận (f chưa huấn luyện trên đó) |
| Dạy được T_A | Agree(học trò, f) trên mẫu khái niệm giữ lại, trừ đối chứng; kiểm định hoán vị dấu chính xác (n ≤ 12) | `concepts.student_curve`, `paired_signflip_p` | **n_init = 3 ⇒ p nhỏ nhất = 1/8**; ngưỡng `p ≤ 0,125` đã đặt theo đó |

## D. Những điều CHƯA kiểm tra / cần bạn xác nhận

1. **Hỏi thầy:** (i) σ, s, θ, L thực sự dùng cho Bảng 2–8; (ii–iii) không còn liên quan (đã bỏ surrogate/trọng số); (iv) cửa sổ trượt là phân hoạch (paper) hay step=5 (repo); (v) lỗi `ap_at_k` với `relevant_class` dạng list.
2. **Thẻ K1** (Zheng): ngưỡng HCR = 19° và VCR = 0,26 lấy từ **UbiComp 2008** (đã kiểm với bản PDF); **bản TIST 2010** dùng 15° và 0,7 (và SR: 2,5 m/s thay vì 3,4). SR của ta dùng ngưỡng tương đối 0,25 × bước trung vị vì không có đơn vị vật lý. Sinuosity: công thức **đã kiểm** với Benhamou 2004 (S = 2[p((1+c)/(1−c)+b²)]^−½, c = trung bình cos góc rẽ, b = hệ số biến thiên độ dài bước).
3. TrajFormer là bản lite. Mọi phương pháp giải thích khác (XAI4Traj, LIME, SHAP, occlusion, IG) đã **xoá khỏi code** theo yêu cầu.
4. `leak_overlap`, `relevant_mass_ratio`, `iou_topk` chưa có test độc lập.
5. Các ngưỡng đặt trước (ρ₀, N₀, τ, `twin_eps_ok`, `twin_c2st_ok`) là mặc định **chưa hiệu chuẩn**. Đề xuất gốc ε_K ≤ 0,05σ **không đạt được** (300 bước chiếu: ε_K lớn nhất ≈ 0,5σ), nên mặc định 0,5σ kèm `leak_gK`; cần bạn và thầy chốt.
6. **CRT chưa được kiểm chứng là kiểm định hợp lệ:** ở thế giới rỗng (b=0) CRT có đối chứng vẫn bác bỏ ở một số seed (xem `tests/check_crt_power.py`). Dùng `Δ gap` + bộ lọc khái niệm làm bằng chứng chính, CRT là bằng chứng bổ trợ, và luôn báo tỉ lệ bác bỏ ở b=0 của cùng bộ dữ liệu như **mốc null thực nghiệm**.
7. Chưa cài đặt: E13 (thí điểm với người), E14 (bão + môi trường), một số họ thế giới cài sẵn, Sigma-Lognormal, lối tắt cài sẵn, ROAR.

## E. Thước đo thêm theo thiết kế của Schut và cộng sự (PNAS 2025)
| Thước đo | Định nghĩa trong code | Test |
|---|---|---|
| Học trò chưa biết | `agree_pre` = độ trùng top-1 của checkpoint học trò với f trên mẫu khái niệm *trước khi dạy*; `student_naive = agree_pre ≤ max(0,2; 1/C+0,1)`; `teachable` yêu cầu thêm `student_naive` | chạy trong `student_curve` (hồi quy `quick`) |
| Đối chứng 4 ô | `teach4_{a}_to_{b}`: huấn luyện trên mẫu {khái niệm, ngẫu nhiên}, kiểm tra trên {mẫu khái niệm cực trị, vị trí ngẫu nhiên}; độ trùng top-1 với f, trung bình theo `n_init` | hồi quy `quick` (chưa có test số học độc lập) |
| Kiểm định dạy được | `paired_signflip_p`: hoán vị dấu ghép cặp; n ≤ 12 liệt kê đúng (p nhỏ nhất 1/2ⁿ), n > 12 Monte Carlo có hiệu chỉnh +1 | `test_paired_signflip_p_exact` |
| Liều–đáp ứng | `dose_response`: dịch phần dư (ν, ω) theo mẫu hình d với liều α ∈ {0; 0,5; 1; 2}; `rho` = Spearman(α, trung bình c_v), `effect` = (c(α_max) − c(0))/sd c(0); bản `*_ctrl` chiếu lại về K gốc bằng Adam + Gauss–Newton; `k_drift` = trung vị theo quỹ đạo của max_j \|ΔK_j\|/σ_j | `test_dose_response_monotone_on_synthetic_latent` |
| Khối lượng đúng, IoU top-k, leak_overlap | `local_metrics.py` | `test_local_metrics_hand_cases` |
| Xoá/chèn AUC (trục ngân sách ADE) | `explain.deletion_insertion` | `test_deletion_insertion_ranks_true_segment_better` |
| CI bootstrap | `boot_ci` | `test_boot_ci_covers_mean` |
