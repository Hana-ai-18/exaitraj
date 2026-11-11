# Các quyết định mình tự chốt (thay cho câu hỏi dành cho thầy) và kết quả đo

Mọi quyết định dưới đây là **giả định của mình**, có ghi lý do. Nếu thầy trả lời khác thì đổi theo thầy; mã đã tham số hoá để đổi dễ.

| # | Câu hỏi | Quyết định | Lý do |
|---|---|---|---|
| 1 | σ, s, θ thực sự dùng | Báo **cả hai profile**. `advisor` (mặc định repo: nhiễu 4,5 / s=1,2 / θ=π/18, toạ độ thô) là profile **chính để so sánh với bảng của thầy**; `calibrated` (σ=1, s=1,5, θ=π/4) là profile phụ để R không suy biến | Paper không công bố tham số; repo là nguồn duy nhất kiểm được |
| 2 | L (số mẫu nhiễu) | `full`: L = 256 (repo 1024) | Giảm 4× chi phí; sai số Monte-Carlo của R/P chấp nhận được. Có thể nâng bằng `cfg.L_perturb` |
| 3 | Cửa sổ trượt | Phân hoạch đều, cửa sổ = 5% số điểm (theo paper Fig.3b) | Repo dùng step=5 cố định, không khớp mô tả paper |
| 4 | Precision | Dùng Eq.8 đúng (AP@k), **không** dùng `ap_at_k` của repo | Repo có lỗi (list → luôn 1.0), đã có test |
| 5 | Surrogate / trọng số | Không còn liên quan (đã xoá XAI4Traj) | Theo yêu cầu bỏ XAI4Traj |
| 6 | Thẻ K | **Đóng băng K v1.0 = đúng 28 mục trong `knowledge.py`** (3 tầng). `The_tri_thuc_K_v1.md` (thẻ cho bão) **không dùng** trong các thí nghiệm này | Bản trong code là bản đang được test; muốn đổi K phải tăng phiên bản |
| 7 | Ngưỡng HCR/VCR | HCR 19°, VCR 0,26 (UbiComp 2008); SR tương đối 0,25 × bước trung vị | Chưa có đơn vị vật lý. Phải ghi vào mục Hạn chế của bài |
| 8 | Kiểm định chính | **CRT có đối chứng g_K**; CRT thô chỉ là chẩn đoán | Xem kết quả đo bên dưới |
| 9 | Ngưỡng twin | `twin_eps_ok` = 0,5σ (không đạt 0,05σ), kèm `leak_gK` | Đề xuất gốc không đạt được với K có mục rời rạc |
| 10 | Số seed | ≥ 3 seed cho mọi kết luận | Một seed = một lần chia dữ liệu + huấn luyện lại |

## Kết quả đo FPR của CRT ở thế giới rỗng (b=0)
Script: `tests/crt_fpr.py` (60 bản lặp, nhãn đổi theo seed, twin cố định; f = HGB trên [K, h_raw]; Pen Digits, họ `freq`, 300 quỹ đạo giữ-ra, B=39).

| Cấu hình twin | ε_K (trung vị, max 28 mục) | FPR CRT thô | FPR CRT đối chứng g_K (khoảng tin cậy 95%) |
|---|---|---|---|
| Adam 100 bước | 0,49σ | 0,82 [0,70; 0,91] | 0,017 [0,00; 0,09] |
| Adam 100 bước + Gauss–Newton 8 vòng | 0,22σ | 0,43 [0,31; 0,57] | 0,050 [0,01; 0,14] |

Kết luận có giới hạn:
* **CRT thô không hợp lệ** (bác bỏ sai 43–82%), đúng như đã nghi.
* **CRT đối chứng g_K nhất quán với mức 5%** trên cấu hình này. Con số "1/4" ở lần đo 4 seed là nhiễu mẫu nhỏ.
* Newton giảm rò rỉ K (FPR thô 0,82 → 0,43), nên đặt `twin_newton_iters = 8` làm mặc định (quick = 4).
* **Phạm vi:** chỉ 1 bộ dữ liệu, 1 họ thế giới, f là oracle HGB (không phải mạng sâu), twin cố định qua các bản lặp. Chưa chứng minh cho 7 bộ × 6 black box. Phải chạy `tests/crt_fpr.py` trên từng bộ (DS=...) trước khi báo p-value của bộ đó, quy tắc: FPR > 0,10 ⇒ chỉ báo effect size.
* Phần còn lại của ε_K (≈0,2σ) do các mục có ngưỡng (HCR, VCR) và mục đếm (NCV, NCA); mục đếm được xử lý bằng chọn ứng viên khớp chính xác, mục tỷ lệ chưa.
