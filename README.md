# TrajConcept – mã thực nghiệm

TrajConcept tìm những **khái niệm mà black box dùng nhưng thẻ tri thức K (28 đặc trưng chuyên gia) không chứa**, trên quỹ đạo, và kiểm tra chúng bằng: (1) cổng khoảng cách tri thức, (2) thế giới cài đáp án (biết sự thật nền), (3) học trò chưa biết, (4) bản đồ quan trọng φ^MU chấm bằng Fidelity R/P/H của thầy.
Không còn so sánh với XAI4Traj hay phương pháp nào khác.

## 0. Trạng thái trung thực (đọc trước)
* **Đã kiểm tra:** 11 + 4 + 6 unit test (công thức, metric, so khớp repo chính thức của thầy); toàn bộ E1–E15 chạy xuyên suốt ở preset `quick` (Pen Digits; RF, k-NN, GRU) trên CPU; CRT đối chứng g_K có FPR 0,05 [0,01; 0,14] ở thế giới rỗng (60 bản lặp, **chỉ Pen Digits, họ `freq`, f là oracle HGB**).
* **Chưa kiểm tra:** chạy `medium`/`full` trên T4 và thời gian của chúng; kết quả khoa học (có khoảng cách tri thức không, mạng sâu có phát hiện được yếu tố cài sẵn không); FPR của CRT trên 6 bộ còn lại; bản đồ φ^MU chưa qua kiểm tra Adebayo ở thiết lập mạnh.
* **Số liệu của `quick` KHÔNG phải bằng chứng.** `quick` chỉ để biết mã chạy được; GRU ở `quick` chưa học được yếu tố cài sẵn nên power = 0 là bình thường. Mọi hình của `quick` có dòng chữ đỏ "QUICK PRESET".
* **Chưa cài đặt:** nghiên cứu người (E13), chọn tầng bằng hạng biểu diễn, ablation cho việc g_K được tinh chỉnh nhiều hơn f. TrajFormer là bản lite.
* Các quyết định mình tự chốt thay cho thầy (σ/s/θ/L, cửa sổ trượt, K v1.0, ngưỡng…) và kết quả đo FPR: `QUYET_DINH_THAY_THAY_THAY.md`. Đối chiếu công thức: `CONG_THUC_DOI_CHIEU.md`.

## 1. Chia việc CPU / GPU

| Giai đoạn | Chạy ở đâu | Lệnh | Ghi chú |
|---|---|---|---|
| Unit test | **CPU, máy bạn** | `python tests/test_core.py && python tests/test_pipeline.py` | phải in toàn `OK` |
| Đối chiếu repo của thầy (tùy chọn) | CPU | `git clone --depth 1 https://github.com/DAIR-Group/XAI4Traj /tmp/ref/repo && pip install rdp && python tests/test_vs_official.py` | chỉ để kiểm thước đo |
| E1 (RF, k-NN), E2, E4 (RF, k-NN) | **CPU, máy bạn** | `python run.py --preset full --stage cpu --out results_cpu --cache cache_cpu` | chạy lại được; vài giờ (chưa đo) |
| FPR của CRT cho từng bộ | CPU | `DS=animals REPS=60 python tests/crt_fpr.py` | bộ nào FPR > 0,10 thì chỉ báo effect size |
| E1 (GRU/LSTM/Transformer/TrajFormer), E3, E4, E4w, E5/E6, E7, E8, E9, E10 | **GPU, Kaggle T4** | notebook `TrajConcept_Kaggle.ipynb` | `--stage gpu`, có ngân sách giờ |
| Ghép + bảng + 5 hình | CPU (hoặc cuối notebook) | `python merge_results.py merged results_cpu results_gpu && python run.py --preset full --stage report --out merged` | không huấn luyện gì |

`--stage cpu|gpu|report|all`, `--budget-hours H` (dừng sạch rồi chạy lại cùng lệnh để tiếp tục), `--datasets`, `--bbs`, `--seeds`. `python run.py --help` có đầy đủ.

## 2. Chạy trên máy của bạn (CPU)
```bash
git clone <repo của bạn> && cd <repo>
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_core.py && python tests/test_pipeline.py
python run.py --preset quick --stage all --out results_quick --cache cache_quick      # ~15 phút, chỉ kiểm tra chạy được
python run.py --preset full --stage cpu --out results_cpu --cache cache_cpu            # phần CPU của bản thật
```
Dữ liệu (7 bộ) tải tự động từ GitHub release của `yupidevs/trajectory-datasets` (không cần pactus/TensorFlow).

## 3. Chạy phần GPU trên Kaggle (qua GitHub)
1. Đẩy repo lên GitHub: **không** đẩy `data/`, `cache*/`, `results*/` (đã có `.gitignore`). Gốc repo phải chứa `trajconcept/`, `run.py`, `tests/`, `merge_results.py`, `requirements.txt`.
2. Tạo notebook Kaggle từ `TrajConcept_Kaggle.ipynb` (tạo lại bằng `python make_notebook.py`): Accelerator **GPU T4 x1**, Internet **On**.
3. Cell 1: `GIT_URL` đã trỏ tới `https://github.com/Hana-ai-18/exaitraj.git`. Nếu repo private: tạo token GitHub (quyền đọc nội dung), lưu ở Kaggle Add-ons -> Secrets với tên `GITHUB_TOKEN` và bật cho notebook.
4. Cell 3: đặt `PRESET = "quick"` để thử một lượt → `"medium"` để đo thời gian → `"full"`.
5. Chạy tuần tự cell 4 → 10. Hết giờ Kaggle: tải `results_gpu.zip` và `cache_gpu.zip` (cell cuối), upload làm Dataset, mở phiên mới và chạy lại từ cell 1: cell 3 tự khôi phục và bỏ qua phần đã xong.
6. Muốn gộp kết quả CPU: upload `results_cpu/` (hoặc zip) làm Dataset; cell 10 tự gộp và vẽ hình.

## 4. Năm hình (tiếng Anh) – mỗi hình một luận điểm
| Hình | Câu hỏi | Từ |
|---|---|---|
| Fig 1 – Knowledge gate | Black box có biết hơn mô hình tốt nhất chỉ dùng thẻ K không? | E1 |
| Fig 2 – Planted-answer | Phương pháp có im lặng khi không có gì (b=0), tìm đúng và định vị đúng khi có (b>0)? | E3, E4w |
| Fig 3 – Teachability | Học trò chưa biết có học được khái niệm, và chỉ từ mẫu khái niệm? (đối chứng 4 ô) | E5/E6 |
| Fig 4 – Fidelity R/P/H | Bản đồ φ^MU có thật sự chỉ ra đoạn mà mô hình dựa vào (so với bản đồ ngẫu nhiên)? | E4 |
| Fig 5 – Worked example | Quỹ đạo gốc vs twin đẳng-K và bản đồ φ^MU (theo đoạn và theo điểm) | E4 |

Hình nào chưa có dữ liệu thì bị bỏ qua, không vẽ hình rỗng.

## 5. Cách đọc số liệu (vì sao có 0 và n/a)
* **`n/a` không phải 0:** nghĩa là *không xác định*. Ví dụ P và H cần R > 0 (nếu nhiễu không làm đổi dự đoán của quỹ đạo nào thì không có gì để chấm bản đồ); `delta_sd` cần ≥ 2 seed; các cột khái niệm chỉ có cho mạng nơ-ron (RF, k-NN không có khái niệm).
* **R = 0** với RF/k-NN ở profile `advisor` (nhiễu nhẹ, mặc định repo của thầy) là bình thường; profile `calibrated` (mạnh hơn) cho R > 0.
* Bảng in số nguyên không có phần thập phân; cột 1/0 như `usable` là tỉ lệ số lần chạy.
* `power_ctrl` ở hàng `b=0` là **tỉ lệ báo động nhầm** (phải ≲ 0,05). `power_raw` cao ở b=0 là bình thường (CRT thô không hợp lệ vì twin rò rỉ K).
* `usable` = bacc ≥ chance + 0,15, ECE < 0,10 và (chỉ với mạng nơ-ron) độ lệch train–test < 0,15. RF/k-NN không bị điều kiện độ lệch vì chúng ghi nhớ tập train (train acc ≈ 1). Dòng `usable` cũ trong `e1_gate.csv` được tính lại mỗi lần chạy E1.
* Mọi kết luận chỉ dùng cặp (bộ, black box) có `usable=1` (Bảng 1) **và** twin dùng được (Bảng 2). `delta_nll > 0` với `ci_lo > 0` ở `tier=2` mới gọi là "máy hơn thẻ K". Nên chạy ≥ 3 seed trước khi viết bài.
* `dose_*_ctrl` (thử liều–đáp ứng ở Bảng E5): chỉ tin khi `dose_kdrift_ctrl_max` ≲ 0,5σ.

## 6. Thí nghiệm
| Mã | Nội dung |
|---|---|
| E1 | Cổng khoảng cách tri thức Δ = S(f) − S(g_K), bootstrap ghép cặp, theo 3 tầng K |
| E2 | Chất lượng twin: ε_K, leak_gK, C2ST |
| E3 | Thế giới cài đáp án: power / FPR (b=0) / khôi phục / định vị |
| E4, E4w | Fidelity R/P/H của φ^MU: 3 chia đoạn × 3 phép nhiễu × 2 profile; định vị trong thế giới `where` |
| E5, E6 | Khái niệm thật → 4 bộ lọc (hữu ích, mới, dạy được, không phải đường tắt); học trò chưa biết; đối chứng 4 ô; liều–đáp ứng |
| E7 | Tái phát hiện mục K bị bỏ |
| E8 | Kiểm tra Adebayo (xáo nhãn/xáo trọng số) + CRT trên mô hình xáo nhãn |
| E9, E10 | Ablation; độ nhạy (B, tầng K, chia đoạn, độ mạnh nhiễu) |
| E15 | Nguyên nhân thất bại theo từng (bộ, black box) |

Thiết kế E5/E6 lấy từ Schut và cộng sự (PNAS 2025): tối ưu ℓ₁, novelty SVD, học trò chưa biết, đối chứng 4 ô.

## 7. Cấu trúc
```
run.py            CLI (--stage cpu|gpu|report|all, --budget-hours)
merge_results.py  gộp results_cpu + results_gpu
make_notebook.py  sinh notebook Kaggle
trajconcept/      config, data, geometry, knowledge (thẻ K), softk, twins, models, concepts, explain, local_metrics,
                  xai_metrics, worlds, pipeline, experiments, plots, store
tests/            test_core, test_pipeline, test_vs_official, crt_fpr (đo FPR), check_crt_power
```

## 8. Khi có lỗi
* Sửa code xong thì xóa thư mục `cache*/` (cache pickle không tự vô hiệu hóa khi đổi kiến trúc).
* `rdp` hỏng với numpy ≥ 2 chỉ ảnh hưởng `test_vs_official.py` (mã chính dùng bản RDP tự cài).
* Kaggle không tải được dữ liệu: kiểm tra Internet = On.
* Twin kém ở một số bộ (ví dụ `animals`: ε_K lớn): Bảng 2 sẽ đánh dấu `usable=0`, bộ đó không dùng để kết luận CRT.

## 9. Thời gian đo được (không phải ước lượng)
Từ lần chạy `--preset full --stage cpu` trên máy của bạn: huấn luyện g_K (RF) tốn ~9 s (animals), 55–750 s (geolife), 310–420 s (hurdat2), ~770 s (pen digits) **mỗi seed**; k-NN dùng lại g_K đã cache. Với 7 bộ × 5 seed, riêng E1 trên CPU có thể mất nhiều giờ. Nếu cần rút ngắn: `--seeds 0,1,2`, hoặc chạy từng bộ (`--datasets animals`) trên các cửa sổ terminal khác nhau (mỗi cái dùng `--out`/`--cache` riêng rồi gộp bằng `merge_results.py`).
Trên Kaggle T4, preset `quick`: E3 mất ~52 s mỗi ô (bộ × họ × b × seed); preset `full` nặng hơn nhiều (n_twin, số bước chiếu, B của CRT đều lớn hơn) và có 3 bộ × 2 họ × 5 mức b × 5 seed × 2 black box = 300 ô. **Hãy chạy `medium` trước và nhân lên**; nếu vượt hạn mức GPU thì giảm `--seeds`, `worlds_b` hoặc `worlds_bbs` trong `trajconcept/config.py`.

## 10. Ghi chú v-rich (đầu vào giàu đặc trưng cho mạng sâu)
`deep_rich=True` (mặc định): GRU/LSTM/Transformer nhận 6 kênh (x, y, dx, dy, tốc độ, góc rẽ) thay vì 4.
Đo trên world planted b=4 (60 epoch, h=96, Pen Digits): gap(g_K−f) GRU .042→.061, LSTM .009→.026,
Transformer .024→.061 (freq); world 'where' Transformer −.012→.040. Vẫn chỉ ~25–35% oracle gap
(oracle freq≈.17, where≈.28) -> mạng sâu vẫn học chưa hết yếu tố cài sẵn; cần chạy preset medium/full để biết.
Lưu ý: cache cũ không dùng lại được (deep_rich nằm trong CACHE_FIELDS).

## 11. Kiểm chứng phương pháp (verify.py)
`python verify.py --smoke --out v_smoke --cache c_smoke` (thử script, CPU, ~2 phút, KHÔNG kết luận) rồi notebook `TrajConcept_Verify.ipynb`
(hoặc `python verify.py --out verify_out --cache verify_cache --budget-hours 8` trên T4; chạy lại để tiếp tục).
Ra `claims.csv` / `claims.md`: PASS / FAIL / INCONCLUSIVE cho 6 tuyên bố (ngưỡng đặt trước trong `trajconcept/claims.py::THRESH`).
Thứ tự: null (C1) -> worlds (C2, C3) -> e1 (C5) -> e8 (C4) -> e5 (C5) -> e4, e4w (C6). Phương pháp chính của C4/C6 là `trajconcept_n`
(chốt trước khi chạy; `trajconcept` thô chỉ là INFO). INCONCLUSIVE không phải PASS: ví dụ C5 trên Pen Digits có thể INCONCLUSIVE vì K đã đủ.
Thời gian ~4-6 giờ T4 là ƯỚC LƯỢNG chưa đo.
