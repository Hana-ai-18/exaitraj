"""Generate the Kaggle notebook:  python make_notebook.py  ->  TrajConcept_Kaggle.ipynb"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# TrajConcept – phần chạy trên GPU (Kaggle T4)

**Trước khi chạy:** Settings → Accelerator = **GPU T4 x1**, Internet = **On**.

**Chia việc CPU / GPU**
| Chạy ở đâu | Gồm | Lệnh |
|---|---|---|
| **Máy của bạn (CPU)** | unit test, E1 (RF, k-NN), E2 (chất lượng twin), E4 (RF, k-NN), kiểm tra FPR của CRT | `python run.py --preset full --stage cpu` (xem README) |
| **Notebook này (GPU)** | E1 (GRU, LSTM, Transformer, TrajFormer), E3, E4, E4w, E5/E6, E7, E8, E9, E10 | các cell bên dưới |
| Ghép + vẽ hình | gộp kết quả CPU và GPU, bảng, 5 hình | cell cuối |

Mọi cell **chạy lại được và tự tiếp tục**: kết quả lưu từng đơn vị trong `results/*.csv`, mô hình đã huấn luyện lưu trong `cache/`. Hết giờ phiên Kaggle thì tải `results_gpu.zip` + `cache_gpu.zip`, upload làm Dataset, mở phiên mới và chạy lại từ cell 1: cell 3 tự khôi phục.""")

md("## 1. Lấy mã từ GitHub")
code("""import os, sys, glob, subprocess, shutil
GIT_URL = "https://github.com/YOUR_USER/YOUR_REPO.git"     # <-- ĐỔI thành repo của bạn (repo chứa thư mục trajconcept/, run.py, tests/ ở gốc)
BRANCH = "main"
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath("./runs")
os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
PKG = os.path.join(WORK, "repo")
if os.path.exists(os.path.join(PKG, ".git")):
    print(subprocess.run(["git", "-C", PKG, "pull", "--ff-only"], capture_output=True, text=True).stdout)
elif "YOUR_USER" in GIT_URL:
    # Phương án dự phòng: gói zip upload làm Kaggle Dataset (hoặc đặt cạnh notebook)
    zips = glob.glob("/kaggle/input/*/trajconcept_pkg*.zip") + glob.glob("./trajconcept_pkg*.zip")
    assert zips, "Hãy đổi GIT_URL ở trên, hoặc upload trajconcept_pkg.zip làm Dataset."
    shutil.unpack_archive(zips[0], WORK); PKG = os.path.join(WORK, "trajconcept_pkg")
else:
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, GIT_URL, PKG], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
assert os.path.exists(os.path.join(PKG, "run.py")), f"Không thấy run.py trong {PKG}"
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "requests", "scipy", "scikit-learn", "pandas", "matplotlib"], check=False)
os.environ["TRAJ_DATA_DIR"] = os.path.join(WORK, "data")
print("PKG =", PKG, "| WORK =", WORK)""")

md("## 2. Kiểm tra GPU")
code("""import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU)")
if not torch.cuda.is_available():
    print("CẢNH BÁO: không có GPU. Preset 'quick' vẫn chạy được; 'medium'/'full' sẽ rất chậm.")""")

md("## 3. Cấu hình phiên này")
code("""PRESET = "quick"     # "quick" (kiểm tra chạy được, vài phút) -> "medium" (đo thời gian) -> "full" (thật)
SEEDS = ""           # để trống = theo preset; ví dụ "0,1,2"
DATASETS = ""        # để trống = tất cả; ví dụ "uci_characters,animals"
BBS = ""             # để trống = cả 4 mạng sâu (gru,lstm,transformer,trajformer); ví dụ "gru,transformer"
BUDGET_H = 8.0       # tổng số giờ cho phiên này; hết giờ thì dừng sạch (chạy lại cell để tiếp tục)
OUT, CACHE = os.path.join(WORK, "results"), os.path.join(WORK, "cache")

# Khôi phục kết quả/cache của phiên trước (nếu bạn upload results_gpu.zip / cache_gpu.zip làm Dataset)
for pat, dst in (("/kaggle/input/*/results_gpu*.zip", OUT), ("/kaggle/input/*/cache_gpu*.zip", CACHE)):
    for z in glob.glob(pat):
        shutil.unpack_archive(z, dst); print("đã khôi phục", z, "->", dst)

import time, pandas as pd
from IPython.display import display, Markdown, Image
T_START = time.time()
def run(exp, extra=""):
    cmd = f'{sys.executable} -u {PKG}/run.py --preset {PRESET} --stage gpu --exp {exp} --out {OUT} --cache {CACHE} --no-plots'
    if SEEDS: cmd += f" --seeds {SEEDS}"
    if DATASETS: cmd += f" --datasets {DATASETS}"
    if BBS: cmd += f" --bbs {BBS}"
    left = BUDGET_H - (time.time() - T_START) / 3600
    if left <= 0:
        print(f"[{exp}] hết ngân sách {BUDGET_H} h của phiên - bỏ qua; chạy lại ở phiên sau"); return
    cmd += f" --budget-hours {left:.3f} {extra}"
    print(">>", cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=PKG)
    for line in p.stdout:
        if "Warning" in line or "detach" in line or "self.T" in line: continue
        print(line, end="")
    p.wait(); assert p.returncode == 0, f"{exp} lỗi (mã {p.returncode})"
def show(name):
    p = f"{OUT}/tables/{name}.md"
    display(Markdown(open(p).read()) if os.path.exists(p) else f"(chưa có {name})")""")

md("## 4. Unit test (phải xanh trước khi chạy tiếp)")
code("""for t in ("test_core", "test_pipeline"):
    r = subprocess.run([sys.executable, f"{PKG}/tests/{t}.py"], capture_output=True, text=True, cwd=PKG)
    print(r.stdout[-1200:], r.stderr[-1200:]); assert r.returncode == 0, f"{t} lỗi - dừng lại"
print("TẤT CẢ TEST XANH")""")

md("## 5. E1 – Máy có biết hơn thẻ K không? (cổng, làm đầu tiên) → Bảng 1")
code('run("e1"); show("table1_gate")')
md("""`delta_nll > 0` và `ci_lo > 0` ở `tier=2` nghĩa là máy hơn *mô hình tốt nhất chỉ dùng thẻ K*. Nếu hầu hết dòng `tier=2` có `ci_lo ≤ 0` thì **không có khoảng cách tri thức**: bài chuyển sang kết luận âm. `n/a` ≠ 0 (ví dụ `delta_sd` cần ≥ 2 seed).""")

md("## 6. E3 – Thế giới cài đáp án (biết sự thật nền) → Bảng 3")
code('run("e3"); show("table3_worlds")')
md("""Hàng `b=0` là thế giới rỗng: `power_ctrl` ở đó là **tỉ lệ báo động nhầm** (phải ≲ 0,05). `power_raw` (CRT thô) cao ở b=0 là *bình thường*: twin rò rỉ K nên CRT thô không hợp lệ, vì vậy kiểm định chính là CRT đối chứng g_K.""")

md("## 7. E4 – Độ trung thực R/P/H của bản đồ quan trọng φ^MU → Bảng 4 (phần nặng nhất)")
code('run("e4"); run("e4w")')
code("""for f in sorted(glob.glob(f"{OUT}/tables/table4_summary_*.md")) + [f"{OUT}/tables/table4w_local.md"]:
    if os.path.exists(f): display(Markdown(open(f).read()))""")
md("""`R` = tỉ lệ quỹ đạo mà nhiễu làm đổi dự đoán; `P` = độ chính xác của bản đồ; `H` = trung bình điều hòa. **`P`, `H` = n/a khi `R` = 0** (không có gì để giải thích; không phải 0). Có hai mức nhiễu: `advisor` (mặc định repo của thầy, nhẹ) và `calibrated` (mạnh hơn).""")

md("## 8. E5/E6 – Khái niệm trên dữ liệu thật qua 4 bộ lọc + dạy được")
code('run("e5")')
code("""f5 = pd.read_csv(f"{OUT}/e5_funnel.csv") if os.path.exists(f"{OUT}/e5_funnel.csv") else pd.DataFrame()
display(f5.groupby(["dataset","bb"]).mean(numeric_only=True).round(3) if not f5.empty else "chưa có")
cp = f"{OUT}/e5_concepts.csv"
if os.path.exists(cp):
    c5 = pd.read_csv(cp)
    cols = [x for x in ["dataset","bb","seed","id","useful","novel","teachable","student_naive","agree_pre","teach4_concept_to_concept","teach4_random_to_concept","teach4_concept_to_random","teach4_random_to_random","dose_rho_ctrl","dose_effect_ctrl","dose_kdrift_ctrl_max"] if x in c5.columns]
    display(c5[cols].round(3))""")
md("""`student_naive=True`: học trò trước khi dạy gần mức ngẫu nhiên. `teach4_*`: kỳ vọng `concept_to_concept` > `random_to_concept`. `dose_*_ctrl`: điểm khái niệm tăng khi dịch quỹ đạo theo mẫu hình *sau khi* giữ K; **chỉ tin khi `dose_kdrift_ctrl_max` ≲ 0,5σ**.""")

md("## 9. E7 (tái phát hiện), E8 (kiểm tra hợp lệ), E9 (ablation), E10 (độ nhạy)")
code('run("e7"); run("e8"); run("e9"); run("e10")')
code('for n in ["table7_rediscovery", "table8_sanity", "table9_ablation"]: show(n)')
md("""E8: `spearman_*` tốt phải gần 0 (Adebayo 2018) và CRT trên mô hình xáo nhãn **không** được bác bỏ.""")

md("""## 10. Ghép với kết quả CPU, vẽ 5 hình
Upload `results_cpu/` (hoặc `results_cpu.zip`) từ máy của bạn làm Dataset Kaggle; cell này tự gộp. Không có thì chỉ dùng kết quả GPU.""")
code("""import zipfile
CPU_DIR = os.path.join(WORK, "results_cpu")
for z in glob.glob("/kaggle/input/*/results_cpu*.zip"):
    shutil.unpack_archive(z, CPU_DIR); print("giải nén", z)
for d in glob.glob("/kaggle/input/*/results_cpu"):
    shutil.copytree(d, CPU_DIR, dirs_exist_ok=True)
MERGED = os.path.join(WORK, "results_merged")
srcs = [d for d in (CPU_DIR, OUT) if os.path.isdir(d)]
r = subprocess.run([sys.executable, f"{PKG}/merge_results.py", MERGED] + srcs, capture_output=True, text=True, cwd=PKG); print(r.stdout[-1500:], r.stderr[-800:])
r = subprocess.run([sys.executable, f"{PKG}/run.py", "--preset", PRESET, "--stage", "report", "--out", MERGED, "--cache", CACHE], capture_output=True, text=True, cwd=PKG)
print(r.stdout[-1500:], r.stderr[-800:])
for f in sorted(glob.glob(f"{MERGED}/figs/Fig*.png")): display(Image(f))
show_m = lambda n: display(Markdown(open(f"{MERGED}/tables/{n}.md").read())) if os.path.exists(f"{MERGED}/tables/{n}.md") else None
show_m("table15_failure")""")

md("## 11. Nén để tải về")
code("""shutil.make_archive(os.path.join(WORK, "results_gpu"), "zip", OUT)
shutil.make_archive(os.path.join(WORK, "results_merged"), "zip", MERGED) if os.path.isdir(MERGED) else None
if os.path.isdir(CACHE): shutil.make_archive(os.path.join(WORK, "cache_gpu"), "zip", CACHE)
print("Đã nén: results_gpu.zip (kết quả GPU), results_merged.zip (CPU+GPU + hình), cache_gpu.zip (mô hình đã huấn luyện)")""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "TrajConcept_Kaggle.ipynb"); print("ok", len(C), "cells")
