"""Generate the verification notebook:  python make_verify_notebook.py  ->  TrajConcept_Verify.ipynb"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# TrajConcept – KIỂM CHỨNG phương pháp (1 cell mạnh, Kaggle T4)

Mục tiêu: trả lời **phương pháp có thực sự hiệu quả không** bằng bảng PASS / FAIL / INCONCLUSIVE cho 6 tuyên bố (C1–C6, tiêu chí đặt trước trong `trajconcept/claims.py`).
Settings → Accelerator = **GPU T4 x1**, Internet = **On**. Cell chạy chính tự tiếp tục: hết giờ thì chạy lại.

| Tuyên bố | Nội dung |
|---|---|
| C1 | Không báo sai: tỉ lệ báo sai của CRT có kiểm soát ở thế giới rỗng (b=0), ≥ 27 lần chạy |
| C2 | Có sức mạnh: phát hiện được khi b ≥ 4, và gap tăng theo b |
| C3 | Tìm đúng yếu tố cài sẵn, hơn baseline ngẫu nhiên và PCA |
| C4 | Sanity Adebayo: giải thích đổi khi xáo nhãn / khởi tạo lại mạng |
| C5 | Trên dữ liệu thật: concept ngoài K qua mọi bộ lọc, lặp lại ở ≥ 2/3 seed |
| C6 | Fidelity (thước đo XAI4Traj): tốt hơn mặt nạ ngẫu nhiên, khối lượng đúng vị trí ≥ 1.2 |""")

md("## 1. Lấy mã từ GitHub")
code("""import os, sys, glob, subprocess, shutil
GIT_URL = "https://github.com/Hana-ai-18/exaitraj.git"
BRANCH = ""          # để trống = nhánh mặc định của repo
WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath("./runs")
os.makedirs(WORK, exist_ok=True); os.chdir(WORK)
REPO = os.path.join(WORK, "repo")

# Repo private? Tạo token GitHub (quyền đọc nội dung repo) và lưu vào Kaggle: Add-ons -> Secrets -> tên GITHUB_TOKEN, bật cho notebook này.
url = GIT_URL
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    if tok: url = GIT_URL.replace("https://", f"https://x-access-token:{tok}@")
except Exception:
    pass

if os.path.exists(os.path.join(REPO, ".git")):
    print(subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], capture_output=True, text=True).stdout)
else:
    cmd = ["git", "clone", "--depth", "1"] + (["-b", BRANCH] if BRANCH else []) + [url, REPO]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if r.returncode != 0:
        msg = r.stderr.replace(url, GIT_URL)
        raise RuntimeError("git clone lỗi. Nếu repo là private hãy thêm Kaggle Secret GITHUB_TOKEN (xem ghi chú trên cell). Chi tiết: " + msg)

# Tìm thư mục chứa run.py (ở gốc repo hoặc trong một thư mục con, ví dụ trajconcept_pkg/)
hits = sorted(glob.glob(os.path.join(REPO, "run.py")) + glob.glob(os.path.join(REPO, "*", "run.py")), key=len)
assert hits, "Không thấy run.py trong repo. Repo cần chứa run.py, trajconcept/, tests/, merge_results.py, requirements.txt (ở gốc hoặc trong 1 thư mục con)."
PKG = os.path.dirname(hits[0])
for need in ("trajconcept", "tests", "merge_results.py"):
    assert os.path.exists(os.path.join(PKG, need)), f"Thiếu {need} trong {PKG}"
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "requests", "scipy", "scikit-learn", "pandas", "matplotlib"], check=False)
os.environ["TRAJ_DATA_DIR"] = os.path.join(WORK, "data")
print("PKG =", PKG, "| WORK =", WORK)""")


md("## 2. Kiểm tra GPU và unit test")
code("""import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU)")
r = subprocess.run([sys.executable, "-u", "-m", "pytest", "-q", os.path.join(PKG, "tests", "test_core.py")], capture_output=True, text=True, cwd=PKG); print(r.stdout[-600:]); assert r.returncode == 0, "test lỗi"
OUT, CACHE = os.path.join(WORK, "verify_out"), os.path.join(WORK, "verify_cache")
for pat, dst in (("/kaggle/input/*/verify_out*.zip", OUT), ("/kaggle/input/*/verify_cache*.zip", CACHE)):
    for z in glob.glob(pat):
        shutil.unpack_archive(z, dst); print("đã khôi phục", z, "->", dst)""")

md("## 3. Chạy kiểm chứng (tiếp tục được; ~4–6 giờ trên T4)\nĐổi `SMOKE = True` để thử nhanh trên bất kỳ máy nào (không có giá trị kết luận).")
code("""SMOKE = False
BUDGET_H = 8.0
cmd = [sys.executable, "-u", os.path.join(PKG, "verify.py"), "--out", OUT, "--cache", CACHE, "--budget-hours", str(BUDGET_H)] + (["--smoke"] if SMOKE else [])
print(">>", " ".join(cmd)); subprocess.run(cmd, cwd=PKG)""")

md("## 4. Bảng kết luận (tính lại bất cứ lúc nào từ CSV đã có)")
code("""subprocess.run([sys.executable, "-u", os.path.join(PKG, "verify.py"), "--claims-only", "--out", OUT], cwd=PKG)
import pandas as pd
pd.set_option("display.max_colwidth", 90, "display.width", 250)
df = pd.read_csv(os.path.join(OUT, "claims.csv")); display(df)
print(open(os.path.join(OUT, "claims.md")).read().split("\n")[-2])""")

md("## 5. Nén kết quả để tải về / dùng lại")
code("""for name, d in (("verify_out", OUT), ("verify_cache", CACHE)):
    if os.path.isdir(d): shutil.make_archive(os.path.join(WORK, name), "zip", d); print("đã nén", name + ".zip")""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "TrajConcept_Verify.ipynb"); print("wrote TrajConcept_Verify.ipynb", len(C), "cells")
