import os, sys, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajconcept.store import Store
from trajconcept import claims


def test_wilson_and_boot():
    lo, hi = claims.wilson(0, 30); assert lo == 0 or lo < 1e-9; assert 0.09 < hi < 0.13
    m, lo, hi = claims.boot_ci(np.r_[np.ones(30), np.zeros(5)]); assert lo <= m <= hi


def test_claims_pass_and_fail_on_synthetic():
    with tempfile.TemporaryDirectory() as d:
        S = Store(d); rng = np.random.RandomState(0)
        # tinh huong tot: 30 lan null, khong bao sai; power cao khi b>=4; hoi phuc tot
        S.add("v_null", [dict(dataset="x", seed=i, bb="gru", b=0.0, detect_ctrl=False, detect_raw=bool(i % 3 == 0)) for i in range(30)])
        rows = []
        for fam in ("freq", "where"):
            for b in (2.0, 4.0, 8.0):
                for s in range(4):
                    rows.append(dict(dataset="x", family=fam, b=b, seed=s, bb="gru", detect_crt=True, detect_gate=True, delta_nll=0.02 * b,
                                     rec_all=0.7, rec_rand=0.2, rec_pca=0.3))
        S.add("v_worlds", rows)
        df = claims.evaluate(S); v = dict(zip(df.claim, df.verdict))
        assert v["C1 sai duong CRT co kiem soat [gru]"] == "PASS"
        assert all(v[k] == "PASS" for k in v if k.startswith("C2") or k.startswith("C3"))
    with tempfile.TemporaryDirectory() as d:
        S = Store(d)       # tinh huong xau: bao sai nhieu
        S.add("v_null", [dict(dataset="x", seed=i, bb="gru", b=0.0, detect_ctrl=bool(i % 2 == 0), detect_raw=True) for i in range(30)])
        df = claims.evaluate(S); assert dict(zip(df.claim, df.verdict))["C1 sai duong CRT co kiem soat [gru]"] == "FAIL"
        assert all(x == "INCONCLUSIVE" for x in df[df.claim.str.startswith("C5")].verdict)   # khong du lieu -> khong bao gio PASS
