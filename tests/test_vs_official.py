"""Doi chieu voi repo chinh thuc cua thay (DAIR-Group/XAI4Traj). Lay repo:  git clone --depth 1 https://github.com/DAIR-Group/XAI4Traj <dir>
roi dat bien moi truong XAI4TRAJ_REPO=<dir>. Khong co repo -> cac test 'official' duoc bo qua (SKIP), cac test con lai van chay.
Luu y: thu vien `rdp` cua repo goi np.cross tren vector 2-D nen loi voi numpy>=2; test dung ham khoang cach 2-D tuong duong."""
import os, sys, importlib.util
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trajconcept.config import Config
from trajconcept.explain import apply_op
from trajconcept.segment import _mdl_repo, _rdp, _bp_to_mask
from trajconcept.xai_metrics import precision_X, fidelity

REPO = os.environ.get("XAI4TRAJ_REPO", "/tmp/ref/repo")
SRC = os.path.join(REPO, "traj_xai", "src")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _have():
    return os.path.exists(os.path.join(SRC, "evaluation.py"))


def test_perturbations_match_official():
    if not _have(): print("SKIP official perturbations"); return
    P = _load(os.path.join(SRC, "perturbation", "perturbation.py"), "po").Perturbation()
    rng0 = np.random.RandomState(1); X = np.cumsum(rng0.randn(20, 2), 0)
    mask = np.zeros(20, bool); mask[5:12] = True
    seg = [tuple(p) for p in X[mask]]
    # nhieu: repo = normal(0,std)*scale, thu tu (x, y) tung diem
    np.random.seed(7); ref = np.array(P._gaussian_perturbation(seg, 0, 3.0, 1.5))
    cfg = Config(noise_sigma=3.0 * 1.5)
    mine = apply_op("noise", X, mask, cfg, np.random.RandomState(7))[mask]
    assert np.allclose(ref, mine), np.abs(ref - mine).max()
    ref = np.array(P._scaling_perturbation(seg, 1.2)); mine = apply_op("scale", X, mask, Config(scale_factor=1.2), None)[mask]
    assert np.allclose(ref, mine)
    ref = np.array(P._rotation_perturbation(seg, np.pi / 18)); mine = apply_op("rotate", X, mask, Config(rot_angle=np.pi / 18), None)[mask]
    assert np.allclose(ref, mine)
    # diem ngoai doan khong doi
    assert np.array_equal(apply_op("rotate", X, mask, Config(), None)[~mask], X[~mask])


def test_raw_frame_ops_match_official_on_raw_coordinates():
    """Profile 'advisor': ap phep nhieu cua repo len TOA DO THO roi quy ve khong gian chuan hoa phai trung khop."""
    if not _have(): print("SKIP official raw-frame"); return
    P = _load(os.path.join(SRC, "perturbation", "perturbation.py"), "po2").Perturbation()
    rng0 = np.random.RandomState(3); Xraw = np.cumsum(rng0.randn(24, 2) * 7, 0) + np.array([55.0, 31.0])
    origin = Xraw[0].copy(); scale = 6.3
    X = ((Xraw - origin) / scale).astype(np.float64)
    mask = np.zeros(24, bool); mask[8:15] = True
    cfg = Config().ops_for(scale, "advisor", origin)
    seg = [tuple(p) for p in Xraw[mask]]
    for op, ref_fn in (("rotate", lambda: P._rotation_perturbation(seg, np.pi / 18)), ("scale", lambda: P._scaling_perturbation(seg, 1.2))):
        ref = (np.array(ref_fn()) - origin) / scale
        assert np.allclose(apply_op(op, X, mask, cfg, None)[mask], ref, atol=1e-9), op
    np.random.seed(11); ref = (np.array(P._gaussian_perturbation(seg, 0, 3.0, 1.5)) - origin) / scale
    assert np.allclose(apply_op("noise", X, mask, cfg, np.random.RandomState(11))[mask], ref, atol=1e-9)


def test_segmentation_matches_official():
    if not _have(): print("SKIP official segmentation"); return
    S = _load(os.path.join(SRC, "segmentation", "segmentation.py"), "so") if False else None
    # segmentation.py import `rdp` (ban cu loi voi numpy>=2): chi nap neu import duoc
    try:
        S = _load(os.path.join(SRC, "segmentation", "segmentation.py"), "so").Segmentation()
    except Exception as e:
        print("SKIP official segmentation (", e, ")"); return
    for k in range(200):
        rng = np.random.RandomState(k); L = rng.randint(8, 60); P = np.cumsum(rng.randn(L, 2), 0); eps = float(rng.choice([0.5, 0.8, 2.0, 5.0]))
        ref = [len(x) for x in S._mdl_segmentation([tuple(p) for p in P], epsilon=eps)]
        bp = _mdl_repo(P, eps)
        mine = list(_bp_to_mask(bp, L).sum(1)) if len(bp) > 1 else [L]
        assert ref == mine, (k, ref, mine)


def test_rdp_matches_rdp_library():
    try:
        from rdp import rdp
    except Exception:
        print("SKIP rdp lib"); return
    def pld(point, start, end):
        if np.all(np.equal(start, end)): return np.linalg.norm(point - start)
        d = end - start; v = start - point
        return abs(d[0] * v[1] - d[1] * v[0]) / np.linalg.norm(d)
    for k in range(200):
        rng = np.random.RandomState(k); L = rng.randint(8, 80); P = np.cumsum(rng.randn(L, 2), 0); eps = float(rng.choice([0.1, 0.5, 1.0, 3.0]))
        ref = {tuple(p) for p in rdp(P, epsilon=eps, dist=pld)}
        assert np.array_equal(np.array([i for i, p in enumerate(P) if tuple(p) in ref]), _rdp(P, eps))


def test_precision_eq8_equals_repo_ap_at_k_when_used_correctly():
    if not _have(): print("SKIP official ap_at_k"); return
    ev = _load(os.path.join(SRC, "evaluation.py"), "ev")
    rng = np.random.RandomState(0)
    for _ in range(300):
        T = rng.randint(2, 12); f = (rng.rand(T) < rng.rand()).astype(float)
        if f.sum() < 1: continue
        y = ["B" if v else "A" for v in f]
        assert abs(ev.ap_at_k(y, "A", T) - precision_X(f)) < 1e-12
        assert ev.ap_at_k(y, ["A"], T) == 1.0          # LOI CUA REPO: experiment.py truyen relevant_class la LIST (get_Y()) -> luon 1.0


def test_fidelity_none_theta_is_not_explainable():
    fl = [np.array([1, 0, 1], bool), np.array([1, 1, 0], bool)]
    th = [np.array([3., 1., 2.]), None]
    r = fidelity(fl, th)
    assert abs(r["R"] - 0.5) < 1e-12 and abs(r["P"] - 1.0) < 1e-12


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("OK", k)
