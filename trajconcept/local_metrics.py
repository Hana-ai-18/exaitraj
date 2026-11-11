"""Thuoc do cuc bo thuan (khong phu thuoc mo hinh) - tach ra de test doc lap.
 relevant_mass_ratio: phan khoi luong phi nam trong doan chua vi tri that, chia cho ti le doan do (1 = ngau nhien).
 iou_topk: IoU giua top-k doan theo phi (k = so doan chua t*) va tap doan chua t*.
 leak_overlap: sum_j phi^_j kappa^_j * T voi phi^, kappa^ chuan hoa tong = 1 (1 = muc ngau nhien, >1 = trung voi tri thuc cu)."""
import numpy as np


def relevant_set(masks, tstar):
    return np.where(np.asarray(masks)[:, tstar])[0]


def relevant_mass_ratio(phi, masks, tstar):
    phi = np.clip(np.asarray(phi, float), 0, None); jj = relevant_set(masks, tstar)
    if len(jj) == 0 or phi.sum() <= 0:
        return None
    return float(phi[jj].sum() / phi.sum() / (len(jj) / len(phi)))


def iou_topk(phi, masks, tstar):
    phi = np.clip(np.asarray(phi, float), 0, None); jj = relevant_set(masks, tstar)
    if len(jj) == 0 or phi.sum() <= 0:
        return None
    top = set(np.argsort(-phi)[:len(jj)].tolist()); rel = set(jj.tolist())
    return len(top & rel) / len(top | rel)


def leak_overlap(phi, kappa):
    phi = np.clip(np.asarray(phi, float), 0, None); kappa = np.asarray(kappa, float)
    if kappa.sum() <= 0 or phi.sum() <= 0:
        return None
    return float((phi / phi.sum() * kappa / kappa.sum()).sum() * len(phi))
