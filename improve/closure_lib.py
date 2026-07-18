"""诊断闭环 Python 复现沙盒。

用途：在不依赖 MATLAB / 原始数据的前提下，从 closure_sequences.mat 的锁定残差 R
复现"约束反演 + 决策层"，作为算法改进的可验证试验台。严格遵守防泄漏：
阈值/持续/等级/拒识参数只在 calibration 段拟合，测试真值只用于算指标。

基线与 MATLAB 主管线的差异仅来自 NNLS 求解器实现（scipy vs lsqnonneg），
用于对比"基线决策层 vs 改进决策层"时，二者共用同一反演，结论不受该差异影响。
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.optimize import nnls
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT = os.path.join(ROOT, "matlab", "outputs", "t4_diagnostic_closure", "closure_sequences.mat")
HCSV = os.path.join(ROOT, "reference", "fixed_results", "influence_matrix", "H_9cols.csv")
SPAN = np.array([0.018668, 0.223446, 0.121209, 0.025540, 0.070658,
                 0.036194, 0.032908, 0.118767, 0.046187])
FAMILY_OF = {0: "HPT", 1: "Fan", 2: "Fan", 3: "HPC", 4: "HPC",
             5: "LPT", 6: "LPT", 7: "LPC", 8: "LPC"}
FAMS = ["HPT", "Fan", "HPC", "LPT", "LPC"]


def load_Hn():
    H = pd.read_csv(HCSV, index_col=0).values
    return H * SPAN


# ---------------------------------------------------------------- 约束反演
def build_M(Hn, T):
    m, n = Hn.shape
    M = np.zeros((T * m, T * n))
    for t in range(T):
        for i in range(t + 1):
            M[t*m:(t+1)*m, i*n:(i+1)*n] = -Hn
    return M


def build_cum(T, n):
    C = np.zeros((T * n, T * n))
    for t in range(T):
        for i in range(t + 1):
            C[t*n:(t+1)*n, i*n:(i+1)*n] = np.eye(n)
    return C


def solve_D(Hn, R, lam):
    """整段非正—单调约束反演：theta=-cumsum(d), d>=0。"""
    T = R.shape[0]; n = Hn.shape[1]
    Maug = np.vstack([build_M(Hn, T), np.sqrt(lam) * build_cum(T, n)])
    yaug = np.concatenate([R.reshape(-1), np.zeros(T * n)])
    d, _ = nnls(Maug, yaug, maxiter=50 * T * n)
    return -np.cumsum(d.reshape(T, n), axis=0)


# ---------------------------------------------------------------- 数据读取
def _str(ds):
    return ''.join(chr(int(c)) for c in np.array(ds).flatten())


def load_sequences(lam=0.01, cache=True):
    """读取 cal/id/ood 三段，反演得到 theta_hat，返回 dict of list。"""
    cache_path = os.path.join(os.path.dirname(__file__), f"_seq_cache_lam{lam}.npz")
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return z["data"].item()
    Hn = load_Hn()
    out = {}
    with h5py.File(MAT, "r") as f:
        for key in ["calRaw", "idRaw", "oodRaw"]:
            g = f[key]; seqs = []
            for i in range(g.shape[0]):
                st = f[g[i, 0]]
                R = np.array(st["R"]).T
                th_true = np.array(st["theta_true"]).T
                hs = np.array(st["hs"]).flatten().astype(bool)
                cycles = np.array(st["cycles"]).flatten().astype(int)
                fault_idx = np.array(st["fault_idx"]).flatten().astype(int) - 1  # 0-based
                subset = _str(st["subset"]); unit = int(np.array(st["unit"]).flatten()[0])
                th = solve_D(Hn, R, lam)
                seqs.append(dict(R=R, theta_true=th_true, theta_hat=th, hs=hs,
                                 cycles=cycles, fault_idx=fault_idx,
                                 subset=subset, unit=unit,
                                 regime=key.replace("Raw", "")))
            out[key] = seqs
    if cache:
        np.savez(cache_path, data=np.array(out, dtype=object))
    return out


# ---------------------------------------------------------------- 决策层（基线）
def threshold_labels(sev, tau, K):
    """严重度连续超阈值 K 周期后锁存。"""
    above = sev > tau
    lab = np.zeros_like(above, dtype=bool)
    for j in range(above.shape[1]):
        cnt = 0; latched = False
        for t in range(above.shape[0]):
            cnt = cnt + 1 if above[t, j] else 0
            if cnt >= K:
                latched = True
            lab[t, j] = latched
    return lab


def fit_baseline(cal, alpha=0.01, Kgrid=(1, 2, 3, 5), max_far=0.05,
                 stage_K=3, eps=1e-4, seed=316259):
    sev_health = []
    truth_sev = []
    for s in cal:
        sev = np.maximum(0, -s["theta_hat"])
        sev_health.append(sev[s["hs"]])
        act = np.maximum(0, -s["theta_true"])
        truth_sev.append(act[act > eps])
    sev_health = np.vstack(sev_health)
    tau = np.percentile(sev_health, 100 * (1 - alpha), axis=0)
    # persistence
    best = _select_K(cal, tau, Kgrid, max_far)
    # stage GMM on calibration true active severity
    x = np.concatenate(truth_sev); x = x[np.isfinite(x) & (x > 0)]
    gm = GaussianMixture(stage_K, n_init=10, random_state=seed).fit(x.reshape(-1, 1))
    order = np.argsort(gm.means_.flatten())
    stage = dict(mu=gm.means_.flatten()[order], sigma=np.sqrt(gm.covariances_.flatten())[order],
                 w=gm.weights_[order], K=stage_K)
    return dict(tau=tau, K=best, stage=stage, eps=eps, alpha=alpha)


def _select_K(cal, tau, Kgrid, max_far):
    rows = []
    for K in Kgrid:
        fars, f1s = [], []
        for s in cal:
            lab = threshold_labels(np.maximum(0, -s["theta_hat"]), tau, K)
            truth = np.zeros_like(lab);
            truth[~s["hs"][:, None] & np.isin(np.arange(9), s["fault_idx"])[None, :]] = True
            if s["hs"].any():
                fars.append(np.mean(lab[s["hs"]].any(axis=1)))
            f1s.append(macro_f1(lab, truth.astype(bool)))
        rows.append((K, np.mean(fars), np.mean(f1s)))
    rows = pd.DataFrame(rows, columns=["K", "far", "f1"])
    feas = rows[rows.far <= max_far]
    pool = feas if len(feas) else rows[rows.far == rows.far.min()]
    best = pool[pool.f1 == pool.f1.max()]["K"].max()
    return int(best)


def stage_predict(x, stage):
    x = np.asarray(x).reshape(-1, 1)
    p = stage["w"] / np.maximum(stage["sigma"], 1e-8) * \
        np.exp(-0.5 * ((x - stage["mu"]) / np.maximum(stage["sigma"], 1e-8)) ** 2)
    return np.argmax(p, axis=1)


def predict_baseline(s, model):
    sev = np.maximum(0, -s["theta_hat"])
    lab = threshold_labels(sev, model["tau"], model["K"])
    comp_stage = np.zeros_like(sev)
    for j in range(sev.shape[1]):
        act = lab[:, j]
        if act.any():
            cs = np.zeros(sev.shape[0])
            cs[act] = stage_predict(sev[act, j], model["stage"]) + 1
            comp_stage[:, j] = np.maximum.accumulate(cs)
    return dict(sev=sev, label=lab, fault=lab.any(axis=1),
                stage=comp_stage.max(axis=1).astype(int))


def truth_of(s, model):
    lab = np.zeros((len(s["cycles"]), 9), dtype=bool)
    lab[~s["hs"][:, None] & np.isin(np.arange(9), s["fault_idx"])[None, :]] = True
    sev = np.maximum(0, -s["theta_true"])
    comp_stage = np.zeros_like(sev)
    for j in range(9):
        act = lab[:, j]
        if act.any():
            cs = np.zeros(sev.shape[0])
            cs[act] = stage_predict(sev[act, j], model["stage"]) + 1
            comp_stage[:, j] = np.maximum.accumulate(cs)
    return dict(label=lab, fault=~s["hs"], stage=comp_stage.max(axis=1).astype(int))


# ---------------------------------------------------------------- 指标
def binary(pred, truth):
    pred = np.asarray(pred, bool); truth = np.asarray(truth, bool)
    tp = np.sum(pred & truth); tn = np.sum(~pred & ~truth)
    fp = np.sum(pred & ~truth); fn = np.sum(~pred & truth)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(FAR=fp / max(fp + tn, 1), DR=rec, precision=prec, F1=f1)


def macro_f1(pred, truth):
    inc = pred.any(0) | truth.any(0)
    if not inc.any():
        return 1.0
    vals = []
    for j in np.where(inc)[0]:
        vals.append(binary(pred[:, j], truth[:, j])["F1"])
    return float(np.mean(vals))


def weighted_kappa(pred, truth):
    pred = np.asarray(pred); truth = np.asarray(truth)
    K = int(max(pred.max(), truth.max())) + 1
    C = np.zeros((K, K))
    for p, t in zip(pred, truth):
        C[t, p] += 1
    n = C.sum()
    if n == 0:
        return np.nan
    W = ((np.arange(K)[:, None] - np.arange(K)[None, :]) / max(K - 1, 1)) ** 2
    O = C / n; E = C.sum(1)[:, None] * C.sum(0)[None, :] / n ** 2
    return 1 - (W * O).sum() / max((W * E).sum(), 1e-12)


def family_labels(label9):
    """9 参数多标签 -> 5 部件族多标签（逻辑或）。"""
    out = np.zeros((label9.shape[0], 5), dtype=bool)
    for j in range(9):
        fi = FAMS.index(FAMILY_OF[j])
        out[:, fi] |= label9[:, j]
    return out
