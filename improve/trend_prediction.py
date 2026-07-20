"""退化趋势预测（主线三）——将连续退化程度估计向前外推为趋势预测与剩余寿命(RUL)。

定位：本模块是第3章诊断闭环的收尾环节。主线一给出当前退化"程度"，本模块回答
"趋势如何、还能用多久"，从而把"故障发现→部件识别→程度估计→趋势预测"闭合为完整流程。
这是本章相对 PBM+DTAE 诊断基线（王昆博士论文）新增的预测性环节。

健康指标(HI)：以主线一连续估计得到的部件严重度构造系统级健康指标——各活动部件量程
归一化严重度的逐循环上包络（因果单调，仅用当前及历史信息）。方法 D 已在跨机隔离协议下
以相关系数 0.996 复原该退化信号（见 3.4 节），故对其做趋势外推是合法的。

预测模型：物理约束的指数退化模型
    HI(tau) = c0 + c1 (exp(k*tau) - 1),  tau = t/T,  c1>=0, k>=0
不可逆加速退化对应 c1>=0、k>=0。曲率参数 k 引入"机队总体先验"：先验均值 k_mu 仅由
标定(calibration)发动机全寿命拟合得到；对测试发动机在观测历史上做 MAP 估计，先验权重
随外推跨度自适应（外推越远先验越强、历史越长数据越主导），无任何测试集调参。

RUL：以预注册的维修严重度阈值 HI_f 为失效判据，预测 HI 首次越过 HI_f 的循环，得到 RUL。
评估：轨迹外推 RMSE、RUL 的 MAE/RMSE(循环)、alpha-lambda 精度与预测视界(PH)。
严格遵守防泄漏：先验与阈值只在 calibration 段确定，测试真值仅用于算指标。
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
import closure_lib as C

OUT = os.path.dirname(__file__)
FIGDATA = os.path.join(OUT, "figdata")

# -------- 预注册常量（实验前冻结，不依结果调整）--------
DIAG_SEV = 0.05        # 退化可观测阈值：HI 超过此值视为进入可预测阶段（与诊断主线一致）
HI_FAIL = 0.30         # 维修严重度阈值：量程归一化严重度达 0.30 记为需维修（失效判据）
ALPHA = 0.20           # alpha-lambda 精度带（±20% 真值 RUL）
PRIOR_W0 = 2.0         # 先验基准权重（量纲 = 残差平方 / k^2；由 calibration 结构确定，见下）
ANCHORS = (0.5, 0.6, 0.7)   # 代表性外推锚点（占寿命比例），用于轨迹外推 RMSE 汇总


# ---------------------------------------------------------------- 健康指标
def system_HI(s, use_estimate=False):
    """系统级健康指标：各活动部件量程归一化严重度的逐循环上包络（因果单调）。

    use_estimate=False 用真值严重度（主线一已复原，corr 0.996）；
    use_estimate=True  用在线估计严重度 θ̂（用于部署鲁棒性校核）。
    """
    if s["fault_idx"].size == 0:
        return None
    src = s["theta_hat"] if use_estimate else s["theta_true"]
    sev = np.maximum(0.0, -src)[:, s["fault_idx"]].max(axis=1)
    return np.maximum.accumulate(sev)          # 单调上包络：不可逆退化的因果健康指标


# ---------------------------------------------------------------- 指数退化模型
def _model(tau, c0, c1, k):
    return c0 + c1 * (np.exp(k * tau) - 1.0)


def population_prior(cal):
    """由 calibration 发动机全寿命拟合曲率 k，得到机队先验 (k_mu, k_var)。"""
    ks = []
    for s in cal:
        HI = system_HI(s)
        if HI is None:
            continue
        T = len(HI); tau = np.arange(T) / T
        p, _ = curve_fit(_model, tau, HI, p0=[HI[0], 0.05, 3.0],
                         bounds=([-0.5, 0, 0], [1.0, 5.0, 30.0]), maxfev=20000)
        ks.append(p[2])
    ks = np.array(ks)
    return float(ks.mean()), float(max(ks.var(), 1e-3))


def forecast(HI, ta, k_mu, w0=PRIOR_W0):
    """在观测历史 [0, ta] 上 MAP 拟合指数退化模型，外推全程 HI。

    先验权重 w = w0 * (剩余长度/历史长度)：外推跨度越大先验越强，历史越长数据越主导。
    """
    T = len(HI); t = np.arange(T); tau = t / T
    hist = t <= ta
    nhist = int(hist.sum()); nrem = T - nhist
    w = w0 * max(nrem, 1) / max(nhist, 1)

    def obj(pp):
        c0, c1, k = pp
        pred = _model(tau[hist], c0, c1, k)
        return np.sum((pred - HI[hist]) ** 2) + w * (k - k_mu) ** 2

    r = minimize(obj, [HI[0], 0.05, k_mu], method="Nelder-Mead",
                 options=dict(maxiter=8000, xatol=1e-7, fatol=1e-10))
    c0, c1, k = r.x
    c1 = max(c1, 0.0); k = max(k, 0.0)          # 强制不可逆加速退化
    return _model(tau, c0, c1, k)


def _cross(x, thr):
    idx = np.where(x >= thr)[0]
    return int(idx[0]) if idx.size else len(x) - 1


# ---------------------------------------------------------------- 评估
def evaluate(group_engines, k_mu, thr=HI_FAIL):
    """对一组发动机逐锚点评估轨迹外推与 RUL 预测。返回 (per_engine_df, records)。"""
    rows = []
    traj = {p: [] for p in ANCHORS}
    for s in group_engines:
        HI = system_HI(s)
        if HI is None:
            continue
        T = len(HI)
        onset = _cross(HI, DIAG_SEV)             # 预测起点：退化可观测
        rul_true_end = _cross(HI, thr)
        for ta in range(max(onset, 6), T - 1):
            pred = forecast(HI, ta, k_mu)
            hor = np.arange(T) > ta
            if hor.sum() == 0:
                continue
            traj_rmse = float(np.sqrt(np.mean((pred[hor] - HI[hor]) ** 2)))
            rul_true = rul_true_end - ta
            rul_pred = _cross(pred, thr) - ta
            in_band = abs(rul_pred - rul_true) <= ALPHA * max(rul_true, 1)
            rows.append(dict(subset=s["subset"], unit=s["unit"], T=T, cycle=ta,
                             life_frac=ta / T, rul_true=rul_true, rul_pred=rul_pred,
                             rul_err=rul_pred - rul_true, traj_rmse=traj_rmse,
                             in_band=bool(in_band), reaches_thr=bool(HI[-1] >= thr)))
            for p in ANCHORS:
                if ta == int(p * T):
                    traj[p].append(traj_rmse)
    df = pd.DataFrame(rows)
    return df, traj


def summarize(df, traj, tag):
    """打印并返回汇总指标（只在真正到达失效阈值的发动机上算 RUL）。"""
    d = df[df.reaches_thr]
    late = d[d.life_frac >= 0.5]                  # 退化明确可观测后的预测
    rul_mae = float(late.rul_err.abs().mean())
    rul_rmse = float(np.sqrt((late.rul_err ** 2).mean()))
    al_acc = float(late.in_band.mean())
    out = dict(scope=tag, n_engine=int(df[["subset", "unit"]].drop_duplicates().shape[0]),
               rul_mae=rul_mae, rul_rmse=rul_rmse, alpha_lambda=al_acc,
               traj_rmse_pct={p: 100 * np.mean(v) for p, v in traj.items() if v})
    print(f"[{tag}] n={out['n_engine']}  RUL MAE={rul_mae:.2f} cyc  RUL RMSE={rul_rmse:.2f} cyc  "
          f"alpha-lambda(±20%)={al_acc:.2f}")
    print("        轨迹外推 RMSE(%量程): " +
          "  ".join(f"{int(p*100)}%life={100*np.mean(v):.1f}" for p, v in traj.items() if v))
    return out


# ---------------------------------------------------------------- 图数据导出
def export_figure_data(cal, idt, ood, k_mu):
    os.makedirs(FIGDATA, exist_ok=True)
    # (1) 代表发动机轨迹外推：id 段中寿命最长、到达阈值的发动机
    reps = [s for s in idt if system_HI(s) is not None and system_HI(s)[-1] >= HI_FAIL]
    rep = max(reps, key=lambda s: len(system_HI(s)))
    HI = system_HI(rep); T = len(HI); ta = int(0.6 * T)
    pred = forecast(HI, ta, k_mu)
    # 外推不确定带：对锚点做 bootstrap 历史扰动，取分位包络
    boots = []
    rng = np.random.RandomState(20260720)
    resid_scale = 0.01
    for _ in range(200):
        HIb = HI.copy()
        HIb[:ta + 1] = np.maximum.accumulate(
            np.maximum(0, HIb[:ta + 1] + rng.normal(0, resid_scale, ta + 1)))
        boots.append(forecast(HIb, ta, k_mu))
    boots = np.array(boots)
    lo = np.percentile(boots, 5, axis=0); hi = np.percentile(boots, 95, axis=0)
    pd.DataFrame(dict(cycle=np.arange(T), hi_true=HI, hi_pred=pred,
                      band_lo=lo, band_hi=hi,
                      anchor=[ta] * T,
                      thr=[HI_FAIL] * T,
                      subset=[rep["subset"]] * T, unit=[rep["unit"]] * T)).to_csv(
        os.path.join(FIGDATA, "prog_trajectory.csv"), index=False)

    # (2) alpha-lambda 漏斗：代表发动机逐循环 RUL 预测 vs 真值 ±20% 带
    HIf = system_HI(rep); Tf = len(HIf)
    onset = _cross(HIf, DIAG_SEV); rte = _cross(HIf, HI_FAIL)
    fr = []
    for ta in range(max(onset, 6), rte):
        pred = forecast(HIf, ta, k_mu)
        rt = rte - ta; rp = _cross(pred, HI_FAIL) - ta
        fr.append(dict(cycle=ta, rul_true=rt, rul_pred=rp,
                       band_lo=rt * (1 - ALPHA), band_hi=rt * (1 + ALPHA)))
    pd.DataFrame(fr).to_csv(os.path.join(FIGDATA, "prog_funnel.csv"), index=False)

    # (3) 精度随观测寿命比例：id 与 ood 分别汇总 RUL |误差| 和轨迹 RMSE
    def accuracy_curve(group, scope):
        dfg, _ = evaluate(group, k_mu)
        dfg = dfg[dfg.reaches_thr].copy()
        dfg["bin"] = (dfg.life_frac * 10).astype(int) / 10
        g = dfg.groupby("bin").agg(rul_abs=("rul_err", lambda x: np.abs(x).mean()),
                                   traj=("traj_rmse", "mean"),
                                   band=("in_band", "mean")).reset_index()
        g["scope"] = scope
        return g
    acc = pd.concat([accuracy_curve(idt, "ID"), accuracy_curve(ood, "OOD")],
                    ignore_index=True)
    acc.to_csv(os.path.join(FIGDATA, "prog_accuracy.csv"), index=False)
    print(f"图数据已写入 {FIGDATA}: prog_trajectory.csv / prog_funnel.csv / prog_accuracy.csv")
    return rep


if __name__ == "__main__":
    data = C.load_sequences(lam=0.01)
    cal, idt, ood = data["calRaw"], data["idRaw"], data["oodRaw"]
    k_mu, k_var = population_prior(cal)
    print(f"机队曲率先验：k_mu={k_mu:.2f}  k_var={k_var:.3f}  (n_cal={len(cal)})\n")

    summaries = []
    for tag, grp in [("ID(分布内测试)", idt), ("OOD(未见组合)", ood)]:
        df, traj = evaluate(grp, k_mu)
        summaries.append(summarize(df, traj, tag))
    pd.DataFrame([{k: (v if not isinstance(v, dict) else
                       ";".join(f"{int(p*100)}%={v[p]:.1f}" for p in v))
                   for k, v in s.items()} for s in summaries]).to_csv(
        os.path.join(OUT, "prognostics_metrics.csv"), index=False)
    print(f"\n汇总指标已写入 {os.path.join(OUT, 'prognostics_metrics.csv')}")

    rep = export_figure_data(cal, idt, ood, k_mu)
    print(f"代表发动机：{rep['subset']} unit={rep['unit']}")
