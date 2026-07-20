"""图3-18、图3-19：退化趋势预测（主线三）——轨迹外推、RUL 预测与精度随观测寿命演化。

数据来源：improve/figdata/prog_trajectory.csv、prog_funnel.csv、prog_accuracy.csv
（由 improve/trend_prediction.py 在真实退化序列上生成，不重新拟合、不触碰测试真值选参）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
FD = os.path.join(D.ROOT, "improve", "figdata")


def fig_trajectory_rul():
    """图3-18：代表发动机退化轨迹外推 + alpha-lambda 剩余寿命预测漏斗。"""
    t = pd.read_csv(os.path.join(FD, "prog_trajectory.csv"), encoding="utf-8-sig")
    f = pd.read_csv(os.path.join(FD, "prog_funnel.csv"), encoding="utf-8-sig")
    anchor = int(t["anchor"].iloc[0]); thr = float(t["thr"].iloc[0])
    subset = str(t["subset"].iloc[0]); unit = int(t["unit"].iloc[0])

    fig = plt.figure(figsize=(12.4, 4.9))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.28, figure=fig)

    # (a) 轨迹外推
    ax = fig.add_subplot(gs[0])
    hist = t.cycle <= anchor
    ax.fill_between(t.cycle, t.band_lo, t.band_hi, color=S.D_COLOR, alpha=0.14,
                    label="90% 预测区间")
    ax.plot(t.cycle, t.hi_true, color=S.TRUTH_COLOR, lw=2.4, label="真实退化轨迹")
    ax.plot(t.cycle[hist], t.hi_true[hist], color="#1F77B4", lw=3.0, alpha=0.9,
            label="观测历史", solid_capstyle="round")
    ax.plot(t.cycle[~hist], t.hi_pred[~hist], color=S.D_COLOR, lw=2.2, ls="--",
            label="趋势外推")
    ax.axhline(thr, color="#777777", ls=":", lw=1.3)
    ax.text(t.cycle.min(), thr + 0.012, f"维修阈值 {thr:.2f}", fontsize=9, color="#555555")
    ax.axvline(anchor, color="#999999", lw=1.0, ls="-", alpha=0.7)
    ax.text(anchor, ax.get_ylim()[1] * 0.02, "预测发起点", rotation=90, va="bottom",
            ha="right", fontsize=9, color="#666666")
    # 真值/预测越阈循环标记
    def _cross(y):
        idx = np.where(np.asarray(y) >= thr)[0]
        return int(t.cycle.iloc[idx[0]]) if idx.size else None
    ct, cp = _cross(t.hi_true), _cross(t.hi_pred)
    if ct is not None:
        ax.plot([ct], [thr], "o", color=S.TRUTH_COLOR, ms=8, zorder=5)
    if cp is not None:
        ax.plot([cp], [thr], "D", color=S.D_COLOR, ms=8, zorder=5)
    if ct is not None and cp is not None:
        ax.annotate(f"越阈误差 {abs(cp-ct)} 循环", xy=(max(ct, cp), thr),
                    xytext=(0.60, 0.30), textcoords="axes fraction", fontsize=9,
                    color="#333333", ha="left")
    ax.set_xlabel("飞行循环"); ax.set_ylabel("量程归一化健康指标")
    ax.set_title(f"(a) 退化轨迹外推（{subset} u{unit}，自 {anchor} 循环外推）", loc="left")
    ax.set_ylim(bottom=-0.02)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9.5)

    # (b) alpha-lambda RUL 漏斗
    ax = fig.add_subplot(gs[1])
    ax.fill_between(f.cycle, f.band_lo, f.band_hi, color="#2CA02C", alpha=0.16,
                    label="±20% 精度带")
    ax.plot(f.cycle, f.rul_true, color=S.TRUTH_COLOR, lw=2.3, label="真实 RUL")
    ax.plot(f.cycle, f.rul_pred, color=S.D_COLOR, lw=1.6, marker="o", ms=3.4,
            markevery=2, label="预测 RUL")
    ax.set_xlabel("飞行循环"); ax.set_ylabel("剩余寿命 RUL（循环）")
    ax.set_title("(b) 逐循环 RUL 预测与精度带收敛", loc="left")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9.5)
    mae = float(np.abs(f.rul_pred - f.rul_true).mean())
    ax.text(0.03, 0.06, f"该机 RUL MAE = {mae:.1f} 循环", transform=ax.transAxes,
            fontsize=9.5, bbox=dict(fc="#FFF3E0", ec="#D9A55A", pad=4))

    fig.suptitle("退化趋势外推与剩余寿命预测（连续估计→趋势预测闭环）", y=1.02)
    return S.save(fig, os.path.join(FIGDIR, "图3-18_退化趋势外推与RUL预测"))


def fig_accuracy_vs_life():
    """图3-19：趋势预测精度随观测寿命比例演化（ID 分布内 vs OOD 未见组合）。"""
    a = pd.read_csv(os.path.join(FD, "prog_accuracy.csv"), encoding="utf-8-sig")
    col = {"ID": S.REGIME_COLORS["test_id"], "OOD": S.REGIME_COLORS["ood_combo"]}
    lab = {"ID": "分布内测试", "OOD": "未见组合"}

    fig = plt.figure(figsize=(12.0, 4.7))
    gs = GridSpec(1, 2, wspace=0.26, figure=fig)

    ax = fig.add_subplot(gs[0])
    for scope in ["ID", "OOD"]:
        d = a[a.scope == scope].sort_values("bin")
        ax.plot(d.bin * 100, d.rul_abs, marker="o", ms=6, lw=2.1, color=col[scope],
                label=lab[scope])
    ax.set_xlabel("已观测寿命比例（%）"); ax.set_ylabel("RUL 绝对误差（循环）")
    ax.set_title("(a) 剩余寿命预测误差随观测演化", loc="left")
    ax.axhline(2.0, color="#999999", ls="--", lw=1.0)
    ax.text(41, 2.1, "2 循环参考线", fontsize=9, color="#666666")
    ax.grid(alpha=0.3); ax.legend()

    ax = fig.add_subplot(gs[1])
    for scope in ["ID", "OOD"]:
        d = a[a.scope == scope].sort_values("bin")
        ax.plot(d.bin * 100, d.traj * 100, marker="s", ms=6, lw=2.1, color=col[scope],
                label=lab[scope])
    ax.set_xlabel("已观测寿命比例（%）"); ax.set_ylabel("轨迹外推 RMSE（%量程）")
    ax.set_title("(b) 退化轨迹外推精度随观测演化", loc="left")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("退化趋势预测精度随观测寿命比例的演化", y=1.02)
    return S.save(fig, os.path.join(FIGDIR, "图3-19_趋势预测精度随观测寿命"))


if __name__ == "__main__":
    S.apply()
    print(fig_trajectory_rul())
    print(fig_accuracy_vs_life())
