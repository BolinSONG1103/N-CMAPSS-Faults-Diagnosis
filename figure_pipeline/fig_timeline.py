"""图3-8 诊断闭环时间线（代表发动机 DS03 unit 13，未见故障组合）。

在单台代表发动机的整个寿命轨迹上，将连续估计→故障检测→部件族隔离→有序退化等级
四个环节沿飞行循环并置展示，说明本章方法如何形成从异常出现到部件级健康状态的闭环。
代表发动机在查看方法结果前按固定规则选定（与退化轨迹图一致）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
STAGE_COLORS = ["#E8E8E8", "#FFD24C", "#FB8C00", "#D62728"]  # 健康/早期/中期/严重


def _engine(regime="ood_combo", subset="DS03", unit=13):
    df = D.raw_cycles()
    return df[(df.regime == regime) & (df.subset == subset) & (df.unit == unit)].copy()


def _family_severity(e):
    """逐循环五部件族严重度 z=RMS(max(0,-theta_hat))。"""
    e = e.copy()
    e["sev"] = np.maximum(0, -e["theta_hat"])
    fam = (e.groupby(["cycle", "family"])["sev"]
             .apply(lambda s: np.sqrt(np.mean(s.values ** 2))).reset_index())
    piv = fam.pivot(index="cycle", columns="family", values="sev").sort_index()
    return piv


def fig_timeline():
    e = _engine()
    cycles = np.sort(e["cycle"].unique())
    sev = _family_severity(e)
    # 逐循环真值/预测族标签
    lab = (e.groupby(["cycle", "family"]).agg(t=("truth_label", "max"),
                                              p=("pred_label", "max")).reset_index())
    piv_t = lab.pivot(index="cycle", columns="family", values="t").reindex(cycles)
    piv_p = lab.pivot(index="cycle", columns="family", values="p").reindex(cycles)
    # 逐循环有序等级
    stg = e.groupby("cycle").agg(ts=("truth_stage", "first"),
                                 ps=("pred_stage", "first")).reindex(cycles)

    fig = plt.figure(figsize=(11.4, 7.4))
    gs = GridSpec(3, 1, height_ratios=[1.55, 0.9, 0.9], hspace=0.34, figure=fig)

    # ---- (a) 部件族严重度轨迹 + 检测阈值 + 检测起点 ----
    ax = fig.add_subplot(gs[0])
    active = ["HPT", "LPT"]
    for f in active:
        ax.plot(sev.index, sev[f], color=S.FAMILY_COLORS[f], lw=2.0, label=f"{f} 严重度")
    ymax = max(sev[active].max().max(), 0.1)
    ax.set_ylim(-0.02, ymax * 1.18)
    # 真实故障起点（首个 truth_label=1 的循环）
    onset = int(e.loc[e["truth_label"] == 1, "cycle"].min())
    ax.axvline(onset, color="#555555", ls="--", lw=1.3)
    # 检测报警起点（发动机级：任一族预测为1的首循环）
    det = piv_p.max(axis=1)
    det_on = int(det[det > 0].index.min())
    ax.axvline(det_on, color=S.D_COLOR, ls=":", lw=1.7)
    # 竖线标注置于底部，避免与曲线图例碰撞
    ax.annotate(f"检测报警\n(第{det_on}循环)", xy=(det_on, 0), xytext=(det_on - 1, ymax * 0.44),
                fontsize=9.5, color=S.D_COLOR, ha="right", va="center")
    ax.annotate(f"真实故障起点\n(第{onset}循环)", xy=(onset, 0), xytext=(onset + 1.2, ymax * 0.72),
                fontsize=9.5, color="#555555", ha="left", va="center")
    ax.set_ylabel("部件族严重度 z")
    ax.set_title("(a) 连续估计与故障检测", loc="left", fontsize=12.5)
    ax.grid(alpha=0.4)
    ax.legend(loc="upper left", ncol=1, framealpha=0.95)
    ax.set_xlim(cycles.min(), cycles.max())

    # ---- (b) 有序退化等级时间线（真值 vs 预测）----
    ax = fig.add_subplot(gs[1])
    cmap = ListedColormap(STAGE_COLORS)
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
    M = np.vstack([stg["ts"].values, stg["ps"].values])
    ax.imshow(M, aspect="auto", cmap=cmap, norm=norm,
              extent=[cycles.min(), cycles.max(), 0, 2], interpolation="nearest")
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["预测", "真值"])
    ax.set_title("(b) 有序退化等级", loc="left", fontsize=12.5)
    ax.set_xlim(cycles.min(), cycles.max()); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    legend_stage = [Patch(facecolor=STAGE_COLORS[i], edgecolor="#999",
                          label=S.STAGE_CN[i]) for i in range(4)]
    ax.legend(handles=legend_stage, loc="center left", bbox_to_anchor=(1.005, 0.5),
              fontsize=10, title="等级")

    # ---- (c) 部件族隔离时间线（真值 vs 预测）----
    ax = fig.add_subplot(gs[2])
    fams_plot = ["HPT", "LPT"]           # 该发动机激活族
    rows, ylabels, colors = [], [], []
    for f in fams_plot:
        rows.append(piv_t[f].values); ylabels.append(f"{f} 真值"); colors.append(f)
        rows.append(piv_p[f].values); ylabels.append(f"{f} 预测"); colors.append(f)
    for i, (r, f) in enumerate(zip(rows, colors)):
        base = len(rows) - 1 - i
        active_c = np.where(r > 0)[0]
        ax.broken_barh([(cycles[k] - 0.5, 1) for k in active_c], (base, 0.8),
                       facecolors=S.FAMILY_COLORS[f], edgecolor="none")
    ax.set_ylim(0, len(rows)); ax.set_yticks(np.arange(len(rows)) + 0.4)
    ax.set_yticklabels(ylabels[::-1], fontsize=10)
    ax.set_xlim(cycles.min(), cycles.max())
    ax.set_xlabel("飞行循环")
    ax.set_title("(c) 部件族隔离", loc="left", fontsize=12.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.suptitle("诊断闭环时间线：DS03 发动机 13（未见故障组合，HPT+LPT 退化）",
                 y=0.985, fontsize=13.5)
    return S.save(fig, os.path.join(FIGDIR, "图3-7_诊断闭环时间线"))


if __name__ == "__main__":
    S.apply()
    print(fig_timeline())
