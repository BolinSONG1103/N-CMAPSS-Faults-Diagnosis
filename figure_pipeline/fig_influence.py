"""图3-2 影响矩阵结构与可辨识性。

从辨识所得影响矩阵 H 与量程矩阵 THETA_SPAN 重建量程归一化算子 Hn=H·diag(span)，
展示：部件故障指纹（列方向热图）、奇异值谱（病态性）、部件列夹角（可辨识性/混叠）。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
HCSV = os.path.join(D.ROOT, "reference", "fixed_results", "influence_matrix", "H_9cols.csv")
THETA_SPAN = np.array([0.018668, 0.223446, 0.121209, 0.025540, 0.070658,
                       0.036194, 0.032908, 0.118767, 0.046187])
PARAM_CN = ["HPT效率", "Fan效率", "Fan流量", "HPC效率", "HPC流量",
            "LPT效率", "LPT流量", "LPC效率", "LPC流量"]
SENSOR_CN = ["Nf", "Nc", "Wf", "T24", "T30", "T48", "T50",
             "P15", "P21", "P24", "Ps30", "P40", "P50"]
DIVERGE = LinearSegmentedColormap.from_list(
    "bwr2", ["#2166AC", "#6DA9D2", "#F7F7F7", "#E08214", "#B2182B"])


def fig_influence():
    H = pd.read_csv(HCSV, index_col=0).values          # 13×9
    Hn = H * THETA_SPAN                                  # 量程归一化
    U, sv, Vt = np.linalg.svd(Hn, full_matrices=False)
    cond = sv[0] / sv[-1]

    fig = plt.figure(figsize=(13.6, 4.6))
    gs = GridSpec(1, 3, width_ratios=[1.25, 0.95, 1.05], wspace=0.5, figure=fig)

    # (a) 部件故障指纹：列方向归一化热图（每列单位化，凸显传感器响应方向）
    ax = fig.add_subplot(gs[0])
    Hdir = Hn / np.linalg.norm(Hn, axis=0, keepdims=True)
    vmax = np.abs(Hdir).max()
    im = ax.imshow(Hdir, cmap=DIVERGE, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(9)); ax.set_xticklabels(PARAM_CN, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(13)); ax.set_yticklabels(SENSOR_CN, fontsize=9)
    ax.set_title("(a) 部件故障指纹（列方向）", loc="left", fontsize=12.5)
    ax.set_xlabel("健康参数"); ax.set_ylabel("气路残差通道")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)

    # (b) 奇异值谱：病态性
    ax = fig.add_subplot(gs[1])
    ax.semilogy(range(1, 10), sv, "o-", color=S.D_COLOR, lw=1.8, mfc="white", ms=7)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("奇异值序号"); ax.set_ylabel("奇异值")
    ax.set_title("(b) 奇异值谱", loc="left", fontsize=12.5)
    ax.grid(alpha=0.4, which="both")
    ax.text(0.95, 0.9, f"cond(Hn) = {cond:.1f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#E0A060"))

    # (c) 部件列方向夹角矩阵：可辨识性/混叠
    ax = fig.add_subplot(gs[2])
    Hc = Hn / np.linalg.norm(Hn, axis=0, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.abs(Hc.T @ Hc), 0, 1)))
    np.fill_diagonal(ang, np.nan)
    cmap2 = plt.cm.YlGnBu_r
    im = ax.imshow(ang, cmap=cmap2, vmin=0, vmax=90, aspect="equal")
    ax.set_xticks(range(9)); ax.set_xticklabels(PARAM_CN, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(9)); ax.set_yticklabels(PARAM_CN, fontsize=8)
    ax.set_title("(c) 部件列方向夹角 (°)", loc="left", fontsize=12.5)
    # 标注最小夹角（最易混叠）
    amin = np.nanmin(ang); i, j = np.unravel_index(np.nanargmin(ang), ang.shape)
    ax.text(j, i, f"{amin:.0f}", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)

    fig.suptitle("影响矩阵结构与可辨识性", y=1.02, fontsize=13.5)
    print(f"cond(Hn) = {cond:.6f}; min column angle = {amin:.2f} deg "
          f"({PARAM_CN[i]}–{PARAM_CN[j]})")
    return S.save(fig, os.path.join(FIGDIR, "图3-2_影响矩阵结构与可辨识性"))


if __name__ == "__main__":
    S.apply()
    print(fig_influence())
