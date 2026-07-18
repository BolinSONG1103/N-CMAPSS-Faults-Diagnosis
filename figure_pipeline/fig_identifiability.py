"""图3-4/3-5：影响矩阵故障指纹、病态性与子空间可辨识性。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
HCSV = os.path.join(D.ROOT, "reference", "fixed_results", "influence_matrix", "H_9cols.csv")
SPAN = np.array([0.018668, 0.223446, 0.121209, 0.025540, 0.070658,
                 0.036194, 0.032908, 0.118767, 0.046187])
PARAM_CN = ["HPT效率", "Fan效率", "Fan流量", "HPC效率", "HPC流量",
            "LPT效率", "LPT流量", "LPC效率", "LPC流量"]
SENSOR_CN = ["Nf", "Nc", "Wf", "T24", "T30", "T48", "T50",
             "P15", "P21", "P24", "Ps30", "P40", "P50"]
DIVERGE = LinearSegmentedColormap.from_list(
    "bwr2", ["#2166AC", "#6DA9D2", "#F7F7F7", "#E08214", "#B2182B"])
FAMILY_INDEX = {"HPT": [0], "Fan": [1, 2], "HPC": [3, 4], "LPT": [5, 6], "LPC": [7, 8]}


def _matrix():
    return pd.read_csv(HCSV, index_col=0).values * SPAN


def _principal_angle(A, idx):
    other = [j for j in range(A.shape[1]) if j not in idx]
    q1 = np.linalg.qr(A[:, idx])[0][:, :len(idx)]
    q2 = np.linalg.qr(A[:, other])[0][:, :len(other)]
    s = np.linalg.svd(q1.T @ q2, compute_uv=False)[0]
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def fig_fingerprint():
    Hn = _matrix(); Hdir = Hn / np.linalg.norm(Hn, axis=0, keepdims=True)
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    vmax = np.abs(Hdir).max()
    im = ax.imshow(Hdir, cmap=DIVERGE, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(9)); ax.set_xticklabels(PARAM_CN, rotation=35, ha="right")
    ax.set_yticks(range(13)); ax.set_yticklabels(SENSOR_CN)
    ax.set_xlabel("健康参数"); ax.set_ylabel("标准化气路残差通道")
    ax.set_title("量程归一化影响矩阵的部件故障指纹")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("列方向归一化响应")
    return S.save(fig, os.path.join(FIGDIR, "图3-4_影响矩阵故障指纹"))


def fig_condition_identifiability():
    Hn = _matrix(); sv = np.linalg.svd(Hn, compute_uv=False); cond = sv[0] / sv[-1]
    fam = list(FAMILY_INDEX)
    angles = np.array([_principal_angle(Hn, FAMILY_INDEX[k]) for k in fam])
    fig = plt.figure(figsize=(11.8, 4.7))
    gs = GridSpec(1, 2, wspace=0.34, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.semilogy(range(1, 10), sv, "o-", color=S.D_COLOR, mfc="white", ms=7)
    ax.set_xticks(range(1, 10)); ax.set_xlabel("奇异值序号"); ax.set_ylabel("奇异值（对数坐标）")
    ax.set_title("(a) 病态性：奇异值谱", loc="left"); ax.grid(alpha=0.35, which="both")
    ax.text(0.96, 0.92, f"cond($H_n$) = {cond:.1f}", transform=ax.transAxes,
            ha="right", va="top", bbox=dict(fc="#FFF3E0", ec="#D9A55A", pad=4))

    ax = fig.add_subplot(gs[1])
    cols = [S.FAMILY_COLORS[x] for x in fam]
    bars = ax.bar(fam, angles, color=cols, edgecolor="white")
    for b, v in zip(bars, angles):
        ax.text(b.get_x()+b.get_width()/2, v+0.7, f"{v:.2f}°", ha="center", fontsize=10)
    ax.axhline(10, color="#777777", ls="--", lw=1, label="10°参考线")
    ax.set_ylabel("该部件族子空间到其余部件子空间的最小主夹角")
    ax.set_title("(b) 部件族子空间可辨识性", loc="left")
    ax.grid(axis="y", alpha=0.35); ax.legend(loc="upper left")
    ax.text(0.98, 0.88, "HPT/LPT 子空间几乎与\n其余部件组合空间相交",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(fc="white", ec="#BBBBBB", alpha=.88, pad=3))
    fig.suptitle("影响矩阵病态性与可辨识性边界", y=1.02, fontsize=14)
    return S.save(fig, os.path.join(FIGDIR, "图3-5_病态性与子空间可辨识性"))
