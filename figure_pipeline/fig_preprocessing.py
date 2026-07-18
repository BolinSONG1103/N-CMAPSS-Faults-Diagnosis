"""图3-2/3-3：相似修正与健康基准质量审查。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
DATA = os.path.join(D.OUT, "complete_figures", "data")
A1 = os.path.join(DATA, "a1_similarity_correction.csv")
A2 = os.path.join(DATA, "a2_baseline_fit.csv")


def fig_similarity_correction():
    d = pd.read_csv(A1)
    summary = (d.groupby(["raw_channel", "stage"], as_index=False)
                 .first()[["raw_channel", "stage", "binned_drift_span_pct"]])
    p = summary.pivot(index="raw_channel", columns="stage", values="binned_drift_span_pct")
    order = ["Nf", "Nc", "Wf", "T24", "T30", "T48", "T50",
             "P15", "P21", "P24", "Ps30", "P40", "P50"]
    p = p.reindex(order)
    reduction = p["raw"] - p["corrected"]

    fig = plt.figure(figsize=(12.8, 5.0))
    gs = GridSpec(1, 2, width_ratios=[1.55, 1], wspace=0.30, figure=fig)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(p)); w = 0.37
    ax.bar(x - w/2, p["raw"], w, color="#9E9E9E", label="修正前")
    ax.bar(x + w/2, p["corrected"], w, color=S.D_COLOR, label="相似修正后")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("分箱漂移跨度 (%)")
    ax.set_title("(a) 13 个气路通道的工况漂移", loc="left")
    ax.grid(axis="y", alpha=0.35); ax.legend(ncol=2, loc="upper left")

    ax = fig.add_subplot(gs[1])
    colors = np.where(reduction >= 0, "#2E8B57", "#D95F02")
    y = np.arange(len(p))
    ax.barh(y, reduction, color=colors, edgecolor="white")
    ax.axvline(0, color="#777777", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(order); ax.invert_yaxis()
    ax.set_xlabel("漂移跨度减少量 (百分点)")
    ax.set_title("(b) 修正收益及剩余通道偏差", loc="left")
    ax.grid(axis="x", alpha=0.35)
    ax.text(0.02, 0.02, "绿色：漂移收敛；橙色：需由健康基准继续吸收",
            transform=ax.transAxes, fontsize=9, color="#444444")
    fig.suptitle("相似修正对工况漂移的影响", y=1.01, fontsize=14)
    return S.save(fig, os.path.join(FIGDIR, "图3-2_相似修正与工况漂移"))


def fig_baseline_quality():
    d = pd.read_csv(A2)
    channels = ["Nf_c", "T48_c", "P50_c"]
    fig = plt.figure(figsize=(12.8, 5.0))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.36, figure=fig)
    rng = np.random.RandomState(314159)
    for i, ch in enumerate(channels):
        ax = fig.add_subplot(gs[i])
        z = d[d.channel == ch]
        take = rng.choice(len(z), min(900, len(z)), replace=False)
        s = z.iloc[take]
        ax.scatter(s.measured, s.predicted, s=8, color=S.B_COLOR, alpha=0.22,
                   edgecolor="none", rasterized=True)
        lo = min(s.measured.min(), s.predicted.min()); hi = max(s.measured.max(), s.predicted.max())
        ax.plot([lo, hi], [lo, hi], "--", color=S.TRUTH_COLOR, lw=1.2)
        ax.set_title(f"({chr(97+i)}) {ch}", loc="left")
        ax.set_xlabel("实测值")
        if i == 0: ax.set_ylabel("健康基准预测值")
        ax.text(0.05, 0.92, f"$R^2$ = {z.R2.iloc[0]:.5f}", transform=ax.transAxes,
                va="top", fontsize=10)
        ax.grid(alpha=0.30)

    ax = fig.add_subplot(gs[3])
    q = d.groupby("channel", as_index=False).first()
    q["one_minus_r2_ppm"] = (1 - q.R2) * 1e6
    q = q.sort_values("one_minus_r2_ppm")
    ax.barh(np.arange(len(q)), q.one_minus_r2_ppm, color="#6BAED6", edgecolor="white")
    ax.set_yticks(np.arange(len(q))); ax.set_yticklabels(q.channel, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel(r"$(1-R^2)\times10^6$")
    ax.set_title("(d) 全通道拟合误差", loc="left")
    ax.grid(axis="x", alpha=0.30)
    fig.suptitle("健康基准拟合与残差质量", y=1.01, fontsize=14)
    return S.save(fig, os.path.join(FIGDIR, "图3-3_健康基准拟合与残差质量"))
