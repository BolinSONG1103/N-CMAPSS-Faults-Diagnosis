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
    """健康基准质量审查：以最差拟合通道给出诚实的拟合散点，
    并用全通道 (1-R²) 排序刻画基准整体质量与残差量级。"""
    d = pd.read_csv(A2)
    q = d.groupby("channel", as_index=False).first()
    q["one_minus_r2_ppm"] = (1 - q.R2) * 1e6
    q = q.sort_values("one_minus_r2_ppm")
    worst = q.channel.iloc[-1]                      # 最差拟合通道（诚实展示上界）

    fig = plt.figure(figsize=(11.8, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1, 1.25], wspace=0.30, figure=fig)

    # (a) 最差拟合通道的预测—实测散点：即便最难通道，残差也紧贴对角线
    ax = fig.add_subplot(gs[0])
    z = d[d.channel == worst]
    rng = np.random.RandomState(314159)
    take = rng.choice(len(z), min(1200, len(z)), replace=False)
    s = z.iloc[take]
    ax.scatter(s.measured, s.predicted, s=9, color=S.B_COLOR, alpha=0.22,
               edgecolor="none", rasterized=True)
    lo = min(s.measured.min(), s.predicted.min()); hi = max(s.measured.max(), s.predicted.max())
    ax.plot([lo, hi], [lo, hi], "--", color=S.TRUTH_COLOR, lw=1.3, label="理想 y=x")
    res_pct = 100 * (z.measured - z.predicted).std() / z.measured.mean()
    ax.text(0.05, 0.93, f"最差拟合通道 {worst}\n$R^2$ = {z.R2.iloc[0]:.4f}\n残差 std = {res_pct:.3f}% 读数",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="#FFF3E0", ec="#D9A55A", pad=4))
    ax.set_xlabel("实测值"); ax.set_ylabel("健康基准预测值")
    ax.set_title("(a) 最难通道的基准拟合", loc="left")
    ax.grid(alpha=0.30); ax.legend(loc="lower right")

    # (b) 全通道拟合误差排序：整体近乎完美，涡轮出口温度/压力通道相对最难
    ax = fig.add_subplot(gs[1])
    hot = {"T50_c", "P50_c", "T48_c"}
    colors = ["#D95F02" if c in hot else "#6BAED6" for c in q.channel]
    ax.barh(np.arange(len(q)), q.one_minus_r2_ppm, color=colors, edgecolor="white")
    ax.set_yticks(np.arange(len(q))); ax.set_yticklabels(q.channel, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel(r"$(1-R^2)\times10^6$（越小越好）")
    ax.set_title("(b) 全通道拟合误差排序", loc="left")
    ax.grid(axis="x", alpha=0.30)
    ax.text(0.97, 0.42, "橙色：涡轮出口温压通道\n（受强非线性影响，残差相对最大）",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color="#444444")
    fig.suptitle("健康基准拟合与残差质量", y=1.01, fontsize=14)
    return S.save(fig, os.path.join(FIGDIR, "图3-3_健康基准拟合与残差质量"))
