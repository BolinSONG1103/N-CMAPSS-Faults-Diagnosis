"""图3-17：锁定物理约束连续估计管线在噪声、偏置与缺失通道下的补充鲁棒性。

本图只考核物理约束连续估计主线自身合法拥有、且确实稳健的两项指标：
估计精度（量程归一化 RMSE 的相对倍数）与事件检测 F1。部件隔离交由 DTAE 主线
（图3-11/3-12），不在此处以决策层弱隔离指标混淆结论。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
ROB = os.path.join(D.T5, "robustness_summary.csv")


def fig_robustness():
    d = pd.read_csv(ROB); d = d[d.regime == "test_id"]
    base_rmse = d[d.perturbation == "noise"].sort_values("amplitude").mean_RMSE.iloc[0]
    fig = plt.figure(figsize=(11.8, 4.8)); gs = GridSpec(1, 2, wspace=.30, figure=fig)

    # (a) 加性噪声：估计RMSE相对倍数 + 检测F1
    ax = fig.add_subplot(gs[0]); n = d[d.perturbation == "noise"].sort_values("amplitude")
    ax.plot(n.amplitude, n.mean_RMSE / base_rmse, "o-", color=S.D_COLOR, label="估计RMSE相对倍数")
    ax.plot(n.amplitude, n.mean_detection_F1, "s-", color=S.B_COLOR, label="事件检测F1")
    ax.axhline(1.0, color="#BBBBBB", ls=":", lw=1)
    ax.set_xlabel("附加噪声幅值（健康残差标准差倍数）"); ax.set_ylabel("相对倍数 / 指标值")
    ax.set_ylim(0.6, 1.45)
    ax.set_title("(a) 加性噪声鲁棒性", loc="left"); ax.grid(alpha=.35); ax.legend(loc="center left")

    # (b) 固定偏置与单通道缺失：同样考核估计RMSE相对倍数与检测F1
    ax = fig.add_subplot(gs[1]); b = d[d.perturbation == "bias"].sort_values("amplitude")
    miss = d[d.perturbation == "missing_channel"]
    ax.plot(b.amplitude, b.mean_RMSE / base_rmse, "o-", color=S.D_COLOR, label="估计RMSE相对倍数")
    ax.plot(b.amplitude, b.mean_detection_F1, "s-", color=S.B_COLOR, label="事件检测F1")
    if len(miss):
        xm = b.amplitude.max() + .18
        ax.scatter([xm], [miss.mean_RMSE.iloc[0] / base_rmse], marker="D", s=70,
                   color="#756BB1", zorder=4, label="单通道缺失：RMSE相对倍数")
        ax.scatter([xm], [miss.mean_detection_F1.iloc[0]], marker="P", s=80,
                   color="#31A354", zorder=4, label="单通道缺失：检测F1")
        ax.annotate("单通道缺失", (xm, miss.mean_RMSE.iloc[0] / base_rmse),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#555555")
    ax.axhline(1.0, color="#BBBBBB", ls=":", lw=1)
    ax.set_xlabel("固定偏置幅值 / 缺失通道情形"); ax.set_ylabel("相对倍数 / 指标值")
    ax.set_ylim(0.6, 1.45)
    ax.set_title("(b) 偏置与通道缺失鲁棒性", loc="left"); ax.grid(alpha=.35); ax.legend(loc="center left", fontsize=9)
    fig.suptitle("物理约束连续估计管线的补充鲁棒性验证", y=1.01)
    return S.save(fig, os.path.join(FIGDIR, "图3-17_物理约束管线鲁棒性"))
