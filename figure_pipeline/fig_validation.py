"""图3-19：锁定物理约束管线在噪声、偏置与缺失通道下的补充鲁棒性。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR=os.path.join(D.ROOT,"thesis","figures")
ROB=os.path.join(D.T5,"robustness_summary.csv")


def fig_robustness():
    d=pd.read_csv(ROB); d=d[d.regime=="test_id"]
    fig=plt.figure(figsize=(11.8,4.8));gs=GridSpec(1,2,wspace=.32,figure=fig)
    ax=fig.add_subplot(gs[0]); n=d[d.perturbation=="noise"].sort_values("amplitude")
    base=n.iloc[0]
    ax.plot(n.amplitude,n.mean_RMSE/base.mean_RMSE,"o-",color=S.D_COLOR,label="RMSE相对倍数")
    ax.plot(n.amplitude,n.mean_detection_F1,"s-",color=S.B_COLOR,label="检测F1")
    ax.plot(n.amplitude,n.mean_isolation_macroF1,"^-",color=S.ACCENT,label="隔离macro-F1")
    ax.set_xlabel("附加噪声幅值（健康残差标准差倍数）");ax.set_ylabel("相对误差 / 指标值")
    ax.set_title("(a) 加性噪声鲁棒性",loc="left");ax.grid(alpha=.35);ax.legend()
    ax=fig.add_subplot(gs[1]);b=d[d.perturbation=="bias"].sort_values("amplitude")
    miss=d[d.perturbation=="missing_channel"]
    ax.plot(b.amplitude,b.mean_detection_F1,"o-",color=S.B_COLOR,label="检测F1")
    ax.plot(b.amplitude,b.mean_isolation_macroF1,"s-",color=S.ACCENT,label="隔离macro-F1")
    if len(miss):
        ax.scatter([b.amplitude.max()+.18],[miss.mean_detection_F1.iloc[0]],marker="D",s=65,color="#756BB1",label="单通道缺失：检测F1")
        ax.scatter([b.amplitude.max()+.18],[miss.mean_isolation_macroF1.iloc[0]],marker="P",s=70,color="#31A354",label="单通道缺失：隔离macro-F1")
    ax.set_xlabel("固定偏置幅值 / 缺失通道情形");ax.set_ylabel("指标值");ax.set_ylim(0,1.02)
    ax.set_title("(b) 偏置与通道缺失",loc="left");ax.grid(alpha=.35);ax.legend(fontsize=9)
    fig.suptitle("物理约束连续估计及其统一决策层的补充鲁棒性验证",y=1.01)
    return S.save(fig,os.path.join(FIGDIR,"图3-17_物理约束管线鲁棒性"))
