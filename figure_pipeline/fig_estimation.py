"""图3-4 部件退化轨迹估计（趋势与程度）；图3-5 约束反演的独立价值与虚警抑制。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
TRAJ = os.path.join(D.OUT, "stage_bc", "figure_data", "fig3_2_ds03_unit13_trajectory.csv")
ABL = os.path.join(D.T5, "ablation_unit_metrics.csv")
TAB = os.path.join(D.ROOT, "thesis", "tables")  # 表格位于 thesis/tables


def fig_trajectory():
    t = pd.read_csv(TRAJ)
    cyc = "cycle_index"
    active = [("HPT_eff_mod", "HPT 效率"), ("LPT_eff_mod", "LPT 效率"),
              ("LPT_flow_mod", "LPT 流量")]
    inactive = ["fan_eff_mod", "fan_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
                "LPC_eff_mod", "LPC_flow_mod"]

    fig = plt.figure(figsize=(11.6, 7.0))
    gs = GridSpec(2, 2, wspace=0.26, hspace=0.34, figure=fig)
    tags = ["(a)", "(b)", "(c)"]
    for k, (p, name) in enumerate(active):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        d = t[t.parameter == p].sort_values(cyc)
        ax.plot(d[cyc], d.theta_true_pct, color=S.TRUTH_COLOR, lw=2.6,
                label="真值", zorder=3)
        ax.plot(d[cyc], d.theta_B_pct, color=S.B_COLOR, lw=1.3, ls="--",
                alpha=0.9, label="无约束 B")
        ax.plot(d[cyc], d.theta_D_pct, color=S.D_COLOR, lw=2.0,
                label="约束反演 D")
        ax.set_title(f"{tags[k]} {name}", loc="left", fontsize=12.5)
        ax.set_xlabel("飞行循环"); ax.set_ylabel("健康参数变化 θ (%)")
        ax.grid(alpha=0.4)
        if k == 0:
            ax.legend(loc="lower left", fontsize=10)

    # (d) 未退化部件估计幅值包络：D 贴零，B 散布 → 虚警抑制
    ax = fig.add_subplot(gs[1, 1])
    piv_D = t[t.parameter.isin(inactive)].pivot_table(index=cyc, columns="parameter",
                                                      values="theta_D_pct")
    piv_B = t[t.parameter.isin(inactive)].pivot_table(index=cyc, columns="parameter",
                                                      values="theta_B_pct")
    envD = piv_D.abs().max(axis=1)
    envB = piv_B.abs().max(axis=1)
    ax.fill_between(envB.index, 0, envB.values, color=S.B_COLOR, alpha=0.18)
    ax.plot(envB.index, envB.values, color=S.B_COLOR, lw=1.6, ls="--", label="无约束 B")
    ax.fill_between(envD.index, 0, envD.values, color=S.D_COLOR, alpha=0.25)
    ax.plot(envD.index, envD.values, color=S.D_COLOR, lw=2.0, label="约束反演 D")
    ax.set_title("(d) 未退化部件估计幅值（虚警）", loc="left", fontsize=12.5)
    ax.set_xlabel("飞行循环"); ax.set_ylabel("6 个未退化参数 max|θ| (%)")
    ax.grid(alpha=0.4); ax.legend(loc="upper left", fontsize=10)

    fig.suptitle("约束反演的部件退化轨迹估计：DS03 发动机 13", y=0.98, fontsize=13.5)
    return S.save(fig, os.path.join(FIGDIR, "图3-3_部件退化轨迹估计"))


def fig_constraint_value():
    ab = pd.read_csv(ABL)
    fig = plt.figure(figsize=(11.8, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.12], wspace=0.36, figure=fig)

    # (a) 约束 vs 无约束：RMSE 与未退化部件虚警（同轴分组条形）
    ax = fig.add_subplot(gs[0])
    sub = ab[ab.method.isin(["D_full", "B_unconstrained"])]
    rmse = {m: sub[sub.method == m]["RMSE"].mean() for m in ["D_full", "B_unconstrained"]}
    fa = {m: sub[sub.method == m]["inactive_component_error"].mean()
          for m in ["D_full", "B_unconstrained"]}
    groups = ["估计 RMSE", "未退化部件虚警"]
    x = np.arange(2); w = 0.36
    dvals = [rmse["D_full"], fa["D_full"]]
    bvals = [rmse["B_unconstrained"], fa["B_unconstrained"]]
    bd = ax.bar(x - w/2, dvals, w, color=S.D_COLOR, label="约束反演 D", edgecolor="white")
    bb = ax.bar(x + w/2, bvals, w, color=S.B_COLOR, label="无约束 B", edgecolor="white")
    for b in list(bd) + list(bb):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.006,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9.3)
    # 倍数标注
    ax.text(0, max(bvals[0], dvals[0]) + 0.045, f"×{bvals[0]/dvals[0]:.0f}",
            ha="center", fontsize=10, color="#333", fontweight="bold")
    ax.text(1, bvals[1] + 0.045, f"×{bvals[1]/dvals[1]:.0f}",
            ha="center", fontsize=10, color="#333", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("量程归一化误差"); ax.set_ylim(0, 0.36)
    ax.set_title("(a) 硬约束的独立价值", loc="left", fontsize=12.5)
    ax.grid(axis="y", alpha=0.4); ax.legend(loc="upper right", fontsize=10)

    # (b) 与数据驱动方法的组合外推对照：技能分 + 虚警倍数
    ax = fig.add_subplot(gs[1])
    t8 = pd.read_csv(os.path.join(TAB, "表8_外推技能与虚警.csv"))
    t8 = t8[t8.arm != "Z_zero"].copy()
    name_map = {"D": "D（本章）", "E_Lin": "线性", "E_MixLinear": "线性时序",
                "E_MLP": "MLP", "E_WPMixer": "WPMixer"}
    t8["name"] = t8.arm.map(name_map)
    t8 = t8.sort_values("ood_skill")
    y = np.arange(len(t8))
    palette = {"D": S.D_COLOR, "E_Lin": "#1F77B4", "E_MixLinear": "#2CA02C",
               "E_MLP": "#FF7F0E", "E_WPMixer": "#9467BD"}
    cols = [palette[a] for a in t8.arm]
    skill_disp = t8.ood_skill.clip(lower=-0.6)   # WPMixer 越界，单独标注
    ax.barh(y, skill_disp, color=cols, edgecolor="white", height=0.62)
    for yi, (_, r) in zip(y, t8.iterrows()):
        lbl = f"技能 {r.ood_skill:.2f}｜虚警 ×{r.false_alarm_ratio_to_D:.0f}"
        xoff = 0.02 if r.ood_skill >= 0 else 0.02
        ha = "left"
        xpos = max(r.ood_skill, -0.6) + xoff
        ax.text(xpos if r.ood_skill > -0.5 else -0.55, yi, lbl, va="center",
                ha="left", fontsize=9.2, color="#222")
    ax.axvline(0, color="#999", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(t8.name)
    ax.set_xlim(-0.65, 1.35); ax.set_xlabel("组合外推技能分")
    ax.set_title("(b) 物理约束 vs 数据驱动（未见组合）", loc="left", fontsize=12.5)
    ax.grid(axis="x", alpha=0.4)
    fig.suptitle("约束反演的独立价值与虚警抑制", y=1.0, fontsize=13.5)
    return S.save(fig, os.path.join(FIGDIR, "图3-4_约束反演的独立价值"))


if __name__ == "__main__":
    S.apply()
    print(fig_trajectory())
    print(fig_constraint_value())
