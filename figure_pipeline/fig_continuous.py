"""图3-6至图3-9：连续退化估计的轨迹、误差、正则化与约束价值。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
TRAJ = os.path.join(D.OUT, "stage_bc", "figure_data", "fig3_2_ds03_unit13_trajectory.csv")
DATA = os.path.join(D.OUT, "complete_figures", "data")
B3 = os.path.join(DATA, "b3_lambda_trajectory.csv")
B4 = os.path.join(DATA, "b4_multiunit_component_error.csv")
ABL = os.path.join(D.T5, "ablation_unit_metrics.csv")
TABLE8 = os.path.join(D.ROOT, "thesis", "tables", "表8_外推技能与虚警.csv")
ACTIVE = ["HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod"]
PARAM_LABEL = {
    "HPT_eff_mod": "HPT效率", "fan_eff_mod": "Fan效率", "fan_flow_mod": "Fan流量",
    "HPC_eff_mod": "HPC效率", "HPC_flow_mod": "HPC流量", "LPT_eff_mod": "LPT效率",
    "LPT_flow_mod": "LPT流量", "LPC_eff_mod": "LPC效率", "LPC_flow_mod": "LPC流量",
}


def fig_tracking():
    t = pd.read_csv(TRAJ); inactive = [p for p in t.parameter.unique() if p not in ACTIVE]
    fig = plt.figure(figsize=(11.6, 7.0)); gs = GridSpec(2, 2, wspace=0.26, hspace=0.34, figure=fig)
    for k, p in enumerate(ACTIVE):
        ax = fig.add_subplot(gs[k // 2, k % 2]); d = t[t.parameter == p].sort_values("cycle_index")
        ax.plot(d.cycle_index, d.theta_true_pct, color=S.TRUTH_COLOR, lw=2.5, label="真值")
        ax.plot(d.cycle_index, d.theta_B_pct, color=S.B_COLOR, lw=1.2, ls="--", label="无约束 B")
        ax.plot(d.cycle_index, d.theta_D_pct, color=S.D_COLOR, lw=1.9, label="约束 D")
        ax.set_title(f"({chr(97+k)}) {PARAM_LABEL[p]}", loc="left")
        ax.set_xlabel("飞行循环"); ax.set_ylabel("健康参数变化 (%)"); ax.grid(alpha=0.35)
        if k == 0: ax.legend(loc="lower left")
    ax = fig.add_subplot(gs[1, 1])
    d = t[t.parameter.isin(inactive)]
    env_b = d.pivot(index="cycle_index", columns="parameter", values="theta_B_pct").abs().max(axis=1)
    env_d = d.pivot(index="cycle_index", columns="parameter", values="theta_D_pct").abs().max(axis=1)
    ax.fill_between(env_b.index, 0, env_b, color=S.B_COLOR, alpha=.16)
    ax.plot(env_b.index, env_b, color=S.B_COLOR, ls="--", label="无约束 B")
    ax.fill_between(env_d.index, 0, env_d, color=S.D_COLOR, alpha=.22)
    ax.plot(env_d.index, env_d, color=S.D_COLOR, label="约束 D")
    ax.set_title("(d) 未退化部件最大估计幅值", loc="left")
    ax.set_xlabel("飞行循环"); ax.set_ylabel("max |θ| (%)"); ax.grid(alpha=0.35); ax.legend()
    act = t[t.parameter.isin(ACTIVE)]
    rmse = np.sqrt(np.mean((act.theta_D_pct-act.theta_true_pct)**2))
    corr = np.corrcoef(act.theta_D_pct, act.theta_true_pct)[0, 1]
    fig.suptitle(f"代表发动机连续退化轨迹：相关系数 {corr:.3f}，RMSE {rmse:.3f}%", y=.99)
    return S.save(fig, os.path.join(FIGDIR, "图3-6_代表发动机连续退化轨迹"))


def fig_multiunit_error():
    d = pd.read_csv(B4); active = d[d.is_fault == 1].copy()
    fig = plt.figure(figsize=(11.8, 4.8)); gs = GridSpec(1, 2, width_ratios=[1.45, 1], wspace=.32, figure=fig)
    ax = fig.add_subplot(gs[0])
    units = active.subset.astype(str) + "-U" + active.unit.astype(str)
    colors = [S.FAMILY_COLORS["HPT"] if p.startswith("HPT") else S.FAMILY_COLORS["LPT"] for p in active.parameter]
    x = np.arange(len(active))
    ax.scatter(x, active.Z_zero_rmse_pct, marker="s", s=48, color="#AAAAAA", label="全零哨兵")
    ax.scatter(x, active.rmse_pct, marker="o", s=55, color=colors, edgecolor="white", label="约束反演 D")
    for i in range(len(active)):
        ax.plot([i, i], [active.rmse_pct.iloc[i], active.Z_zero_rmse_pct.iloc[i]], color="#CCCCCC", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(units, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("活动部件轨迹 RMSE (%)"); ax.set_title("(a) 跨发动机活动部件误差", loc="left")
    ax.grid(axis="y", alpha=.35); ax.legend(ncol=2, loc="upper left")

    ax = fig.add_subplot(gs[1])
    vals = [d[d.is_fault == 1].rmse_pct, d[d.is_fault == 0].rmse_pct,
            d[d.is_fault == 1].terminal_abs_error_pct]
    bp = ax.boxplot(vals, patch_artist=True, widths=.62, showfliers=True)
    for box, c in zip(bp["boxes"], [S.D_COLOR, S.B_COLOR, S.ACCENT]): box.set_facecolor(c); box.set_alpha(.72)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(["活动部件\n轨迹RMSE", "未活动部件\n轨迹RMSE", "活动部件\n终点误差"])
    ax.set_ylabel("误差 (%)"); ax.set_title("(b) 多口径误差分布", loc="left"); ax.grid(axis="y", alpha=.35)
    fig.suptitle("多发动机、多部件连续估计精度", y=1.01)
    return S.save(fig, os.path.join(FIGDIR, "图3-7_多发动机多部件估计误差"))


def fig_lambda_sensitivity():
    d = pd.read_csv(B3); lambdas = sorted(d["lambda"].unique())
    agg = []
    for lam in lambdas:
        z = d[(d["lambda"] == lam) & d.parameter.isin(ACTIVE)]
        agg.append(np.sqrt(np.mean((z.theta_hat_pct-z.theta_true_pct)**2)))
    fig = plt.figure(figsize=(11.8, 4.8)); gs = GridSpec(1, 2, width_ratios=[.85, 1.5], wspace=.30, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax.plot(lambdas, agg, "o-", color=S.D_COLOR, mfc="white", ms=8)
    ax.set_xscale("log"); ax.set_xlabel("Tikhonov 正则化参数 λ"); ax.set_ylabel("活动部件 RMSE (%)")
    ax.set_title("(a) 聚合精度敏感性", loc="left"); ax.grid(alpha=.35, which="both")
    best = int(np.argmin(agg)); ax.scatter([lambdas[best]], [agg[best]], s=90, color=S.ACCENT, zorder=4)
    ax.text(lambdas[best], agg[best], f"  主工作点 {lambdas[best]:g}", va="bottom")

    ax = fig.add_subplot(gs[1])
    p = "LPT_eff_mod"
    for lam, c in zip(lambdas, [S.B_COLOR, S.D_COLOR, S.ACCENT]):
        z = d[(d["lambda"] == lam) & (d.parameter == p)].sort_values("cycle")
        ax.plot(z.cycle, z.theta_hat_pct, color=c, label=f"λ={lam:g}")
    truth = d[(d["lambda"] == lambdas[0]) & (d.parameter == p)].sort_values("cycle")
    ax.plot(truth.cycle, truth.theta_true_pct, color=S.TRUTH_COLOR, lw=2.4, ls="--", label="真值")
    ax.set_xlabel("飞行循环"); ax.set_ylabel("LPT效率变化 (%)")
    ax.set_title("(b) 欠收缩—适中—过收缩的轨迹差异", loc="left"); ax.grid(alpha=.35); ax.legend(ncol=2)
    fig.suptitle("正则化参数对连续退化轨迹的影响", y=1.01)
    return S.save(fig, os.path.join(FIGDIR, "图3-8_正则化参数敏感性"))


def fig_constraint_value():
    ab = pd.read_csv(ABL)
    sub = ab[ab.method.isin(["D_full", "B_unconstrained"])]
    rmse = sub.groupby("method").RMSE.mean(); fa = sub.groupby("method").inactive_component_error.mean()
    fig = plt.figure(figsize=(11.8, 4.7)); gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=.34, figure=fig)
    ax = fig.add_subplot(gs[0]); x = np.arange(2); w=.36
    dvals = [rmse["D_full"], fa["D_full"]]; bvals=[rmse["B_unconstrained"], fa["B_unconstrained"]]
    ax.bar(x-w/2, dvals, w, color=S.D_COLOR, label="约束 D")
    ax.bar(x+w/2, bvals, w, color=S.B_COLOR, label="无约束 B")
    ax.set_xticks(x); ax.set_xticklabels(["估计RMSE", "未退化部件虚警"]); ax.set_ylabel("量程归一化误差")
    ax.set_title("(a) 硬约束的独立价值", loc="left"); ax.grid(axis="y", alpha=.35); ax.legend()
    for i in range(2): ax.text(i, max(dvals[i], bvals[i])*1.05, f"×{bvals[i]/dvals[i]:.1f}", ha="center")

    ax = fig.add_subplot(gs[1]); t8 = pd.read_csv(TABLE8); t8=t8[t8.arm!="Z_zero"].sort_values("ood_skill")
    name = {"D":"D（本章）", "E_Lin":"线性", "E_MixLinear":"线性时序", "E_MLP":"MLP", "E_WPMixer":"WPMixer"}
    y=np.arange(len(t8)); vals=t8.ood_skill.clip(lower=-.6)
    ax.barh(y, vals, color=[S.D_COLOR if a=="D" else "#6BAED6" for a in t8.arm])
    ax.set_yticks(y); ax.set_yticklabels([name.get(a,a) for a in t8.arm]); ax.axvline(0,color="#888",lw=.8)
    ax.set_xlabel("未见组合技能分"); ax.set_title("(b) 与数据驱动对照", loc="left"); ax.grid(axis="x", alpha=.35)
    for yi, (_, r) in enumerate(t8.iterrows()):
        ax.text(max(r.ood_skill,-.58)+.02, yi, f"虚警×{r.false_alarm_ratio_to_D:.0f}", va="center", fontsize=9)
    fig.suptitle("约束反演的误差与虚警抑制价值", y=1.01)
    return S.save(fig, os.path.join(FIGDIR, "图3-9_约束反演的独立价值"))
