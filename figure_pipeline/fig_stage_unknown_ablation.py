"""图3-8 有序退化等级判定；图3-9 未知故障拒识；图3-10 消融与鲁棒性。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")


def _draw_stage_conf(ax, M, tag, title):
    row = M.sum(1, keepdims=True)
    N = M / np.where(row == 0, 1, row)
    im = ax.imshow(N, cmap=S.CONF_CMAP, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(S.STAGE_CN, rotation=25, ha="right"); ax.set_yticklabels(S.STAGE_CN)
    ax.set_xlabel("预测"); ax.set_ylabel("真值"); ax.set_title(title, pad=8)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4); ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(4):
        for j in range(4):
            if N[i, j] >= 0.005:
                ax.text(j, i, f"{N[i,j]*100:.0f}", ha="center", va="center", fontsize=9.5,
                        color="white" if N[i, j] > 0.55 else "#1a1a1a")
    S.panel_tag(ax, tag)
    return im


def fig_stage():
    sens = pd.read_csv(os.path.join(D.T4, "stage_component_sensitivity.csv"))
    fig = plt.figure(figsize=(13.4, 4.4))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.5, figure=fig)
    ax = fig.add_subplot(gs[0]); _draw_stage_conf(ax, D.stage_confusion("test_id"), "(a)", "分布内测试")
    ax = fig.add_subplot(gs[1]); im = _draw_stage_conf(ax, D.stage_confusion("ood_combo"), "(b)", "未见故障组合")
    cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.025, pad=0.02); cb.outline.set_visible(False)
    # (c) 成分数敏感性（2/3/4）weighted kappa
    ax = fig.add_subplot(gs[2])
    for rg, lab in [("test_id", "分布内"), ("ood_combo", "未见组合")]:
        s = sens[sens.regime == rg].sort_values("stage_components")
        ax.plot(s.stage_components, s.weighted_kappa, "o-", color=S.REGIME_COLORS[rg],
                lw=2, ms=8, mfc="white", label=lab)
    ax.axhline(0, color="#999", ls=":", lw=0.9)
    ax.set_xticks([2, 3, 4]); ax.set_ylim(0, 1)
    ax.set_xlabel("退化后等级数"); ax.set_ylabel("加权 Cohen kappa")
    ax.set_title("等级定义敏感性", pad=8); ax.grid(alpha=0.4)
    ax.legend(loc="lower right"); S.panel_tag(ax, "(c)")
    fig.suptitle("有序退化等级判定（三成分为主定义）", y=1.02, fontsize=13.5)
    return S.save(fig, os.path.join(FIGDIR, "图3-8_有序退化等级判定"))


def fig_unknown():
    summ = pd.read_csv(os.path.join(D.T4, "unknown_family_summary.csv"))
    pts = pd.read_csv(os.path.join(D.T4, "unknown_family_points.csv"))
    fams = ["HPT_eff", "Fan", "HPC", "LPT", "LPC"]
    fam_cn = {"HPT_eff": "HPT", "Fan": "Fan", "HPC": "HPC", "LPT": "LPT", "LPC": "LPC"}
    fig = plt.figure(figsize=(12.6, 4.6))
    gs = GridSpec(1, 2, width_ratios=[1.35, 1], wspace=0.3, figure=fig)

    # (a) 逐留一折：重构不一致度散点（已知 vs 未知）+ 阈值
    ax = fig.add_subplot(gs[0])
    for k, fam in enumerate(fams):
        p = pts[pts.family == fam]
        thr = p.threshold.iloc[0]
        known = p[p.is_unknown == 0].score.values
        unk = p[p.is_unknown == 1].score.values
        jit = lambda n: (np.random.RandomState(k).rand(n) - .5) * 0.28
        ax.scatter(k + jit(len(known)), known, s=55, color="#4C72B0",
                   edgecolor="white", linewidth=0.8, zorder=3, label="已知故障族" if k == 0 else None)
        ax.scatter(k + jit(len(unk)), unk, s=95, marker="^", color=S.D_COLOR,
                   edgecolor="white", linewidth=0.8, zorder=4, label="留出未知故障族" if k == 0 else None)
        ax.plot([k - 0.35, k + 0.35], [thr, thr], color="#333", lw=1.6, ls="--",
                label="拒识阈值" if k == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(5)); ax.set_xticklabels([fam_cn[f] for f in fams])
    ax.set_xlabel("留出故障族"); ax.set_ylabel("物理重构不一致度 q（对数轴）")
    ax.set_title("(a) 留一故障族拒识得分", loc="left", fontsize=12.5)
    ax.grid(alpha=0.35, which="both"); ax.legend(loc="upper center", ncol=3, fontsize=9)

    # (b) 逐折未知拒识率与已知接受率
    ax = fig.add_subplot(gs[1])
    order = summ.set_index("family").loc[fams]
    y = np.arange(5)[::-1]; h = 0.38
    ax.barh(y + h/2, order.unknown_reject_rate.values, h, color=S.D_COLOR,
            edgecolor="white", label="未知拒识率")
    ax.barh(y - h/2, order.known_accept_rate.values, h, color="#4C72B0",
            edgecolor="white", label="已知接受率")
    for yi, f in zip(y, fams):
        rr = order.loc[f, "unknown_reject_rate"]
        if rr == 0:
            ax.text(0.02, yi + h/2, "拒识失败", va="center", fontsize=9, color=S.D_COLOR)
    ax.set_yticks(y); ax.set_yticklabels([fam_cn[f] for f in fams])
    ax.set_xlim(0, 1.05); ax.set_xlabel("比率")
    ax.set_title("(b) 逐折拒识与接受率", loc="left", fontsize=12.5)
    ax.grid(axis="x", alpha=0.4); ax.legend(loc="lower right", fontsize=9.5)
    fig.suptitle("未知故障拒识（留一故障族，逐折报告）", y=1.02, fontsize=13.5)
    return S.save(fig, os.path.join(FIGDIR, "图3-9_未知故障拒识"))


def fig_ablation():
    ab = pd.read_csv(os.path.join(D.T5, "ablation_unit_metrics.csv"))
    rob = pd.read_csv(os.path.join(D.T5, "robustness_summary.csv"))
    fig = plt.figure(figsize=(13.0, 4.6))
    gs = GridSpec(1, 2, width_ratios=[1.15, 1.1], wspace=0.32, figure=fig)

    # (a) 消融（分布内）：检测F1 / 隔离macroF1 / 阶段kappa
    ax = fig.add_subplot(gs[0])
    methods = ["D_full", "D_no_shrink", "sign_only", "B_unconstrained"]
    mlab = {"D_full": "完整 D", "D_no_shrink": "去收缩", "sign_only": "仅符号约束",
            "B_unconstrained": "无约束 B"}
    metrics = [("detection_F1", "检测 F1"), ("isolation_macroF1", "隔离 macro-F1"),
               ("stage_kappa", "阶段 kappa")]
    x = np.arange(len(metrics)); w = 0.2
    cols = [S.D_COLOR, "#E8A33D", "#7BA0C4", S.B_COLOR]
    idsub = ab[ab.regime == "test_id"]
    for i, m in enumerate(methods):
        vals = [idsub[idsub.method == m][mc].mean() for mc, _ in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, color=cols[i], label=mlab[m], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylim(0, 1); ax.set_ylabel("指标值")
    ax.set_title("(a) 结构消融（分布内测试）", loc="left", fontsize=12.5)
    ax.grid(axis="y", alpha=0.4); ax.legend(loc="upper right", fontsize=8.6, ncol=2)

    # (b) 噪声鲁棒性：随高斯噪声幅度的性能
    ax = fig.add_subplot(gs[1])
    noise = rob[rob.perturbation == "noise"].sort_values("amplitude")
    styles = {"test_id": "-", "ood_combo": "--"}
    series = [("mean_detection_F1", "检测 F1", S.D_COLOR, "o"),
              ("mean_isolation_macroF1", "隔离 macro-F1", "#2CA02C", "s"),
              ("mean_stage_kappa", "阶段 kappa", "#1F77B4", "^")]
    for col, lab, c, mk in series:
        for rg in ["test_id", "ood_combo"]:
            s = noise[noise.regime == rg]
            ax.plot(s.amplitude, s[col], styles[rg], color=c, marker=mk, ms=6, lw=1.8,
                    mfc="white", label=f"{lab}·{S.REGIME_LABEL[rg]}")
    ax.set_xlabel("高斯噪声幅度（σ 倍）"); ax.set_ylabel("指标值"); ax.set_ylim(0, 1)
    ax.set_title("(b) 噪声鲁棒性", loc="left", fontsize=12.5); ax.grid(alpha=0.4)
    ax.legend(loc="lower left", fontsize=7.6, ncol=2)
    fig.suptitle("结构消融与噪声鲁棒性", y=1.02, fontsize=13.5)
    return S.save(fig, os.path.join(FIGDIR, "图3-10_消融与鲁棒性"))


if __name__ == "__main__":
    S.apply()
    print(fig_stage())
    print(fig_unknown())
    print(fig_ablation())
