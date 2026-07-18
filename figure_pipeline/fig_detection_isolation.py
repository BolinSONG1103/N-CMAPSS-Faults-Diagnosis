"""图3-7 故障检测混淆矩阵；图3-8 五部件族故障隔离。

对齐参考论文（王昆图3-5）的混淆矩阵审美：序贯蓝色、单元格计数+占比、右侧色条、
中文轴标签。检测为二分类（健康/故障），隔离为五部件族多标签共现。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")


def draw_confusion(ax, M, row_labels, col_labels, title, tag,
                   normalize_rows=True, cbar=True, vmax=None):
    """在 ax 上绘制混淆矩阵（行=真值，列=预测）。"""
    M = np.asarray(M, dtype=float)
    row_sum = M.sum(axis=1, keepdims=True)
    Nshow = M / np.where(row_sum == 0, 1, row_sum) if normalize_rows else M
    vmax = vmax if vmax is not None else (1.0 if normalize_rows else M.max())
    im = ax.imshow(Nshow, cmap=S.CONF_CMAP, vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真值")
    ax.set_title(title, pad=8)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            frac = Nshow[i, j]
            txt = f"{frac*100:.0f}%\n({int(M[i,j])})" if normalize_rows else f"{int(M[i,j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10.5,
                    color="white" if frac > 0.55 * vmax else "#1a1a1a")
    S.panel_tag(ax, tag)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9)
    return im


def fig_detection():
    fig = plt.figure(figsize=(9.6, 4.2))
    gs = GridSpec(1, 2, wspace=0.42, figure=fig)
    labels = ["健康", "故障"]
    for k, (rg, tag, ttl) in enumerate([
            ("test_id", "(a)", "分布内测试"),
            ("ood_combo", "(b)", "未见故障组合")]):
        ax = fig.add_subplot(gs[0, k])
        M = D.detection_confusion(rg)
        draw_confusion(ax, M, labels, labels, ttl, tag, normalize_rows=True)
    fig.suptitle("发动机级故障检测混淆矩阵", y=1.02, fontsize=14)
    return S.save(fig, os.path.join(FIGDIR, "图3-5_故障检测混淆矩阵"))


def _aliasing_matrix(regime):
    """真值族 i 退化的循环中各族被预测的平均比例；对角=召回，非对角=族间混叠。"""
    fams = D.FAMILY_ORDER
    piv_t, piv_p = D.family_confusion(regime)
    T, P = piv_t.values, piv_p.values
    C = np.full((len(fams), len(fams)), np.nan)
    for i in range(len(fams)):
        rows = (T[:, i] == 1)
        if rows.sum():
            C[i] = P[rows].mean(axis=0)
    return C


def fig_family_isolation():
    fig = plt.figure(figsize=(13.6, 4.5))
    gs = GridSpec(1, 3, width_ratios=[1.02, 1.15, 1.0], wspace=0.55, figure=fig)
    fams = D.FAMILY_ORDER

    # (a) 分布内 5×5 族间混叠热图
    ax = fig.add_subplot(gs[0, 0])
    C = _aliasing_matrix("test_id")
    Cm = np.where(np.isnan(C), 0, C)
    im = ax.imshow(Cm, cmap=S.CONF_CMAP, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(fams))); ax.set_yticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=30, ha="right"); ax.set_yticklabels(fams)
    ax.set_xlabel("预测部件族"); ax.set_ylabel("真值部件族")
    ax.set_title("分布内族间隔离与混叠", pad=8)
    ax.set_xticks(np.arange(-.5, len(fams), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(fams), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(fams)):
        for j in range(len(fams)):
            if not np.isnan(C[i, j]) and C[i, j] >= 0.005:
                ax.text(j, i, f"{C[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=10, color="white" if C[i, j] > 0.55 else "#1a1a1a")
    S.panel_tag(ax, "(a)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9)

    # (b) 逐部件族 F1（分布内全部 5 族；未见组合仅激活的 HPT/LPT，其余标注虚警率）
    ax = fig.add_subplot(gs[0, 1])
    m_id = D.family_metrics("test_id")[0].set_index("family")
    m_ood = D.family_metrics("ood_combo")[0].set_index("family")
    y = np.arange(len(fams))[::-1]
    h = 0.38
    ax.barh(y - h/2, [m_id.loc[f, "F1"] for f in fams], height=h,
            color=S.REGIME_COLORS["test_id"], label="分布内", edgecolor="white", linewidth=0.6)
    for k, f in enumerate(fams):
        yy = y[k]
        if m_ood.loc[f, "support"] > 0:
            ax.barh(yy + h/2, m_ood.loc[f, "F1"], height=h,
                    color=S.REGIME_COLORS["ood_combo"],
                    label="未见组合" if f == "HPT" else None,
                    edgecolor="white", linewidth=0.6)
        else:  # 该族在未见组合中无真实退化
            ax.text(0.015, yy + h/2, f"未激活 (虚警率 {m_ood.loc[f,'FAR']*100:.0f}%)",
                    va="center", ha="left", fontsize=9, color="#888888")
    ax.set_yticks(y); ax.set_yticklabels(fams)
    ax.set_xlim(0, 1); ax.set_xlabel("F1 分数")
    ax.set_title("逐部件族隔离 F1", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.legend(loc="lower right")
    S.panel_tag(ax, "(b)")

    # (c) 部件族聚合收益：9 参数级 vs 5 部件族 macro-F1
    ax = fig.add_subplot(gs[0, 2])
    p9 = {"test_id": 0.4146, "ood_combo": 0.6724}   # 9 参数级 macro-F1（自举汇总）
    p5 = {"test_id": D.family_metrics("test_id")[1]["macro_f1"],
          "ood_combo": np.nanmean([m_ood.loc[f, "F1"] for f in fams
                                   if m_ood.loc[f, "support"] > 0])}
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w/2, [p9["test_id"], p9["ood_combo"]], w, color="#B0B0B0",
                label="9 参数级", edgecolor="white")
    b2 = ax.bar(x + w/2, [p5["test_id"], p5["ood_combo"]], w, color=S.ACCENT,
                label="5 部件族", edgecolor="white")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(["分布内", "未见组合\n(激活族)"])
    ax.set_ylim(0, 1); ax.set_ylabel("macro-F1")
    ax.set_title("部件族聚合的一致性收益", pad=8)
    ax.grid(axis="y", alpha=0.5)
    ax.legend(loc="upper left")
    S.panel_tag(ax, "(c)")

    fig.suptitle("五部件族故障隔离结果", y=1.04, fontsize=14)
    return S.save(fig, os.path.join(FIGDIR, "图3-6_五部件族故障隔离"))


if __name__ == "__main__":
    S.apply()
    print(fig_detection())
    print(fig_family_isolation())
    m_id, agg_id = D.family_metrics("test_id")
    m_ood, agg_ood = D.family_metrics("ood_combo")
    print("ID  family macro-F1 =", round(agg_id["macro_f1"], 3),
          " micro-F1 =", round(agg_id["micro_f1"], 3))
    print("OOD family macro-F1 =", round(agg_ood["macro_f1"], 3),
          " micro-F1 =", round(agg_ood["micro_f1"], 3))
    print(m_id.round(3).to_string(index=False))
    print(m_ood.round(3).to_string(index=False))
