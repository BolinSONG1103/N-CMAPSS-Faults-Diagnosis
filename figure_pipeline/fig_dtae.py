"""图3-11至图3-18：DTAE 检测、隔离、潜空间、时间线与诊断粒度分析。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

import style as S
import dataio as D

FIGDIR = os.path.join(D.ROOT, "thesis", "figures")
FD = os.path.join(D.ROOT, "improve", "figdata")
PART_METRICS = os.path.join(D.ROOT, "improve", "dtae_metrics_part.csv")
FAMILY_METRICS = os.path.join(D.ROOT, "improve", "dtae_metrics_family.csv")


def _read_matrix(name):
    return pd.read_csv(os.path.join(FD, name), index_col=0, encoding="utf-8-sig")


def _heatmap(ax, df, title, percent=True):
    M = df.values.astype(float)
    im = ax.imshow(M, cmap=S.CONF_CMAP, vmin=0, vmax=max(1, np.nanmax(M)), aspect="equal")
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=28, ha="right")
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别"); ax.set_title(title)
    ax.set_xticks(np.arange(-.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4); ax.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = f"{100*v:.1f}%" if percent else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if v > .58 * max(1, np.nanmax(M)) else "#222222")
    for sp in ax.spines.values(): sp.set_visible(False)
    return im


def fig_detection():
    conf = _read_matrix("detection_confusion.csv")
    row = conf.div(conf.sum(axis=1), axis=0)
    summary = pd.read_csv(os.path.join(FD, "detection_summary.csv"))
    vals = dict(zip(summary.metric, summary.value))
    fig = plt.figure(figsize=(12.0, 4.6)); gs = GridSpec(1, 2, width_ratios=[1, 1.35], wspace=.35, figure=fig)
    ax = fig.add_subplot(gs[0]); _heatmap(ax, row, "(a) 逐循环检测混淆矩阵")
    ax = fig.add_subplot(gs[1])
    labels = ["事件级\n检出率", "健康循环\n正确拒绝率", "四部件\nmacro-F1", "四部件\nmicro-F1"]
    values = [vals["event_detection_rate"], 1-vals["healthy_cycle_far"], vals["macro_f1"], vals["micro_f1"]]
    bars = ax.bar(labels, values, color=[S.D_COLOR, "#5AAE61", S.B_COLOR, S.ACCENT], edgecolor="white")
    ax.set_ylim(0, 1.08); ax.axhline(.95, color="#777777", ls="--", lw=1, label="0.95参考线")
    ax.set_ylabel("指标值"); ax.set_title("(b) 检测与主隔离关键指标", loc="left"); ax.grid(axis="y", alpha=.35)
    ax.tick_params(axis="x", labelsize=10)
    for b, v in zip(bars, values): ax.text(b.get_x()+b.get_width()/2, v+.018, f"{v:.3f}", ha="center")
    ax.legend(loc="lower right")
    fig.suptitle("DTAE 两阶段故障检测（分布内在线诊断）", y=1.02)
    return S.save(fig, os.path.join(FIGDIR, "图3-11_DTAE故障检测"))


def fig_part_confusion():
    df = _read_matrix("confusion_part.csv")
    fig, ax = plt.subplots(figsize=(7.2, 6.1)); _heatmap(ax, df, "四部件主诊断逐类召回/共现矩阵")
    ax.text(0.01, -0.19, "注：对角为逐部件召回率；并发退化部件的正确共现按真值计入、不计为混淆。",
            transform=ax.transAxes, fontsize=9)
    return S.save(fig, os.path.join(FIGDIR, "图3-12_四部件主诊断混淆矩阵"))


def fig_part_metrics():
    d = pd.read_csv(PART_METRICS)
    fig, ax = plt.subplots(figsize=(9.6, 4.8)); x=np.arange(len(d)); w=.24
    ax.bar(x-w, d.precision, w, label="精确率", color="#6BAED6")
    ax.bar(x, d.recall, w, label="召回率", color="#31A354")
    ax.bar(x+w, d.F1, w, label="F1", color=S.D_COLOR)
    ax.set_xticks(x); ax.set_xticklabels(d.name); ax.set_ylim(.90, 1.015)
    ax.axhline(.95, color="#777", ls="--", lw=1); ax.set_ylabel("指标值")
    ax.set_title("四部件主诊断逐部件性能（5 seed 聚合）")
    ax.grid(axis="y", alpha=.35); ax.legend(ncol=3, loc="lower left")
    for i, v in enumerate(d.F1): ax.text(i+w, v+.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.text(.99, .05, f"macro-F1 = {d.F1.mean():.3f}", transform=ax.transAxes, ha="right",
            bbox=dict(fc="#FFF3E0", ec="#D9A55A", pad=4))
    return S.save(fig, os.path.join(FIGDIR, "图3-13_四部件逐类性能"))


def fig_latent():
    d = pd.read_csv(os.path.join(FD, "latent_tsne.csv"), encoding="utf-8-sig")
    order = ["正常"] + S.PART_ORDER
    colors = {"正常":"#9E9E9E", **S.PART_COLORS}
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for label in order:
        z=d[d.label==label]
        ax.scatter(z.tsne1, z.tsne2, s=15, alpha=.58, color=colors[label], label=label,
                   edgecolor="none", rasterized=True)
    ax.set_xlabel("t-SNE 维度 1"); ax.set_ylabel("t-SNE 维度 2")
    ax.set_title("DTAE 潜空间的类别聚集结构")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5,-.08), frameon=False)
    ax.grid(alpha=.18)
    return S.save(fig, os.path.join(FIGDIR, "图3-13_DTAE潜空间tSNE"))


def fig_timeline():
    d = pd.read_csv(os.path.join(FD, "timeline.csv"), encoding="utf-8-sig")
    fig = plt.figure(figsize=(11.5, 6.1)); gs=GridSpec(3,1,height_ratios=[2.1,.65,.9],hspace=.20,figure=fig)
    ax=fig.add_subplot(gs[0]);
    ax.plot(d.cycle,d.severity_true,color=S.TRUTH_COLOR,lw=2.3,label="真实严重度")
    ax.plot(d.cycle,d.severity_est,color=S.D_COLOR,lw=1.8,label="估计严重度")
    ax.fill_between(d.cycle,0,d.severity_est,color=S.D_COLOR,alpha=.08)
    ax.set_ylabel("量程归一化严重度"); ax.set_title("(a) 连续退化程度",loc="left"); ax.grid(alpha=.3); ax.legend()
    ax=fig.add_subplot(gs[1],sharex=ax)
    ax.step(d.cycle,d.detected,where="mid",color=S.B_COLOR,lw=1.8)
    ax.fill_between(d.cycle,0,d.detected,step="mid",color=S.B_COLOR,alpha=.22)
    ax.set_yticks([0,1]); ax.set_yticklabels(["正常","报警"]); ax.set_title("(b) 故障检测",loc="left")
    ax=fig.add_subplot(gs[2],sharex=ax)
    labels=list(dict.fromkeys(["正常"]+d.truth.tolist()+d.pred.tolist()))
    cmap=ListedColormap(["#F0F0F0"]+[S.PART_COLORS.get(x,"#756BB1") for x in labels[1:]])
    arr=np.vstack([[labels.index(x) for x in d.truth],[labels.index(x) for x in d.pred]])
    ax.imshow(arr,aspect="auto",interpolation="nearest",cmap=cmap,vmin=0,vmax=max(len(labels)-1,1),
              extent=[d.cycle.min(),d.cycle.max(),-.5,1.5])
    ax.set_yticks([1,0]); ax.set_yticklabels(["真实部件","预测部件"]); ax.set_xlabel("飞行循环")
    ax.set_title("(c) 部件隔离",loc="left")
    handles=[plt.Line2D([0],[0],color=cmap(i),lw=7,label=l) for i,l in enumerate(labels)]
    ax.legend(handles=handles,ncol=min(5,len(handles)),loc="upper center",bbox_to_anchor=(.5,-.38),frameon=False)
    fig.suptitle("代表发动机从退化出现到部件判定的诊断时间线",y=.99)
    return S.save(fig, os.path.join(FIGDIR, "图3-14_代表发动机诊断时间线"))


def fig_family_confusion():
    df=_read_matrix("confusion_family.csv")
    fig,ax=plt.subplots(figsize=(7.4,6.2)); _heatmap(ax,df,"五部件族细分逐类召回/共现矩阵")
    ax.text(.01,-.17,"注：对角为逐类召回；非对角仅涡轮内 HPT↔LPT 真实混叠（气路指纹近共线所致）。",
            transform=ax.transAxes,fontsize=9)
    return S.save(fig,os.path.join(FIGDIR,"图3-15_五部件族细分混淆矩阵"))


def fig_turbine_detail():
    m=pd.read_csv(FAMILY_METRICS).set_index("name"); c=_read_matrix("confusion_family.csv")
    fig=plt.figure(figsize=(10.8,4.7)); gs=GridSpec(1,2,width_ratios=[1.25,1],wspace=.34,figure=fig)
    ax=fig.add_subplot(gs[0]); names=["HPT","LPT"]; x=np.arange(2); w=.24
    for k,(metric,color,label) in enumerate([("precision","#6BAED6","精确率"),("recall","#31A354","召回率"),("F1",S.D_COLOR,"F1")]):
        ax.bar(x+(k-1)*w,m.loc[names,metric],w,color=color,label=label)
    ax.set_xticks(x);ax.set_xticklabels(names);ax.set_ylim(.60,1.02);ax.axhline(.85,color="#777",ls="--",lw=1)
    ax.set_ylabel("指标值");ax.set_title("(a) 涡轮内部细分性能",loc="left");ax.grid(axis="y",alpha=.35);ax.legend(ncol=3)
    ax=fig.add_subplot(gs[1]); vals=[c.loc["HPT","LPT"],c.loc["LPT","HPT"]]
    bars=ax.bar(["HPT→LPT","LPT→HPT"],vals,color=[S.FAMILY_COLORS["HPT"],S.FAMILY_COLORS["LPT"]])
    ax.set_ylim(0,max(vals)*1.45);ax.set_ylabel("跨类共现率");ax.set_title("(b) 涡轮内相互混叠",loc="left");ax.grid(axis="y",alpha=.35)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.006,f"{100*v:.1f}%",ha="center")
    ax.text(.5,.93,"混叠边界与影响矩阵近共线一致",transform=ax.transAxes,ha="center",fontsize=9)
    fig.suptitle("HPT/LPT 细分的可辨识性深度分析",y=1.01)
    return S.save(fig,os.path.join(FIGDIR,"图3-16_涡轮内部细分边界"))


def fig_granularity_gain():
    part=pd.read_csv(PART_METRICS).set_index("name"); fam=pd.read_csv(FAMILY_METRICS).set_index("name")
    labels=["风扇/Fan","高压压气机/HPC","低压压气机/LPC","涡轮"]
    fine=[fam.loc["Fan","F1"],fam.loc["HPC","F1"],fam.loc["LPC","F1"],fam.loc[["HPT","LPT"],"F1"].mean()]
    main=[part.loc["风扇","F1"],part.loc["高压压气机","F1"],part.loc["低压压气机","F1"],part.loc["涡轮","F1"]]
    fig,ax=plt.subplots(figsize=(9.8,4.9));x=np.arange(4);w=.34
    ax.bar(x-w/2,fine,w,color="#9ECAE1",label="五部件细分（涡轮项为HPT/LPT均值）")
    ax.bar(x+w/2,main,w,color=S.D_COLOR,label="四部件主诊断")
    ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylim(.72,1.02);ax.set_ylabel("F1")
    ax.set_title("诊断粒度与可辨识性匹配带来的性能收益")
    ax.grid(axis="y",alpha=.35);ax.legend(loc="lower left")
    macro_f=fam.F1.mean();macro_p=part.F1.mean()
    ax.text(.99,.06,f"macro-F1：{macro_f:.3f} → {macro_p:.3f}",transform=ax.transAxes,ha="right",
            bbox=dict(fc="#FFF3E0",ec="#D9A55A",pad=4))
    return S.save(fig,os.path.join(FIGDIR,"图3-18_四部件诊断粒度收益"))
