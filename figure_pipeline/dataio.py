"""结果数据读取与部件族聚合。

所有图件的唯一数据来源是 MATLAB 闭环实验落盘的锁定 CSV，本模块只做读取与
确定性聚合（九参数→五部件族），不重新拟合、不重新选参、不触碰测试真值做选择。
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "matlab", "outputs")
T4 = os.path.join(OUT, "t4_diagnostic_closure")
T5 = os.path.join(OUT, "t5_robustness_ablation")
T6 = os.path.join(OUT, "t6_repeated_splits")

# 九参数(1..9) → 五部件族 冻结映射
FAMILY_OF_PARAM = {1: "HPT", 2: "Fan", 3: "Fan", 4: "HPC", 5: "HPC",
                   6: "LPT", 7: "LPT", 8: "LPC", 9: "LPC"}
FAMILY_ORDER = ["HPT", "Fan", "HPC", "LPT", "LPC"]


def raw_cycles():
    """逐参数逐循环的诊断闭环原始结果。"""
    df = pd.read_csv(os.path.join(T4, "diagnostic_cycle_raw.csv"))
    df["family"] = df["component"].map(FAMILY_OF_PARAM)
    return df


def family_cycle_labels(df=None):
    """把九参数多标签逻辑或聚合为五部件族的逐(engine,cycle)真值/预测标签。

    返回：每行一个 (regime, subset, unit, cycle) × family 的 0/1 真值与预测。
    """
    if df is None:
        df = raw_cycles()
    g = (df.groupby(["regime", "subset", "unit", "cycle", "family"])
           .agg(truth=("truth_label", "max"),
                pred=("pred_label", "max"))
           .reset_index())
    return g


def family_confusion(regime, df=None):
    """五部件族多标签混淆式统计：真=行族、预测=列族的共现计数。

    多标签任务无单一互斥混淆矩阵，这里给出“真值为族 i 的循环中被预测为族 j 的比例”，
    对角为逐族召回，非对角揭示族间混叠；额外给出健康行/列。
    """
    fam = family_cycle_labels(df)
    fam = fam[fam["regime"] == regime]
    labels = FAMILY_ORDER
    # 每个 (unit,cycle) 的真值族集合与预测族集合
    key = ["subset", "unit", "cycle"]
    piv_t = fam.pivot_table(index=key, columns="family", values="truth", fill_value=0)
    piv_p = fam.pivot_table(index=key, columns="family", values="pred", fill_value=0)
    for L in labels:
        if L not in piv_t:
            piv_t[L] = 0
        if L not in piv_p:
            piv_p[L] = 0
    piv_t, piv_p = piv_t[labels], piv_p[labels]
    return piv_t, piv_p


def family_metrics(regime, df=None):
    """逐部件族 precision/recall/F1/FAR + macro/micro F1 + Hamming loss。"""
    piv_t, piv_p = family_confusion(regime, df)
    rows = []
    T, P = piv_t.values, piv_p.values
    for j, L in enumerate(FAMILY_ORDER):
        t, p = T[:, j], P[:, j]
        tp = int(np.sum((t == 1) & (p == 1)))
        fp = int(np.sum((t == 0) & (p == 1)))
        fn = int(np.sum((t == 1) & (p == 0)))
        tn = int(np.sum((t == 0) & (p == 0)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        far = fp / (fp + tn) if fp + tn else 0.0
        rows.append(dict(family=L, precision=prec, recall=rec, F1=f1, FAR=far,
                         support=int(np.sum(t))))
    m = pd.DataFrame(rows)
    macro_f1 = m["F1"].mean()
    micro_tp = np.sum((T == 1) & (P == 1))
    micro_fp = np.sum((T == 0) & (P == 1))
    micro_fn = np.sum((T == 1) & (P == 0))
    micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0
    micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0
    hamming = np.mean(T != P)
    return m, dict(macro_f1=macro_f1, micro_f1=micro_f1, hamming=hamming)


def detection_confusion(regime):
    d = pd.read_csv(os.path.join(T4, "detection_confusion.csv"))
    d = d[d["regime"] == regime]
    M = np.zeros((2, 2), dtype=int)  # 行=真值, 列=预测
    for _, r in d.iterrows():
        M[int(r["truth"]), int(r["prediction"])] = int(r["count"])
    return M


def stage_confusion(regime):
    d = pd.read_csv(os.path.join(T4, "stage_confusion.csv"))
    d = d[d["regime"] == regime]
    M = np.zeros((4, 4), dtype=int)
    for _, r in d.iterrows():
        M[int(r["truth"]), int(r["prediction"])] = int(r["count"])
    return M
