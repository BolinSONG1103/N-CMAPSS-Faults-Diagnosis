"""导出 DTAE 诊断的全部图数据为 CSV（供本地 MATLAB / R 出王昆级图）。

产出（均在 improve/figdata/）：
  detection_confusion.csv   检测 2×2 混淆（真值×预测，计数）
  confusion_part.csv        四部件隔离混淆（真值类×预测类，故障可观测样本，多seed聚合）
  confusion_family.csv      五部件族细分混淆
  metrics_part.csv/family   逐类 P/R/F1（已由 dtae_diagnosis 落盘，此处复算含检测汇总）
  latent_tsne.csv           DTAE 潜空间 t-SNE 二维坐标 + 类别标签
  timeline.csv              代表发动机逐循环：真值/预测部件、退化程度、检测标志
"""
import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import closure_lib as C
import dtae_diagnosis as DD
from dtae import DTAE

FD = os.path.join(os.path.dirname(__file__), "figdata")
os.makedirs(FD, exist_ok=True)
DIAG = DD.DIAG_SEV


def _train(level, seed):
    names, fam_of = DD.LEVELS[level]; k = len(names)
    return DD.train_split(seed, fam_of, k) + (names, fam_of, k)


def export_confusion(level):
    names, fam_of = DD.LEVELS[level]; k = len(names)
    hit = np.zeros((k, k)); rowN = np.zeros(k); detM = np.zeros((2, 2))
    for seed in DD.SEEDS:
        net, mu, sd, te = DD.train_split(seed, fam_of, k)
        for (s, X, Y, S, idx) in te:
            P = net.predict((X[idx] - mu) / sd); Yt = Y[idx]; St = S[idx]; hs = s["hs"][idx]
            fault = ~hs
            # 检测 2×2
            dp = P.any(1)
            for a, b in zip(fault.astype(int), dp.astype(int)):
                detM[a, b] += 1
            # 隔离逐类召回混淆（多标签命中：对角=召回，非对角=并发/误判比例）
            diag = fault & (St.max(1) > DIAG)
            for i in np.where(diag)[0]:
                for a in np.where(Yt[i])[0]:        # 每个真实退化类
                    rowN[a] += 1
                    for b in np.where(P[i])[0]:     # 预测包含的类
                        hit[a, b] += 1
    conf = hit / np.where(rowN[:, None] == 0, 1, rowN[:, None])   # 行归一化=逐类召回
    pd.DataFrame(conf, index=names, columns=names).to_csv(
        os.path.join(FD, f"confusion_{level}.csv"), encoding="utf-8-sig")
    if level == "part":
        pd.DataFrame(detM, index=["正常", "故障"], columns=["正常", "故障"]).to_csv(
            os.path.join(FD, "detection_confusion.csv"), encoding="utf-8-sig")
    print(f"[ok] confusion_{level}.csv")


def export_latent():
    net, mu, sd, te = DD.train_split(1, DD.PART_OF, 4)
    Z = []; y = []
    for (s, X, Y, S, idx) in te:
        diag = (~s["hs"][idx]) | (S[idx].max(1) <= 1e-9)  # 故障+健康都取
        Xte = (X[idx] - mu) / sd
        Z.append(net.encode(Xte));
        lab = np.where(Y[idx].any(1), Y[idx].argmax(1) + 1, 0)  # 0=正常,1..4部件
        y.append(lab)
    Z = np.vstack(Z); y = np.concatenate(y)
    # 抽样上限，t-SNE 稳定
    if len(Z) > 1500:
        pick = np.random.RandomState(0).choice(len(Z), 1500, replace=False); Z = Z[pick]; y = y[pick]
    emb = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
    names = ["正常"] + DD.PART_NAMES
    pd.DataFrame(dict(tsne1=emb[:, 0], tsne2=emb[:, 1],
                      label=[names[i] for i in y])).to_csv(
        os.path.join(FD, "latent_tsne.csv"), index=False, encoding="utf-8-sig")
    print("[ok] latent_tsne.csv")


def export_timeline(subset="DS05_id", unit=2):
    """代表发动机逐循环诊断时间线（part 级）。"""
    net, mu, sd, te = DD.train_split(1, DD.PART_OF, 4)
    data = C.load_sequences(lam=0.01); dev = data["calRaw"] + data["idRaw"]
    s = [x for x in dev if x["subset"] == subset and x["unit"] == unit][0]
    X = DD.features(s); P = net.predict((X - mu) / sd)
    Y = DD.fam_multi(s, DD.PART_OF, 4)
    sev = np.max(np.maximum(0, -s["theta_hat"]), 1)         # 总体退化程度(估计)
    sev_t = np.max(np.maximum(0, -s["theta_true"]), 1)
    part_names = DD.PART_NAMES
    truth_part = ["".join(part_names[g] for g in np.where(Y[t])[0]) or "正常"
                  for t in range(len(s["cycles"]))]
    pred_part = ["".join(part_names[g] for g in np.where(P[t])[0]) or "正常"
                 for t in range(len(s["cycles"]))]
    pd.DataFrame(dict(cycle=s["cycles"], severity_est=sev, severity_true=sev_t,
                      truth=truth_part, pred=pred_part,
                      detected=P.any(1).astype(int))).to_csv(
        os.path.join(FD, "timeline.csv"), index=False, encoding="utf-8-sig")
    print(f"[ok] timeline.csv ({subset} u{unit})")


if __name__ == "__main__":
    export_confusion("part")
    export_confusion("family")
    export_latent()
    export_timeline()
    # 汇总检测指标
    rp = DD.evaluate("part")
    pd.DataFrame(dict(metric=["event_detection_rate", "healthy_cycle_far",
                              "macro_f1", "micro_f1"],
                      value=[rp["event_DR"], rp["healthy_far"],
                             rp["macro_f1"], rp["micro_f1"]])).to_csv(
        os.path.join(FD, "detection_summary.csv"), index=False, encoding="utf-8-sig")
    print("[ok] detection_summary.csv")
