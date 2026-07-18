"""基于 DTAE 的气路故障诊断（分布内、多标签五部件族、两阶段）——参考王昆博士论文第3章。

协议（"对已知机队的在线诊断"，与王昆同机连续架次同 scope；不评估跨新发动机泛化）：
  - 特征：标准化气路残差 R（13）+ 多尺度时序（滚动均值/退化斜率）+ 连续估计部件严重度 θ̂
    及其时序斜率（物理特征，利用退化时间动态突破部件线性共线）；
  - 网络：双任务自编码器（重构+分类联合，加噪+掩码增强），numpy 实现见 dtae.py；
  - 标签：五部件族多标签（HPT/Fan/HPC/LPT/LPC；允许多族同时退化，如 DS06=HPC+LPC）；
  - 划分：池化 dev 发动机逐循环，每台按循环 70/30；报告 5 个随机划分种子聚合结果；
  - 两阶段：先检测（任一族 + 持续 K 循环）再对故障样本隔离，与王昆一致。
指标：事件级检测率、首次检测延迟、健康期虚警；隔离逐族 P/R/F1、macro/micro-F1、混淆。
"""
import os
import numpy as np
import pandas as pd
import closure_lib as C
from dtae import DTAE

# 主诊断层级：四部件（涡轮 = 高低压涡轮合并，工程维修单元；共线的 HPT/LPT 内部细分见 family 层）
PART_NAMES = ["风扇", "高压压气机", "低压压气机", "涡轮"]
PART_OF = {0: 3, 1: 0, 2: 0, 3: 1, 4: 1, 5: 3, 6: 3, 7: 2, 8: 2}
# 细分层级：五部件族（用于涡轮内 HPT vs LPT 可辨识性深度分析）
FAMS = ["HPT", "Fan", "HPC", "LPT", "LPC"]
FAM_OF = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}
LEVELS = {"part": (PART_NAMES, PART_OF), "family": (FAMS, FAM_OF)}
DIAG_SEV = 0.05        # 预注册：真值量程归一化严重度 > 0.05 视为物理可观测退化
PERSIST = 3            # 检测持续循环数
SEEDS = range(5)
OUT = os.path.dirname(__file__)


def features(s):
    """残差 R + 多尺度时序 + 退化加速度 + 连续估计 θ̂ 动态 + 高低压涡轮物理区分特征。"""
    R = s["R"]; df = pd.DataFrame(R)
    m5 = df.rolling(5, min_periods=1).mean().values
    m10 = df.rolling(10, min_periods=1).mean().values
    m15 = df.rolling(15, min_periods=1).mean().values
    slope = (df - df.shift(8)).fillna(0).values
    accel = (df - 2 * df.shift(4) + df.shift(8)).fillna(0).values       # 退化加速度
    sev = np.maximum(0, -s["theta_hat"])
    sdf = pd.DataFrame(sev); ssl = (sdf - sdf.shift(8)).fillna(0).values
    # 高压涡轮出口 T48(idx5) 与低压涡轮出口 T50(idx6) 的差/比，区分 HPT vs LPT
    hpvlp = np.column_stack([R[:, 5] - R[:, 6], R[:, 5] / (np.abs(R[:, 6]) + 0.5)])
    return np.hstack([R, m5, m10, m15, slope, accel, sev, ssl, hpvlp])


def fam_multi(s, fam_of, k):
    T = len(s["cycles"]); Y = np.zeros((T, k), bool)
    for j in s["fault_idx"]:
        Y[~s["hs"], fam_of[int(j)]] = True
    return Y


def fam_sev(s, fam_of, k):
    st = np.maximum(0, -s["theta_true"]); S = np.zeros((st.shape[0], k))
    for j in range(9):
        S[:, fam_of[j]] = np.maximum(S[:, fam_of[j]], st[:, j])
    return S


def train_split(seed, fam_of, k):
    data = C.load_sequences(lam=0.01); dev = data["calRaw"] + data["idRaw"]
    rng = np.random.RandomState(seed)
    Xtr, Ytr, te = [], [], []
    for s in dev:
        X = features(s); Y = fam_multi(s, fam_of, k); n = X.shape[0]
        idx = rng.permutation(n); cut = int(0.7 * n)
        Xtr.append(X[idx[:cut]]); Ytr.append(Y[idx[:cut]])
        te.append((s, X, Y, fam_sev(s, fam_of, k), np.sort(idx[cut:])))
    Xtr = np.vstack(Xtr); Ytr = np.vstack(Ytr).astype(float)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    net = DTAE((Xtr - mu).shape[1], 24, k, lambda_cls=3.0, noise=0.25, mask=0.1,
               lr=3e-3, epochs=700, batch=128, seed=seed).fit((Xtr - mu) / sd, Ytr)
    return net, mu, sd, te


def evaluate(level="part", collect_confusion=False):
    names, fam_of = LEVELS[level]; k = len(names)
    tp = np.zeros(k); fp = np.zeros(k); fn = np.zeros(k)
    evDR = []; delays = []; far_cyc = []
    conf = np.zeros((k, k))         # 隔离共现：真值类 × 预测类（故障可观测样本）
    latent_Z = []; latent_y = []
    for seed in SEEDS:
        net, mu, sd, te = train_split(seed, fam_of, k)
        for (s, X, Y, S, idx) in te:
            Xte = (X[idx] - mu) / sd
            P = net.predict(Xte); Yt = Y[idx]; St = S[idx]; hs = s["hs"][idx]
            fault = ~hs
            # 两阶段检测：任一族持续 PERSIST
            det = P.any(1).astype(int); run = 0; first = None
            for t in range(len(det)):
                run = run + 1 if det[t] else 0
                if run >= PERSIST and first is None:
                    first = t
            if fault.any():
                evDR.append(1.0 if first is not None else 0.0)
            if hs.any():
                far_cyc.append(np.mean(P[hs].any(1)))
            # 隔离：故障且可观测样本上逐类
            for g in range(k):
                obs = fault & (St[:, g] > DIAG_SEV)   # 该类真实退化且可观测
                # 负样本 = 其它族可观测退化的故障样本（隔离难点=族间区分）
                other = fault & (~obs) & (St.max(1) > DIAG_SEV)
                t_pos = P[obs, g]; t_neg = P[other, g]
                tp[g] += np.sum(t_pos); fn[g] += np.sum(~t_pos); fp[g] += np.sum(t_neg)
            if collect_confusion:
                diag = fault & (St.max(1) > DIAG_SEV)
                for i in np.where(diag)[0]:
                    truth_fams = np.where(Yt[i])[0]
                    pred_fams = np.where(P[i])[0]
                    for a in truth_fams:
                        if len(pred_fams):
                            for b in pred_fams:
                                conf[a, b] += 1 / len(pred_fams)
                        else:
                            conf[a, a] += 0
                latent_Z.append(net.encode(Xte[diag])); latent_y.append(Yt[diag])
    f1 = np.array([2 * tp[g] / max(2 * tp[g] + fp[g] + fn[g], 1) for g in range(k)])
    prec = np.array([tp[g] / max(tp[g] + fp[g], 1) for g in range(k)])
    rec = np.array([tp[g] / max(tp[g] + fn[g], 1) for g in range(k)])
    res = dict(names=names, level=level, event_DR=np.mean(evDR), healthy_far=np.mean(far_cyc),
               f1=f1, precision=prec, recall=rec, macro_f1=float(f1.mean()),
               micro_f1=2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1))
    if collect_confusion:
        res["confusion"] = conf
        res["Z"] = np.vstack(latent_Z); res["y"] = np.vstack(latent_y)
    return res


if __name__ == "__main__":
    for level in ["part", "family"]:
        r = evaluate(level)
        tag = "四部件主诊断" if level == "part" else "五部件族细分(涡轮内HPT/LPT)"
        print(f"\n== {tag} ==")
        print(f"事件级检测率 = {r['event_DR']:.3f}   健康期虚警(cycle) = {r['healthy_far']:.3f}")
        for g, nm in enumerate(r["names"]):
            print(f"  {nm:6}  P={r['precision'][g]:.3f} R={r['recall'][g]:.3f} F1={r['f1'][g]:.3f}")
        print(f"  >>> macro-F1 = {r['macro_f1']:.3f}   micro-F1 = {r['micro_f1']:.3f}")
        pd.DataFrame(dict(name=r["names"], precision=r["precision"], recall=r["recall"],
                          F1=r["f1"])).to_csv(
            os.path.join(OUT, f"dtae_metrics_{level}.csv"), index=False)
