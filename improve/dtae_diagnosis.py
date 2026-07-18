"""基于 DTAE 的气路故障诊断（分布内 cycle 级协议，参考王昆博士论文第3章）。

协议（对应"对已知机队的在线诊断"，与王昆同机连续架次一致）：
  - 池化辨识/分布内 dev 发动机（每台单一故障族）的逐循环残差 R；
  - 每台发动机按循环分层 70/30 划分 train/test（同机、循环不重叠）；
  - 6 类单标签：正常 + HPT/Fan/HPC/LPT/LPC；
  - DTAE 联合重构+分类，加噪掩码增强；测试报告检测与隔离指标。
本协议评估对监视中发动机的诊断能力，不评估跨新发动机泛化（后者见连续估计闭环）。
"""
import os
import numpy as np
import pandas as pd

import closure_lib as C
from dtae import DTAE

FAMS = ["HPT", "Fan", "HPC", "LPT", "LPC"]
CLASSES = ["正常"] + FAMS
FAM_OF_PARAM = {0: 1, 1: 2, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5, 8: 5}  # ->类别(1..5)


def features(s):
    R = s["R"]
    sm = pd.DataFrame(R).rolling(3, min_periods=1).mean().values     # 轻时序
    return np.hstack([R, sm])


def cycle_label(s):
    """逐循环 6 类单标签（dev 单元每台单一故障族）。"""
    y = np.zeros(len(s["cycles"]), int)
    fam = FAM_OF_PARAM[int(s["fault_idx"][0])] if len(s["fault_idx"]) else 0
    y[~s["hs"]] = fam
    return y


def build_dataset(seed=0):
    data = C.load_sequences(lam=0.01)
    dev = data["calRaw"] + data["idRaw"]      # 10 台 dev 发动机（每台单故障族）
    rng = np.random.RandomState(seed)
    Xtr, ytr, Xte, yte, ute = [], [], [], [], []
    for k, s in enumerate(dev):
        X = features(s); y = cycle_label(s); n = len(y)
        idx = rng.permutation(n); cut = int(0.7 * n)
        tr, te = idx[:cut], idx[cut:]
        Xtr.append(X[tr]); ytr.append(y[tr])
        Xte.append(X[te]); yte.append(y[te]); ute.append(np.full(len(te), k))
    return (np.vstack(Xtr), np.concatenate(ytr),
            np.vstack(Xte), np.concatenate(yte), np.concatenate(ute))


def onehot(y, k=6):
    Y = np.zeros((len(y), k)); Y[np.arange(len(y)), y] = 1
    return Y


def confusion(yt, yp, k=6):
    M = np.zeros((k, k), int)
    for a, b in zip(yt, yp):
        M[a, b] += 1
    return M


def metrics(yt, yp):
    det_t = yt > 0; det_p = yp > 0
    tp = np.sum(det_p & det_t); tn = np.sum(~det_p & ~det_t)
    fp = np.sum(det_p & ~det_t); fn = np.sum(~det_p & det_t)
    FAR = fp / max(fp + tn, 1); DR = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1); detF1 = 2 * prec * DR / max(prec + DR, 1e-9)
    f1s = []
    for c in range(1, 6):
        t = yt == c; p = yp == c
        if t.any() or p.any():
            tpc = np.sum(p & t); fpc = np.sum(p & ~t); fnc = np.sum(~p & t)
            pr = tpc / max(tpc + fpc, 1); rc = tpc / max(tpc + fnc, 1)
            f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
    return dict(FAR=FAR, DR=DR, detF1=detF1, macroF1=float(np.mean(f1s)),
                acc=np.mean(yt == yp))


def run(seed=0, verbose=True):
    Xtr, ytr, Xte, yte, ute = build_dataset(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    net = DTAE(Xtr.shape[1], d_latent=12, n_class=6, lambda_cls=2.0,
               noise=0.3, mask=0.1, lr=3e-3, epochs=400, batch=128, seed=seed)
    net.fit(Xtr, onehot(ytr))
    proba = net.predict_proba(Xte); yp = np.argmax(proba, axis=1)
    m = metrics(yte, yp); M = confusion(yte, yp)
    if verbose:
        print(f"[seed {seed}] 检测 FAR={m['FAR']:.3f} DR={m['DR']:.3f} F1={m['detF1']:.3f} "
              f"| 隔离 macroF1={m['macroF1']:.3f} 六类acc={m['acc']:.3f}")
    return dict(net=net, mu=mu, sd=sd, Xte=Xte, yte=yte, yp=yp, ute=ute,
                metrics=m, confusion=M)


if __name__ == "__main__":
    accs = []
    for sd in range(5):
        r = run(sd)
        accs.append([r["metrics"]["FAR"], r["metrics"]["DR"], r["metrics"]["detF1"],
                     r["metrics"]["macroF1"]])
    a = np.array(accs)
    print("\n5 seed 均值±std: FAR=%.3f±%.3f DR=%.3f±%.3f detF1=%.3f±%.3f macroF1=%.3f±%.3f"
          % (a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
             a[:, 2].mean(), a[:, 2].std(), a[:, 3].mean(), a[:, 3].std()))
