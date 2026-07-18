"""双任务自编码器（DTAE）—— 参考王昆博士论文第3章的气路故障诊断方法。

在第2章健康基准残差（本章为物理约束反演前的标准化气路残差 R）之上，联合优化
重构任务与分类任务，并在编码器输入端加入加噪与掩码扰动，学习既保留残差信息、
又具判别性的鲁棒潜特征，用于故障检测（正常/故障二分类）与部件族隔离（五类）。

实现为单隐层双任务网络（numpy 手写前向/反向 + Adam）：
    编码器  z = tanh(x W_e + b_e)
    解码器  x_hat = z W_d + b_d           （重构，MSE 对齐干净残差）
    分类头  logit = z W_c + b_c            （多标签，sigmoid + BCE）
    损失    L = MSE(x_hat, x_clean) + lambda_cls * BCE(sigmoid(logit), y)
训练时输入为加噪+掩码的破坏样本，重构目标为干净残差，迫使编码器从受扰输入中
提取稳定判别特征（去噪自编码器 + 监督分类的联合形式）。
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class DTAE:
    def __init__(self, d_in, d_latent=10, n_class=5, lambda_cls=1.0,
                 noise=0.3, mask=0.1, lr=3e-3, epochs=300, batch=128, seed=0):
        rng = np.random.RandomState(seed)
        s = lambda a, b: rng.randn(a, b) * np.sqrt(2.0 / a)
        self.We = s(d_in, d_latent); self.be = np.zeros(d_latent)
        self.Wd = s(d_latent, d_in); self.bd = np.zeros(d_in)
        self.Wc = s(d_latent, n_class); self.bc = np.zeros(n_class)
        self.cfg = dict(lambda_cls=lambda_cls, noise=noise, mask=mask,
                        lr=lr, epochs=epochs, batch=batch)
        self.rng = rng
        self._init_adam()

    def _init_adam(self):
        self._m = {k: np.zeros_like(getattr(self, k)) for k in
                   ["We", "be", "Wd", "bd", "Wc", "bc"]}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._m}
        self._t = 0

    def _adam(self, grads):
        self._t += 1; c = self.cfg; b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - c["lr"] * mh / (np.sqrt(vh) + eps))

    def encode(self, x):
        return np.tanh(x @ self.We + self.be)

    def _corrupt(self, x):
        c = self.cfg
        xa = x + c["noise"] * self.rng.randn(*x.shape)
        m = self.rng.rand(*x.shape) < c["mask"]
        xa = xa.copy(); xa[m] = 0.0
        return xa

    def fit(self, X, Y, Xval=None, Yval=None, verbose=False):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        n, c = X.shape[0], self.cfg
        for ep in range(c["epochs"]):
            idx = self.rng.permutation(n)
            for st in range(0, n, c["batch"]):
                b = idx[st:st + c["batch"]]
                xc = X[b]; xin = self._corrupt(xc); y = Y[b]; nb = len(b)
                # 前向
                z = np.tanh(xin @ self.We + self.be)
                xhat = z @ self.Wd + self.bd
                logit = z @ self.Wc + self.bc; p = _sigmoid(logit)
                # 反向
                dxhat = 2.0 * (xhat - xc) / nb                     # 重构 MSE
                gWd = z.T @ dxhat; gbd = dxhat.sum(0)
                dlogit = c["lambda_cls"] * (p - y) / nb            # 分类 BCE
                gWc = z.T @ dlogit; gbc = dlogit.sum(0)
                dz = dxhat @ self.Wd.T + dlogit @ self.Wc.T
                dpre = dz * (1 - z ** 2)                           # tanh'
                gWe = xin.T @ dpre; gbe = dpre.sum(0)
                self._adam(dict(We=gWe, be=gbe, Wd=gWd, bd=gbd, Wc=gWc, bc=gbc))
            if verbose and Xval is not None and ep % 50 == 0:
                print(f"  epoch {ep}: val_macroF1={self.macro_f1(Xval, Yval):.3f}")
        return self

    def predict_proba(self, X):
        return _sigmoid(self.encode(np.asarray(X, float)) @ self.Wc + self.bc)

    def predict(self, X, thr=0.5):
        return self.predict_proba(X) >= thr

    def macro_f1(self, X, Y):
        P = self.predict(X); Y = np.asarray(Y, bool)
        f = []
        for g in range(Y.shape[1]):
            if Y[:, g].any() or P[:, g].any():
                tp = np.sum(P[:, g] & Y[:, g]); fp = np.sum(P[:, g] & ~Y[:, g])
                fn = np.sum(~P[:, g] & Y[:, g])
                pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                f.append(2 * pr * rc / max(pr + rc, 1e-9))
        return float(np.mean(f)) if f else 1.0
