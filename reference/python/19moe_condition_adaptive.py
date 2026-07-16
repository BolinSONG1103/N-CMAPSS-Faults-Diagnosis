# -*- coding: utf-8 -*-
"""
================================================================================
脚本 19：工况自适应影响系数矩阵  H(w) = H0 · diag(g(w))
          —— 用 Mixture-of-Experts (MoE) 门控实现 g(w)
================================================================================

【这个脚本要解决的公认问题】

  经典 GPA (Urban 1972) 的影响系数矩阵 H 是在【单一工作点】线性化得到的。
  发动机实际在整个飞行包线里工作 (爬升/巡航/下降, TRA 从 20 到 100),
  用一个固定的 H 去反演所有工况, 是这一族方法五十年来的公认局限。

【你自己的数据已经指出了正确答案（⓪-d 实验）】

  按 TRA 四分位分档, 各档单独辨识 H, 再和全局指纹比:

    ✅ 方向【稳】: 所有列、所有档, 与全局指纹的余弦【最差 0.9865】
    ⚠️ 幅值【变】: 指纹范数最大波动【2.12 倍】, 且【随 TRA 单调递增】
         fan_flow  0.59 -> 1.24
         LPT_eff   0.66 -> 1.26
         fan_eff   0.66 -> 1.21
         HPT_eff   0.91 -> 1.10

  物理解释: 功率越大 -> 部件负荷越重 -> 同样的效率损失产生的绝对偏差越大。

  ⇒ 所以正确的结构是:  H(w) = H0 · diag(g(w))
       H0    : 数据驱动辨识出的【物理指纹方向】, 【冻结, 不训练】
       g(w)  : 9 个【正】标量增益, 随工况变化, 只改幅值不改方向

--------------------------------------------------------------------------------
【为什么用 MoE, 而不是随便一个 MLP】

  你手工做的"TRA 四分位分档"本质上就是一个【硬门控的 MoE】:
      低TRA档 -> 用增益向量 s_1
      中低档  -> 用 s_2  ...  (硬切换)

  MoE 只是把它变成【软门控】:
      g(w) = Σ_k α_k(w) · s_k        α = softmax(gate(w)),  Σα_k = 1
      s_k  = softplus(raw_k) ∈ R^9_+  (第 k 个专家的 9 维增益向量, 恒正)

  等价地:   H(w) = Σ_k α_k(w) · [ H0 · diag(s_k) ]
                            └──────── 第 k 个专家 ────────┘
  这就是标准的 MoE: K 个专家 + 一个门控网络。

  ★ 它不是硬塞进来的关键词, 它是你物理发现的【自然数学形式】。

  MoE 相对自由 MLP 的四个好处 (都可以写进论文):
    1. 可解释: 训完把 α_k(w) 画出来。如果门控自动学出了 TRA 的分区,
               ⇒ MoE 自主发现了你手工分档的物理规律。这是极强的一张图。
    2. 有消融: K=1 就退化回全局 H0 (= 现在的方法), K=4 对应你的手工分档。
               K 的消融曲线是白送的实验。
    3. 参数受约束: g(w) 被限制在 K 个专家的【凸包】内, 不能任意乱跑,
                   比自由 MLP 更难过拟合。
    4. 有对照: 本脚本同时跑 FreeMLP 臂, 如果 MoE 打不过它, 如实报告。

--------------------------------------------------------------------------------
【关键的数学事实 —— 这是 MoE 能无缝接进现有 NNLS 反演的原因】

  诊断时, 每个 cycle 的残差是 N 个样本的【平均】。
  每个样本的工况 w_i 不同, 所以 g(w_i) 不同。看起来很麻烦。

  但是: cycle 内 θ 是【常数】(健康参数不会在一次飞行内跳变),
        而模型对 θ 是【线性】的。所以:

      r̄_t = (1/N) Σ_i H0·diag(g(w_i))·θ_t
          = H0 · diag( (1/N) Σ_i g(w_i) ) · θ_t
          = H0 · diag( ḡ_t ) · θ_t

  ⇒ 窗口平均后的等效矩阵就是 H0·diag(ḡ_t), 其中 ḡ_t 是【增益的样本均值】。
  ⇒ 这是【精确的】, 不是近似。
  ⇒ 于是反演时只需把每个 cycle 的 Hn 换成 Hn·diag(ḡ_t), 其余一切不变。

  ⚠️ 注意 diag 可交换:  H·diag(g)·diag(span) = H·diag(span)·diag(g) = Hn·diag(g)
     所以量程归一化和工况增益不打架。

--------------------------------------------------------------------------------
【训练协议 —— 训练的是"正问题", 不是"反问题"】

  训练目标:   min_θ_gate  || Hn·diag(g(w))·θ_n(真值) + b  −  r ||²

  也就是: 给定【真实的】θ 和工况 w, 预测残差 r。这是【正向模型拟合】。
  这和 H0 本身的辨识方式完全一致 (脚本11: 把残差回归到真值 θ 上)。

  ★ 为什么不直接端到端训"反问题"(r -> θ̂)?
     因为那会让门控去补偿反演器的缺陷, 门控就不再是"物理增益"了,
     可解释性直接归零, 而且会和脚本18 的 λ 选择纠缠在一起。
     正向训练让 g(w) 保持【纯粹的物理含义】: 同样的退化, 在这个工况下
     会产生多大的残差。

  训练集: DS01/DS04/DS05/DS06/DS07 (辨识用的 5 个子集, 单/双部件故障)
  验证集: 上述子集中【留出的发动机单元】(按 unit 切分, 不按样本随机切)
  测试集: DS02/DS03  ← 【三部件同时退化 + 从未见过的发动机】, 双重外推

--------------------------------------------------------------------------------
【预注册判据（跑之前写死，跑完照着对，不许改）】

  Q1【前向有效性】
      在 DS02/DS03 上, MoE 的前向重构 NRMSE 相对 H0 降低 >= 5% (两个子集都要)。
      → 不成立: 工况自适应在前向上就没用, 后面不用看了。

  Q2【诊断的逐点占优】★ 核心, 延续脚本18 的纪律
      在所有 (子集 × N × λ) 网格点上, MoE 反演的技能分 > H0 反演的比例 >= 95%。
      → 成立则可写: "工况自适应的增益在整个正则化路径上一致, 不由 λ 的选择产生。"

  Q3【H0 冻结的合法性】★ 最诚实、最容易打脸的一条
      MoE_dir (允许 H0 被扰动) 相对 MoE (H0 冻结) 的技能分增益 < 0.01 (两个子集)。
      → 成立: "冻结方向"这个假设被证成, 可以理直气壮地写。
      → 不成立: 必须如实报告【方向也需要自适应】, 并把 cos>=0.9865 那个
                 证据的解释力下调。⚠️ 不要因为不好写就不做这个臂。

  Q4【MoE 结构 vs 自由 MLP】
      MoE(K=K_MAIN) 的技能分 >= FreeMLP − 0.005 (两个子集)。
      → 成立: 专家结构【没有损失精度】, 但换来了可解释性 (α_k 的物理分区)。
      → 不成立: 如实报告自由 MLP 更强, 把 MoE 降级为"可解释的近似"。

  Q5【门控的物理性】(描述性, 不作通过/失败判定)
      画出各专家的平均 TRA、增益随 TRA 的变化, 和 ⓪-d 的手工分档结果对比。
      如果学出来的增益也随 TRA 单调递增 -> 极强的一张图。

  ⚠️ 全零哨兵 (Z_zero) 必须在每个表里。任何方法跟哨兵打平, 那个指标就是坏的。

--------------------------------------------------------------------------------
运行:  python 19_moe_condition_adaptive_H.py
       先把 FAST_MODE 设 True 跑通流程 (几分钟), 再改 False 跑正式版。
依赖:  h5py numpy pandas scipy scikit-learn matplotlib torch
================================================================================
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ⚠️ 字体：SimHei 没有 U+2212(真减号) 的字形 -> 图上会出现 "10¤1" 这种乱码。
#    axes.unicode_minus=False 让 matplotlib 改用 ASCII 的 '-'。
#    (这就是脚本18 图1 下排负半轴乱码的原因, 这里一并修掉)
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit(
        "需要 PyTorch。请先安装:\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "(CPU 版就够了, 这个网络只有几千个参数)"
    )


# ============================================================================
# 配置 —— 前 6 项必须与脚本 16/17/18 【完全一致】, 否则结果不可比
# ============================================================================
DATA_DIR = "D:/N-CMPASS/data_set".strip()
OUT_DIR  = "./moe_out"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

FAST_MODE = True           # ★ 先 True 跑通流程, 再改 False 跑正式版

IDENT_FILES = {
    "N-CMAPSS_DS01-005.h5": ["HPT_eff_mod"],
    "N-CMAPSS_DS04.h5":     ["fan_eff_mod", "fan_flow_mod"],
    "N-CMAPSS_DS05.h5":     ["HPC_eff_mod", "HPC_flow_mod"],
    "N-CMAPSS_DS07.h5":     ["LPT_eff_mod", "LPT_flow_mod"],
    "N-CMAPSS_DS06.h5":     ["LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod"],
}
N_HEALTHY_ID, N_DEGRADED_ID = 60000, 150000

VALID_FILES = {
    "DS02": ("N-CMAPSS_DS02-006.h5", ["HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod"]),
    "DS03": ("N-CMAPSS_DS03-012.h5", ["HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod"]),
}

THETA_SPAN = {
    "fan_eff_mod":  0.223446, "fan_flow_mod": 0.121209, "LPC_eff_mod":  0.118767,
    "HPC_flow_mod": 0.070658, "LPC_flow_mod": 0.046187, "LPT_eff_mod":  0.036194,
    "LPT_flow_mod": 0.032908, "HPC_eff_mod":  0.025540, "HPT_eff_mod":  0.018668,
}
THETA9 = ["HPT_eff_mod", "fan_eff_mod", "fan_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
          "LPT_eff_mod", "LPT_flow_mod", "LPC_eff_mod", "LPC_flow_mod"]

# ★ 门控的输入 = 第2章健康基准 g(w) 的自变量, 【完全一致】。
#   如果这里多加一个变量(比如 Nf), 就等于开了一条"基准没吃掉、门控吃到了"
#   的信息泄漏通道, 会污染"残差已经把工况影响去除干净"这个前提。
OP_COLS = ["TRA", "Mach", "theta_c", "delta_c"]

WINDOWS      = [1, 10, 100, 1000] if not FAST_MODE else [100, 1000]
MAX_PER_CYCLE = 1000
N_REF_CYCLES  = 3

# λ 网格。脚本18 已经查明: oracle λ 在所有条件下都收敛到 ~31.6,
# 且量程归一化把"每个参数一个 λ"的需求消掉了。
# 这里【不重新研究 λ】, 而是在整条路径上做逐点占优 (免疫"你只是调参"的质疑)。
LAMBDA_GRID = np.logspace(-1, 3, 9) if not FAST_MODE else np.logspace(0, 2, 5)
LAM_MAIN    = 31.6         # 主工作点 (脚本17/18 的 oracle λ)

MAX_TEST_UNITS = None if not FAST_MODE else 2

# ---- MoE 超参 ----
K_MAIN   = 4                                  # 主结果的专家数 (对应 ⓪-d 的 TRA 四分位)
K_ABLATE = [1, 2, 4, 8] if not FAST_MODE else [1, 4]
HIDDEN   = 32                                 # 门控 MLP 的隐层宽度
EPOCHS   = 30 if not FAST_MODE else 6
BATCH    = 4096
LR       = 3e-3
N_TRAIN_MAX = 300000 if not FAST_MODE else 60000   # 训练样本上限 (够用了)
DIR_PENALTY = 1e-3        # MoE_dir 臂里对 H0 扰动的 L2 惩罚

# ---- 预注册判据的阈值 (写死) ----
Q1_FWD_GAIN_MIN = 0.05     # 前向 NRMSE 至少降低 5%
Q2_DOMINANCE_MIN = 0.95    # 逐点占优比例 >= 95%
Q3_DIR_GAIN_MAX = 0.01     # 允许方向扰动带来的额外增益必须 < 0.01, 否则冻结假设不成立
Q4_MOE_TOL      = 0.005    # MoE 允许比 FreeMLP 差多少

# ⓪-d 的手工分档结果 (用来和学出来的门控对比, 见 Q5)
TRA_BIN_REF = {          # 参数 -> (最低TRA档的幅值, 最高TRA档的幅值)
    "fan_flow_mod": (0.59, 1.24),
    "LPT_eff_mod":  (0.66, 1.26),
    "fan_eff_mod":  (0.66, 1.21),
    "HPT_eff_mod":  (0.91, 1.10),
}


# ============================================================================
# 数据读取 —— 与脚本 16/17/18 逐字一致
# ============================================================================
def get_names(f, key):
    """h5 里列名存成【字节串】, 必须 decode 成普通字符串, 否则拿不到列"""
    arr = np.array(f[key]).ravel()
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def load_for_ident(path):
    """辨识用: 先读小的 A_dev 拿行号, 再只读需要的行 (避免把 27GB 全读进内存)"""
    with h5py.File(path, "r") as f:
        A_var, W_var = get_names(f, "A_var"), get_names(f, "W_var")
        Xs_var, T_var = get_names(f, "X_s_var"), get_names(f, "T_var")
        df_A = pd.DataFrame(np.array(f["A_dev"]), columns=A_var)
        ih  = np.where(df_A["hs"].values == 1)[0]     # hs=1 健康
        idg = np.where(df_A["hs"].values == 0)[0]     # hs=0 退化
        take = np.sort(np.concatenate([
            rng.choice(ih,  min(N_HEALTHY_ID,  len(ih)),  replace=False),
            rng.choice(idg, min(N_DEGRADED_ID, len(idg)), replace=False)]))
        W, Xs, T = f["W_dev"][take, :], f["X_s_dev"][take, :], f["T_dev"][take, :]
    return pd.concat([df_A.iloc[take].reset_index(drop=True),
                      pd.DataFrame(W,  columns=W_var),
                      pd.DataFrame(Xs, columns=Xs_var),
                      pd.DataFrame(T,  columns=T_var)], axis=1)


def load_per_cycle(path, split):
    """验证用: 按 (unit, cycle) 分组抽样 —— 因为聚合窗口是 cycle 内部的"""
    with h5py.File(path, "r") as f:
        A_var, W_var = get_names(f, "A_var"), get_names(f, "W_var")
        Xs_var, T_var = get_names(f, "X_s_var"), get_names(f, "T_var")
        df_A = pd.DataFrame(np.array(f[f"A_{split}"]), columns=A_var)
        take = []
        for (u, c), idx in df_A.groupby(["unit", "cycle"]).indices.items():
            if len(idx) > MAX_PER_CYCLE:
                idx = rng.choice(idx, MAX_PER_CYCLE, replace=False)
            take.append(idx)
        take = np.sort(np.concatenate(take))
        W  = f[f"W_{split}"][take, :]
        Xs = f[f"X_s_{split}"][take, :]
        T  = f[f"T_{split}"][take, :]
    return pd.concat([df_A.iloc[take].reset_index(drop=True),
                      pd.DataFrame(W,  columns=W_var),
                      pd.DataFrame(Xs, columns=Xs_var),
                      pd.DataFrame(T,  columns=T_var)], axis=1)


def add_corrected(df, ref):
    """相似理论修正参数 (Walsh & Fletcher 2004) -> 13 个修正通道"""
    df = df.copy()
    df["theta_c"] = df["T2"] / ref["T_ref"]
    df["delta_c"] = df["P2"] / ref["P_ref"]
    sq = np.sqrt(df["theta_c"])
    cols = []
    for c in ["Nf", "Nc"]:                       # 修正转速
        df[f"{c}_c"] = df[c] / sq; cols.append(f"{c}_c")
    df["Wf_c"] = df["Wf"] / (df["delta_c"] * sq); cols.append("Wf_c")   # 修正燃油
    for c in ["T24", "T30", "T48", "T50"]:       # 温比
        df[f"{c}_c"] = df[c] / df["T2"]; cols.append(f"{c}_c")
    for c in ["P15", "P21", "P24", "Ps30", "P40", "P50"]:               # 压比
        df[f"{c}_c"] = df[c] / df["P2"]; cols.append(f"{c}_c")
    return df, cols                              # P2 作分母自己消掉 -> 13 个


# ============================================================================
# MoE 门控网络
# ============================================================================
# softplus(x) = log(1+e^x)。我们希望初始化时增益 g ≈ 1 (即退化回全局 H0),
# 所以专家的原始参数要初始化成 softplus 的反函数在 1 处的值:
#   softplus(x)=1  =>  x = log(e^1 - 1) ≈ 0.5413
SOFTPLUS_INV_1 = float(np.log(np.e - 1.0))


class GainNet(nn.Module):
    """
    三种臂共用这一个类, 靠 mode 切换:

      mode="moe"   : g(w) = Σ_k α_k(w)·s_k     α=softmax(gate(w)), s_k=softplus(raw)>0
                     ★ 主方法。增益被限制在 K 个专家的凸包内。
      mode="free"  : g(w) = softplus(MLP(w))   ← 对照臂 Q4, 自由 MLP, 无专家结构
      mode="const" : g(w) ≡ 1                  ← 基线臂, 就是现在的全局 H0

    learn_dir=True 时额外学一个 13×9 的扰动 dH, 让 H0 也能被改动
      -> 这是【专门用来挑战"H0 冻结"这个假设】的臂 (判据 Q3)。
    """

    def __init__(self, K, mode="moe", learn_dir=False, m=13, n=9,
                 n_in=4, hidden=HIDDEN):
        super().__init__()
        self.mode, self.K, self.m, self.n = mode, K, m, n
        self.learn_dir = learn_dir

        if mode == "moe":
            # 门控网络: 工况 (4维) -> K 个 logit -> softmax -> K 个权重
            self.gate = nn.Sequential(
                nn.Linear(n_in, hidden), nn.Tanh(),
                nn.Linear(hidden, K))
            # K 个专家, 每个是 9 维的原始增益 (过 softplus 后恒正)
            self.experts_raw = nn.Parameter(
                torch.full((K, n), SOFTPLUS_INV_1))
        elif mode == "free":
            # 自由 MLP: 工况 -> 直接吐 9 个增益
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.Tanh(),
                nn.Linear(hidden, n))
            # 最后一层零初始化 -> 初始 g = softplus(0)=0.693, 不是 1
            # 所以加一个可学的偏置, 初始化到 SOFTPLUS_INV_1
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, SOFTPLUS_INV_1)
        # mode=="const" 时没有任何门控参数

        # 全局截距 b (13维)。H0 辨识时 LinearRegression 是带截距的,
        # 这里保持一致。残差本身已经减掉了健康基准, 所以 b 应该 ≈ 0,
        # 训完会打印 ||b||, 如果它很大, 说明健康基准有问题。
        self.bias = nn.Parameter(torch.zeros(m))

        if learn_dir:
            # 允许 H0 被一个自由的 13×9 扰动改动 (零初始化 -> 起点就是冻结的 H0)
            self.dH = nn.Parameter(torch.zeros(m, n))

    def gains(self, w):
        """w: (B,4) 标准化后的工况 -> 返回 (B,9) 的正增益"""
        if self.mode == "const":
            return torch.ones(w.shape[0], self.n, device=w.device)
        if self.mode == "free":
            return F.softplus(self.net(w))
        a = torch.softmax(self.gate(w), dim=-1)      # (B,K) 门控权重, 和为1
        s = F.softplus(self.experts_raw)             # (K,9) 专家增益, 恒正
        return a @ s                                 # (B,9) 凸组合

    def alphas(self, w):
        """只有 moe 模式有意义: 返回 (B,K) 的门控权重, 用于可解释性作图"""
        if self.mode != "moe":
            return None
        return torch.softmax(self.gate(w), dim=-1)

    def H_eff(self, Hn):
        """有效的 H0 (可能被 dH 扰动过)。Hn 是常量, 不参与梯度。"""
        return Hn + self.dH if self.learn_dir else Hn

    def forward(self, w, th_n, Hn):
        """
        正向模型:  r_hat = [H0(+dH)] · diag(g(w)) · θ_n  +  b

        w    : (B,4)   标准化工况
        th_n : (B,9)   真值 θ / 量程  (归一化的健康参数)
        Hn   : (13,9)  冻结的归一化影响系数矩阵 (torch 常量)
        返回  : (B,13) 预测的归一化残差
        """
        g = self.gains(w)                            # (B,9)
        # diag(g)·θ_n 就是逐元素相乘 (这是 diag 矩阵乘法的等价写法, 快很多)
        gt = g * th_n                                # (B,9)
        H = self.H_eff(Hn)                           # (13,9)
        return gt @ H.T + self.bias                  # (B,13)


def fit_gain_ls(Hn, TH_n, R, w_bin_mask=None):
    """
    【非学习的对照臂: TRA 硬分档】
    在给定的样本子集上, 用最小二乘直接解出一个【常数】增益向量 g ∈ R^9:

        min_g  Σ_i || Hn·diag(g)·θ_i − r_i ||²

    注意 Hn·diag(g)·θ = Σ_j g_j · (Hn[:,j] · θ_j), 对 g 是【线性的】。
    所以设计矩阵是 A_i[:, j] = Hn[:, j] * θ_i[j], 直接正规方程求解。

    ⚠️ 某个参数 j 如果在这批样本里 θ_j 恒为 0 (没退化), 那 g_j 无法辨识
       -> 正规方程会奇异 -> 用 ridge 兜底, 并把该 g_j 拉回 1。
    """
    if w_bin_mask is not None:
        TH_n, R = TH_n[w_bin_mask], R[w_bin_mask]
    m, n = Hn.shape
    # A: (B*13, 9)
    A = (TH_n[:, None, :] * Hn[None, :, :]).reshape(-1, n)
    y = R.reshape(-1)
    AtA = A.T @ A
    # 哪些参数在这批数据里根本没动过 -> 无法辨识
    dead = np.diag(AtA) < 1e-8 * max(np.diag(AtA).max(), 1e-12)
    g = np.ones(n)
    ridge = 1e-6 * np.trace(AtA) / n
    try:
        g = np.linalg.solve(AtA + ridge * np.eye(n), A.T @ y)
    except np.linalg.LinAlgError:
        pass
    g[dead] = 1.0                       # 辨识不了的, 保持全局值
    return np.clip(g, 0.05, 5.0)        # 增益必须为正, 且不允许离谱


# ============================================================================
# 反演器 —— 支持【逐 cycle 不同的 H】
# ============================================================================
def build_cum(T, n):
    """累加矩阵: θ = −Cum·d, 把增量 d 变成累计退化 θ (分块下三角, 每块是单位阵)"""
    Cum = np.zeros((T * n, T * n))
    for t in range(T):
        for i in range(t + 1):
            Cum[t*n:(t+1)*n, i*n:(i+1)*n] = np.eye(n)
    return Cum


def build_M_var(Hn_list, T):
    """
    ★ 脚本18 的 build_M 用的是【一个固定的 Hn】。这里改成【每个 cycle 一个 Hn_t】。

    模型:  r(t) = Hn_t · θ(t) = Hn_t · (−Σ_{i<=t} d_i)
    所以设计矩阵的 (t, i) 块 (i <= t) 是  −Hn_t   ← 注意下标是 t, 不是 i
    (因为 r(t) 的所有历史增量都要经过【当前 cycle 的】H 才能变成残差)
    """
    m, n = Hn_list[0].shape
    M = np.zeros((T * m, T * n))
    for t in range(T):
        Ht = -Hn_list[t]
        for i in range(t + 1):
            M[t*m:(t+1)*m, i*n:(i+1)*n] = Ht
    return M


def solve_B_var(Hn_list, R, lam):
    """B: Tikhonov 岭回归 (Doel 1994), 无约束, 逐 cycle 独立解, 每个 cycle 用自己的 H"""
    T, n = len(R), Hn_list[0].shape[1]
    out = np.zeros((T, n))
    for t in range(T):
        Ht = Hn_list[t]
        A = Ht.T @ Ht + lam * np.eye(n)
        out[t] = np.linalg.solve(A, Ht.T @ R[t])
    return out


def solve_D_var(M, Cum, R, lam, T, n):
    """
    D: 约束 + 收缩 (本文方法)
        min ‖H·θ − r‖² + λ‖θ‖²    s.t.  θ <= 0 (退化不可逆), θ(t+1) <= θ(t) (单调)
    重参数化 θ = −Cum·d, d >= 0  ->  约束【自动满足】, 退化成 NNLS, 不需要 cvxpy。
    Tikhonov 项用【数据增广】塞进 NNLS:
        min ‖ [M ; √λ·Cum]·d − [r ; 0] ‖²   s.t. d >= 0
    ⚠️ M 和 Cum 不依赖 λ -> 必须在 λ 循环【外面】构造好, 否则慢到不可接受。
    """
    M_aug = np.vstack([M, np.sqrt(lam) * Cum])
    y_aug = np.concatenate([R.reshape(-1), np.zeros(T * n)])
    d, _ = nnls(M_aug, y_aug, maxiter=50 * T * n)
    return -np.cumsum(d.reshape(T, n), axis=0)


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 100)
    print("脚本 19: 工况自适应影响系数矩阵  H(w) = H0 · diag(g(w))   [MoE 门控]")
    print("=" * 100)
    print(f"\n  FAST_MODE = {FAST_MODE}")
    print(f"  门控输入   = {OP_COLS}   (与第2章健康基准的自变量【完全一致】)")
    print(f"  专家数 K   = {K_MAIN} (主), 消融 {K_ABLATE}")
    print(f"  lambda 网格 = {len(LAMBDA_GRID)} 点, 主工作点 lam = {LAM_MAIN}")
    print(f"  聚合窗口   = {WINDOWS}")
    print("\n  【预注册判据 (已写死, 跑完不许改)】")
    print(f"    Q1 前向:     MoE 在 DS02/DS03 上前向 NRMSE 降低 >= {Q1_FWD_GAIN_MIN:.0%}")
    print(f"    Q2 逐点占优: 所有 (子集 x N x lam) 上 MoE > H0 的比例 >= {Q2_DOMINANCE_MIN:.0%}")
    print(f"    Q3 冻结合法: MoE_dir 相对 MoE 的额外增益 < {Q3_DIR_GAIN_MAX}")
    print(f"    Q4 结构代价: MoE >= FreeMLP - {Q4_MOE_TOL}")
    print( "    Q5 门控物理性: 描述性, 与 (0)-d 手工分档对比")
    print( "    !! Q3 是最可能打脸的: 若不成立, 必须如实报告【方向也需要自适应】,")
    print( "       并把 '余弦>=0.9865 所以方向不变' 这条证据的解释力下调。")

    # ------------------------------------------------------------------
    # [1/6] 重建健康基准与 H0 —— 与脚本 16/17/18 完全一致
    # ------------------------------------------------------------------
    print("\n\n[1/6] 重建健康基准与影响系数矩阵 H0 ...")
    data = {}
    for fn in IDENT_FILES:
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"  !! 找不到文件: {p}")
            return
        data[fn] = load_for_ident(p)
        print(f"    读入 {fn}: {len(data[fn])} 行")

    pool = pd.concat([d[d["hs"] == 1] for d in data.values()], ignore_index=True)
    ref = {"T_ref": pool["T2"].median(), "P_ref": pool["P2"].median()}
    print(f"    全局参考: T_ref={ref['T_ref']:.2f}, P_ref={ref['P_ref']:.2f}")

    for fn in data:
        data[fn], corrected_cols = add_corrected(data[fn], ref)
    pool, _ = add_corrected(pool, ref)
    resid_cols = [f"r_{c}" for c in corrected_cols]

    base = {}
    for c in corrected_cols:
        m = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=SEED)
        # ⚠️ 自变量【绝不能】加 Nf/Nc —— 转速本身会被退化污染, 逻辑上循环。
        m.fit(pool[OP_COLS].values, pool[c].values)
        base[c] = m
    print(f"    健康基准: {len(base)} 个 GBM 拟合完成")

    def residual(df):
        r = pd.DataFrame(index=df.index)
        for c in corrected_cols:
            r[f"r_{c}"] = df[c].values - base[c].predict(df[OP_COLS].values)
        return r

    resid_std = residual(pool).std()      # 用健康样本的残差 std 做归一化 (单位 = σ)

    # ---- 辨识 H0 (与脚本18 逐字一致) ----
    H_cols = {}
    for fn in ["N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS04.h5",
               "N-CMAPSS_DS05.h5", "N-CMAPSS_DS07.h5"]:
        df, params = data[fn], IDENT_FILES[fn]
        r = residual(df) / resid_std
        for rc in resid_cols:
            lr = LinearRegression().fit(df[params].values, r[rc].values)
            for j, p in enumerate(params):
                H_cols.setdefault(p, {})[rc] = lr.coef_[j]

    df6 = data["N-CMAPSS_DS06.h5"]
    df6 = df6[df6["hs"] == 0]
    r6 = residual(df6) / resid_std
    p6 = IDENT_FILES["N-CMAPSS_DS06.h5"]
    for rc in resid_cols:
        lr = LinearRegression().fit(df6[p6].values, r6[rc].values)
        for j, p in enumerate(p6):
            if p.startswith("LPC"):        # DS06 只取 LPC 两列 (HPC 用 DS05 的)
                H_cols.setdefault(p, {})[rc] = lr.coef_[j]

    H = pd.DataFrame(H_cols).reindex(index=resid_cols)[THETA9]
    span = np.array([THETA_SPAN[c] for c in THETA9])
    Hn = H.values * span                              # ⚠️ 列乘量程, 条件数才有物理意义
    U, s, Vt = np.linalg.svd(Hn)
    print(f"    cond(H_n) = {s[0]/s[-1]:.1f} , sigma_min = {s[-1]:.4f} sigma")
    Hn_t = torch.tensor(Hn, dtype=torch.float32)      # 冻结的常量, 不参与梯度

    # ------------------------------------------------------------------
    # [2/6] 构造门控的训练集 (正向模型: 给定 θ 和 w, 预测 r)
    # ------------------------------------------------------------------
    print("\n[2/6] 构造门控训练集 (正向模型: (theta, w) -> r) ...")
    Ws, THs, Rs, UNITs, SRCs = [], [], [], [], []
    for fi, (fn, params) in enumerate(IDENT_FILES.items()):
        df = data[fn]
        df = df[df["hs"] == 0]                        # 只用退化样本 (健康样本 θ≈0, 对增益无信息)
        if len(df) == 0:
            continue
        r = (residual(df) / resid_std).values         # (B,13) 归一化残差
        th = df[THETA9].values / span                 # (B,9)  归一化真值 θ
        Ws.append(df[OP_COLS].values)
        THs.append(th)
        Rs.append(r)
        # unit 编号在不同子集里会重复, 加个偏移让它们全局唯一 (为了按发动机切分)
        UNITs.append(df["unit"].values + 1000 * fi)
        SRCs.append(np.full(len(df), fi))

    W_all  = np.concatenate(Ws).astype(np.float32)
    TH_all = np.concatenate(THs).astype(np.float32)
    R_all  = np.concatenate(Rs).astype(np.float32)
    U_all  = np.concatenate(UNITs)
    del data, pool, Ws, THs, Rs

    # 样本太多就随机下采样 (够用了, 而且训练快)
    if len(W_all) > N_TRAIN_MAX:
        sel = rng.choice(len(W_all), N_TRAIN_MAX, replace=False)
        W_all, TH_all, R_all, U_all = W_all[sel], TH_all[sel], R_all[sel], U_all[sel]

    # ★ 按【发动机单元】切分 train/val —— 不能按样本随机切 (方法论纪律 #1)
    units = np.unique(U_all)
    rng.shuffle(units)
    n_val = max(1, int(0.25 * len(units)))
    val_units = set(units[:n_val].tolist())
    is_val = np.array([u in val_units for u in U_all])
    print(f"    训练样本 {(~is_val).sum()} 行 / 验证样本 {is_val.sum()} 行")
    print(f"    训练单元 {len(units)-n_val} 台 / 验证单元 {n_val} 台 (按发动机切分)")

    # 工况标准化 (只用【训练集】的统计量, 否则泄漏)
    w_mu = W_all[~is_val].mean(axis=0)
    w_sd = W_all[~is_val].std(axis=0) + 1e-8
    Wz_all = (W_all - w_mu) / w_sd

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32)

    Wtr, THtr, Rtr = to_t(Wz_all[~is_val]), to_t(TH_all[~is_val]), to_t(R_all[~is_val])
    Wva, THva, Rva = to_t(Wz_all[is_val]),  to_t(TH_all[is_val]),  to_t(R_all[is_val])

    # ------------------------------------------------------------------
    # [3/6] 训练各个臂
    # ------------------------------------------------------------------
    print("\n[3/6] 训练门控 ...")

    def train_arm(name, K, mode, learn_dir=False):
        """训练一个臂, 返回 (模型, 验证集NRMSE)"""
        net = GainNet(K=K, mode=mode, learn_dir=learn_dir)
        if mode == "const" and not learn_dir:
            # 全局 H0 基线: 除了截距 b 没有任何东西要学, 直接把 b 解析地设成残差均值
            with torch.no_grad():
                pred = THtr @ Hn_t.T
                net.bias.copy_((Rtr - pred).mean(dim=0))
            nrmse = eval_forward(net, Wva, THva, Rva)
            print(f"    [{name:<10s}] (无需训练)  val NRMSE = {nrmse:.4f}")
            return net, nrmse

        opt = torch.optim.Adam(net.parameters(), lr=LR)
        n = len(Wtr)
        best, best_state = 1e9, None
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            net.train()
            for i in range(0, n, BATCH):
                idx = perm[i:i+BATCH]
                pred = net(Wtr[idx], THtr[idx], Hn_t)
                loss = F.mse_loss(pred, Rtr[idx])
                if learn_dir:
                    # 惩罚 H0 的扰动 -> "尽量别改 H0, 除非数据强烈要求"
                    loss = loss + DIR_PENALTY * (net.dH ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            v = eval_forward(net, Wva, THva, Rva)
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            if ep % max(1, EPOCHS // 5) == 0 or ep == EPOCHS - 1:
                print(f"    [{name:<10s}] ep {ep:>3d}  val NRMSE = {v:.4f}")
        net.load_state_dict(best_state)     # 用验证集最优的那一轮 (早停)
        print(f"    [{name:<10s}] ==> best val NRMSE = {best:.4f}   ||b|| = {net.bias.norm():.4f}")
        return net, best

    @torch.no_grad()
    def eval_forward(net, W, TH, R):
        """前向重构的 NRMSE = ‖r_hat − r‖ / ‖r‖ (归一化, 所以跨子集可比)"""
        net.eval()
        pred = net(W, TH, Hn_t)
        return float(torch.norm(pred - R) / (torch.norm(R) + 1e-12))

    arms = {}
    arms["H0"], nrmse_H0 = train_arm("H0", 1, "const")
    for K in K_ABLATE:
        arms[f"MoE_K{K}"], _ = train_arm(f"MoE_K{K}", K, "moe")
    arms["FreeMLP"], _ = train_arm("FreeMLP", 1, "free")
    arms["MoE_dir"], _ = train_arm("MoE_dir", K_MAIN, "moe", learn_dir=True)

    MOE = f"MoE_K{K_MAIN}"

    # ---- MoE_dir 到底把 H0 改了多少? (Q3 的直接证据) ----
    with torch.no_grad():
        dH = arms["MoE_dir"].dH.numpy()
        H_pert = Hn + dH
        col_cos = [float(np.dot(Hn[:, j], H_pert[:, j]) /
                         (np.linalg.norm(Hn[:, j]) * np.linalg.norm(H_pert[:, j]) + 1e-12))
                   for j in range(9)]
        print(f"\n    MoE_dir 对 H0 的扰动: ||dH||_F / ||H0||_F = "
              f"{np.linalg.norm(dH)/np.linalg.norm(Hn):.4f}")
        print(f"    各列方向余弦 (扰动后 vs 冻结的 H0): 最差 = {min(col_cos):.4f}")
        print(f"      -> 对比 (0)-d 的分档余弦最差 0.9865, "
              f"以及 fan_eff vs LPT_eff 的夹角 23.5 度")

    # ---- TRA 硬分档臂 (⓪-d 的复刻, 非学习) ----
    print("\n    [TRA4     ] 手工 TRA 四分位硬分档 (非学习, (0)-d 的复刻)")
    tra_tr = W_all[~is_val][:, 0]
    tra_q = np.quantile(tra_tr, [0.25, 0.5, 0.75])
    g_bins = []
    for b in range(4):
        lo = -np.inf if b == 0 else tra_q[b-1]
        hi = np.inf if b == 3 else tra_q[b]
        mk = (tra_tr > lo) & (tra_tr <= hi)
        g_bins.append(fit_gain_ls(Hn, TH_all[~is_val], R_all[~is_val], mk))
    g_bins = np.array(g_bins)                       # (4, 9)
    print("      各档增益 (对比 (0)-d: fan_flow 0.59->1.24, LPT_eff 0.66->1.26):")
    for j, name in enumerate(THETA9):
        if name in TRA_BIN_REF:
            lo_ref, hi_ref = TRA_BIN_REF[name]
            print(f"        {name:<14s} 学出 {g_bins[0,j]:.2f} -> {g_bins[3,j]:.2f}"
                  f"   |  (0)-d 参考 {lo_ref:.2f} -> {hi_ref:.2f}")

    def gains_tra4(W_raw):
        """TRA 硬分档的增益查表 (W_raw 是【原始】工况, 不是标准化的)"""
        tra = W_raw[:, 0]
        b = np.digitize(tra, tra_q)                 # 0/1/2/3
        return g_bins[b]

    # ------------------------------------------------------------------
    # [4/6] 在 DS02/DS03 上评测 —— 前向 (Q1) + 诊断 (Q2/Q3/Q4)
    # ------------------------------------------------------------------
    print("\n[4/6] 在 DS02/DS03 上评测 (三部件同时退化 + 从未见过的发动机) ...")

    @torch.no_grad()
    def gains_of(arm_name, W_raw):
        """给定原始工况 (B,4), 返回该臂的增益 (B,9)"""
        if arm_name == "TRA4":
            return gains_tra4(W_raw)
        net = arms[arm_name]
        net.eval()
        wz = to_t((W_raw - w_mu) / w_sd)
        return net.gains(wz).numpy()

    def build_windows(df, unit):
        """
        返回 {N: (R, TH, OPS)}
          R   (T,13)  归一化残差 (相对该单元零点)
          TH  (T,9)   真值 θ/量程 (相对该单元零点)
          OPS list[T] 每个 cycle 里【实际抽到的那 k 个样本】的原始工况 (k,4)
                      ★ 必须存下来, 因为 ḡ_t 要在【同一批样本】上求平均

        ⚠️ 零点 = 该单元最初 3 个 cycle, 【不是】健康期均值。
           (脚本14 的 bug: 健康期内 θ 本就在缓慢下降, 用均值当零点会让健康期
            前半段 θ_relative > 0 -> 真值落在 θ<=0 的可行域【外面】-> 方法被自己判死)
        """
        du = df[df["unit"] == unit].copy()
        cycles = sorted(du["cycle"].unique())
        m_ref = du["cycle"].isin(cycles[:N_REF_CYCLES])
        b_unit = du.loc[m_ref, resid_cols].mean().values
        th0 = (du.loc[m_ref, THETA9].values / span).mean(axis=0)

        out = {}
        for N in WINDOWS:
            R_list, TH_list, OP_list = [], [], []
            for c in cycles:
                sub = du[du["cycle"] == c]
                k = min(N, len(sub))
                idx = rng.choice(len(sub), k, replace=False)
                R_list.append(sub[resid_cols].values[idx].mean(axis=0) - b_unit)
                TH_list.append(sub[THETA9].values[0] / span - th0)
                OP_list.append(sub[OP_COLS].values[idx])       # (k,4) 同一批样本
            out[N] = (np.array(R_list), np.array(TH_list), OP_list)
        return out

    rows, fwd_rows = [], []

    for tag, (fn, true_faults) in VALID_FILES.items():
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"\n  !! 缺 {fn}, 跳过 {tag}")
            continue
        print("\n" + "#" * 100)
        print(f"# {tag}    真实退化部件: {true_faults}")
        print("#" * 100)

        ds = {}
        for split in ["dev", "test"]:
            d = load_per_cycle(p, split)
            d, _ = add_corrected(d, ref)
            ds[split] = pd.concat([d, residual(d) / resid_std], axis=1)

        idx_tf = [THETA9.index(q) for q in true_faults]        # 真退化部件的下标
        other  = [i for i in range(9) if i not in idx_tf]      # 未退化部件 -> 看虚警

        # -------- (a) 前向重构 NRMSE (判据 Q1) --------
        # 直接在 test 的【原始样本】上算: 给定真值 θ 和工况 w, 预测残差 r
        dtest = ds["test"]
        Rt  = dtest[resid_cols].values.astype(np.float32)
        THt = (dtest[THETA9].values / span).astype(np.float32)
        Wt  = dtest[OP_COLS].values.astype(np.float32)
        print(f"\n  --- (a) 前向重构 NRMSE (判据 Q1) ---")
        for arm in ["H0", "TRA4"] + [f"MoE_K{K}" for K in K_ABLATE] + ["FreeMLP", "MoE_dir"]:
            g = gains_of(arm, Wt)                              # (B,9)
            if arm == "TRA4":
                bias = arms["H0"].bias.detach().numpy()        # 硬分档臂借用 H0 的截距
                Hm = Hn
            else:
                bias = arms[arm].bias.detach().numpy()
                Hm = arms[arm].H_eff(Hn_t).detach().numpy()
            pred = (g * THt) @ Hm.T + bias
            nrmse = float(np.linalg.norm(pred - Rt) / (np.linalg.norm(Rt) + 1e-12))
            fwd_rows.append(dict(subset=tag, arm=arm, nrmse=nrmse))
            print(f"      {arm:<10s}  NRMSE = {nrmse:.4f}")

        # -------- (b) 诊断: 逐 (unit, N, λ, 臂) 反演 --------
        test_units = sorted(ds["test"]["unit"].unique())
        if MAX_TEST_UNITS is not None:
            test_units = test_units[:MAX_TEST_UNITS]
        eval_arms = ["H0", "TRA4", MOE, "FreeMLP", "MoE_dir"]
        n_solve = len(test_units) * len(WINDOWS) * len(LAMBDA_GRID) * len(eval_arms)
        print(f"\n  --- (b) 诊断反演 (判据 Q2/Q3/Q4) ---")
        print(f"      test 单元: {[int(u) for u in test_units]}")
        print(f"      共需 {n_solve} 次 NNLS, 请耐心 ...")

        for u in test_units:
            wins = build_windows(ds["test"], u)
            for N in WINDOWS:
                R, TH, OPS = wins[N]
                T = len(R)
                Cum = build_cum(T, 9)
                mse_zero = float(np.mean(TH ** 2))          # 全零哨兵的 MSE
                n_early = max(1, int(0.3 * T))              # 早期 = 前 30% cycle

                # ---- 全零哨兵 (与 λ、与臂都无关, 只记一次) ----
                rows.append(dict(
                    subset=tag, unit=int(u), N=N, arm="Z_zero", lam=np.nan,
                    skill=0.0, rmse=float(np.sqrt(mse_zero)),
                    rmse_early=float(np.sqrt(np.mean(TH[:n_early] ** 2))),
                    false_alarm=0.0, detect_corr=0.0))

                for arm in eval_arms:
                    # ★ 每个 cycle 的等效矩阵 = Hn·diag(ḡ_t)
                    #   ḡ_t = 该 cycle 内【实际抽到的那 k 个样本】的增益均值
                    #   (前面推过, 这是精确的, 不是近似)
                    Hbase = Hn if arm == "TRA4" else \
                            arms[arm].H_eff(Hn_t).detach().numpy()
                    Hn_list = []
                    for t in range(T):
                        g_bar = gains_of(arm, OPS[t]).mean(axis=0)   # (9,)
                        Hn_list.append(Hbase * g_bar[None, :])       # 列缩放 = 右乘 diag
                    M = build_M_var(Hn_list, T)

                    for lam in LAMBDA_GRID:
                        TH_hat = solve_D_var(M, Cum, R, lam, T, 9)   # 本文方法 (约束+收缩)
                        err = TH_hat - TH
                        mse = float(np.mean(err ** 2))
                        # 技能分 S = 1 − MSE/MSE(全零)。1=完美, 0=跟哨兵一样, <0=更糟
                        skill = 1.0 - mse / max(mse_zero, 1e-12)
                        # 早期用【绝对RMSE】不用技能分: 早期 θ_true≈0 -> 分母趋零 -> 被除爆
                        rmse_early = float(np.sqrt(np.mean(err[:n_early] ** 2)))
                        # 虚警幅度: 6 个未退化部件上 |θ̂| 的均值 (真值恒为 0)
                        fa = float(np.mean(np.abs(TH_hat[:, other])))
                        # 检出相关性: 3 个真退化部件上 θ̂ 与真值的相关系数
                        cs = [np.corrcoef(TH_hat[:, i], TH[:, i])[0, 1]
                              for i in idx_tf
                              if np.std(TH[:, i]) > 1e-9 and np.std(TH_hat[:, i]) > 1e-9]
                        rows.append(dict(
                            subset=tag, unit=int(u), N=N, arm=arm, lam=float(lam),
                            skill=skill, rmse=float(np.sqrt(mse)), rmse_early=rmse_early,
                            false_alarm=fa,
                            detect_corr=float(np.mean(cs)) if cs else 0.0))
            print(f"      unit {int(u)} 完成")

    df = pd.DataFrame(rows)
    fwd = pd.DataFrame(fwd_rows)
    df.to_csv(os.path.join(OUT_DIR, "moe_raw.csv"), index=False, encoding="utf-8-sig")
    fwd.to_csv(os.path.join(OUT_DIR, "moe_forward.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  原始结果已存: moe_raw.csv ({len(df)} 行), moe_forward.csv")

    # ------------------------------------------------------------------
    # [5/6] 裁决 —— 严格按预注册判据, 不许事后改
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("[5/6] 裁  决 (严格按预注册判据)")
    print("=" * 100)

    tags = sorted(df["subset"].unique())

    # ---- Q1: 前向 ----
    print("\n" + "#" * 100)
    print("# Q1  前向重构 NRMSE —— 工况自适应在【正问题】上有没有用?")
    print("#" * 100)
    pf = fwd.pivot(index="subset", columns="arm", values="nrmse")
    print(pf.round(4).to_string())
    Q1_ok = []
    for tg in tags:
        gain = 1.0 - pf.loc[tg, MOE] / pf.loc[tg, "H0"]
        ok = gain >= Q1_FWD_GAIN_MIN
        Q1_ok.append(ok)
        print(f"\n  {tg}: MoE 相对 H0 的 NRMSE 降低 = {gain:+.1%}   "
              f"(阈值 {Q1_FWD_GAIN_MIN:.0%})  {'[OK]' if ok else '[X]'}")
    Q1 = all(Q1_ok)
    print(f"\n  ==> Q1 {'成立' if Q1 else '不成立'}")

    # ---- Q2: 逐点占优 (MoE vs H0) ----
    print("\n" + "#" * 100)
    print("# Q2  逐点占优 —— 在每个 (子集, N, lambda) 上, MoE 是否优于 H0?")
    print("#     ★ 这是第3章的主图3: 证明工况自适应的增益【不是调 lambda 调出来的】")
    print("#" * 100)
    grid = df[df.arm.isin([MOE, "H0"])].pivot_table(
        index=["subset", "N", "lam"], columns="arm", values="skill", aggfunc="mean")
    grid["gap"] = grid[MOE] - grid["H0"]
    print(f"\n  {'子集':<8s} {'N':>6s} {'网格点':>8s} {'MoE占优':>8s} {'占优率':>8s} "
          f"{'gap中位':>10s} {'gap最小':>10s}")
    print("  " + "-" * 66)
    tot, win = 0, 0
    for (tg, N), g in grid.groupby(level=[0, 1]):
        w = int((g["gap"] > 0).sum()); n = len(g)
        tot += n; win += w
        print(f"  {tg:<8s} {N:>6d} {n:>8d} {w:>8d} {w/n:>7.1%} "
              f"{g['gap'].median():>+10.4f} {g['gap'].min():>+10.4f}")
    rate = win / max(tot, 1)
    print("  " + "-" * 66)
    print(f"  {'总计':<8s} {'':>6s} {tot:>8d} {win:>8d} {rate:>7.1%}")
    Q2 = rate >= Q2_DOMINANCE_MIN
    print(f"\n  ==> Q2 {'成立' if Q2 else '不成立'}  "
          f"(占优率 {rate:.1%}, 阈值 {Q2_DOMINANCE_MIN:.0%})")

    # ---- Q3/Q4/Q5: 主工作点上的对比 ----
    print("\n" + "#" * 100)
    print(f"# Q3/Q4  主工作点 lam = {LAM_MAIN} 上的各臂对比")
    print("#" * 100)
    lam_pick = float(LAMBDA_GRID[np.argmin(np.abs(LAMBDA_GRID - LAM_MAIN))])
    main = df[(df.lam == lam_pick) | (df.arm == "Z_zero")]
    tbl = main.pivot_table(index=["subset", "arm"],
                           values=["skill", "false_alarm", "detect_corr", "rmse_early"],
                           aggfunc="mean")
    for tg in tags:
        z_early = tbl.loc[(tg, "Z_zero"), "rmse_early"]
        print(f"\n  --- {tg}   (lam={lam_pick:.3g}, 全零哨兵的早期RMSE = {z_early:.4f}) ---")
        print(f"    {'臂':<10s} {'技能分':>9s} {'虚警幅度':>10s} {'检出相关':>10s} {'早期RMSE':>10s}")
        print("    " + "-" * 54)
        for arm in ["Z_zero", "H0", "TRA4", MOE, "FreeMLP", "MoE_dir"]:
            if (tg, arm) not in tbl.index:
                continue
            r_ = tbl.loc[(tg, arm)]
            flag = "  !! 不如哨兵" if r_["rmse_early"] > z_early and arm != "Z_zero" else ""
            print(f"    {arm:<10s} {r_['skill']:>9.4f} {r_['false_alarm']:>10.4f} "
                  f"{r_['detect_corr']:>10.4f} {r_['rmse_early']:>10.4f}{flag}")

    print("\n  --- Q3: H0 冻结的合法性 (MoE_dir 相对 MoE 的额外增益) ---")
    Q3_ok = []
    for tg in tags:
        extra = tbl.loc[(tg, "MoE_dir"), "skill"] - tbl.loc[(tg, MOE), "skill"]
        ok = extra < Q3_DIR_GAIN_MAX
        Q3_ok.append(ok)
        print(f"    {tg}: MoE_dir - MoE = {extra:+.4f}   "
              f"(阈值 < {Q3_DIR_GAIN_MAX})  {'[OK]' if ok else '[X]'}")
    Q3 = all(Q3_ok)
    print(f"    ==> Q3 {'成立: 冻结 H0 的方向是合法的' if Q3 else '不成立: 方向也需要自适应!'}")

    print("\n  --- Q4: MoE 的专家结构有没有付出精度代价 (vs 自由 MLP) ---")
    Q4_ok = []
    for tg in tags:
        d_ = tbl.loc[(tg, MOE), "skill"] - tbl.loc[(tg, "FreeMLP"), "skill"]
        ok = d_ >= -Q4_MOE_TOL
        Q4_ok.append(ok)
        print(f"    {tg}: MoE - FreeMLP = {d_:+.4f}   "
              f"(阈值 >= -{Q4_MOE_TOL})  {'[OK]' if ok else '[X]'}")
    Q4 = all(Q4_ok)
    print(f"    ==> Q4 {'成立: 专家结构不损失精度, 白赚可解释性' if Q4 else '不成立: 自由 MLP 更强, 如实报告'}")

    # ---- K 的消融 ----
    print("\n  --- K 的消融 (技能分, 主 lambda) ---")
    kab = df[(df.lam == lam_pick) & (df.arm.str.startswith("MoE_K"))]
    if len(kab):
        kt = kab.pivot_table(index="subset", columns="arm", values="skill", aggfunc="mean")
        print(kt.round(4).to_string())
        print("    (K=1 应该 ≈ H0, 因为单专家 = 一个全局常数增益)")

    # ---- Q5: 门控的物理性 ----
    print("\n  --- Q5: 门控学到了什么 (描述性) ---")
    net = arms[MOE]
    with torch.no_grad():
        s_k = F.softplus(net.experts_raw).numpy()             # (K,9) 各专家的增益
        # 在训练集工况上算门控权重, 看每个专家"负责"哪个 TRA 区间
        a_k = net.alphas(to_t(Wz_all[~is_val])).numpy()        # (B,K)
    tra = W_all[~is_val][:, 0]
    print(f"    {'专家':<8s} {'占比':>7s} {'加权平均TRA':>12s}   各部件增益 (前4个)")
    order = np.argsort([(a_k[:, k] * tra).sum() / a_k[:, k].sum() for k in range(K_MAIN)])
    for k in order:
        wgt = a_k[:, k]
        tra_k = float((wgt * tra).sum() / (wgt.sum() + 1e-12))
        gg = "  ".join(f"{THETA9[j][:8]}={s_k[k, j]:.2f}" for j in [1, 2, 0, 5])
        print(f"    专家{k:<4d} {wgt.mean():>6.1%} {tra_k:>12.2f}   {gg}")
    print("\n    (0)-d 手工分档参考: fan_flow 0.59->1.24, LPT_eff 0.66->1.26,")
    print("                        fan_eff 0.66->1.21, HPT_eff 0.91->1.10 (随 TRA 单调递增)")
    print("    ★ 如果上表按 TRA 排序后增益也单调递增 -> MoE 自主发现了这条物理规律。")

    verdict = {"Q1": bool(Q1), "Q2": bool(Q2), "Q2_rate": float(rate),
               "Q3": bool(Q3), "Q4": bool(Q4),
               "lam_main": lam_pick, "K_main": K_MAIN,
               "forward_nrmse": pf.round(4).reset_index().to_dict(orient="records"),
               # ⚠️ 不能直接 to_dict(): MultiIndex 会产生元组 key, JSON 不支持
               #    (这就是脚本18 崩在 json.dump 的那个坑)
               "main_table": tbl.round(4).reset_index().to_dict(orient="records")}
    with open(os.path.join(OUT_DIR, "verdict19.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)

    # ---- 总裁决 ----
    print("\n\n" + "=" * 100)
    allp = [Q1, Q2, Q3, Q4]
    if all(allp):
        print("[PASS] 全部判据通过 —— 第3章创新点1 成立, 可以按计划写。")
    else:
        print("[MIXED] 部分判据未通过 —— 逐条处理, 不许粉饰")
    print("=" * 100)
    if not Q1:
        print("""
    [X] Q1 不成立: 工况自适应在【正问题】上就没带来足够收益。
        ⇒ 先别怪方法, 先查三件事:
           1. 训练是否收敛? (看 val NRMSE 的曲线, 是不是还在降就停了 -> 加 EPOCHS)
           2. H0 本身是不是已经足够好? 看 H0 的 NRMSE 绝对值。如果它已经很低,
              那"幅值波动 2.12 倍"可能主要发生在【低退化量】的样本上,
              对总的 NRMSE 贡献很小 -> 应该【按退化程度分层】再报 NRMSE。
           3. ⚠️ 如果查完还是不行, 那就【如实报告】: 工况自适应的收益在
              N-CMAPSS 上不显著。这不丢人 —— (0)-d 的 2.12 倍是"分档辨识"
              量出来的, 不等于"用它做反演就一定更准"。
              诚实的负面结论比粉饰过的正面结论有价值得多。""")
    if not Q2:
        print("""
    [X] Q2 不成立: 逐点占优率不够。
        ⇒ 看上面的分组表, 失手点集中在哪?
           - 集中在【大 N】: 和脚本18 一样的机制 —— 噪声已被平均掉,
             额外的结构带来的偏差成本超过了方差收益。这是可解释的, 写进去。
           - 集中在【小 lambda】: 说明 H(w) 让病态性变严重了 (每个 cycle 的
             H 不同 -> 有效条件数可能更差)。⇒ 必须算出 cond(Hn·diag(g)) 的
             分布, 老老实实报告。
           - 散乱分布: 大概率是训练不稳 -> 换 seed 重跑 3 次看方差。""")
    if not Q3:
        print("""
    [X] Q3 不成立: 允许 H0 被扰动带来了【显著】的额外增益。
        ⇒ 这直接挑战了你的核心假设"方向稳、只有幅值变"。
        ⇒ 必须做的事:
           1. 看上面打印的"各列方向余弦": 到底哪几列被改动最大?
           2. 对照 (0)-d: 分档余弦最差 0.9865 (= 9.4 度抖动),
              而 fan_eff vs LPT_eff 的指纹夹角只有 23.5 度。
              抖动约是关键间隔的 1/3 —— 你自己文档里就写了"不能说方向完全不变"。
           3. ⇒ 诚实的写法: 把创新点从 "H(w)=H0·diag(g(w))" 改成
              "H(w) = [H0 + 低秩扰动(w)] · diag(g(w))", 并明确报告
              方向自适应带来了多少额外增益。这【依然是一个创新点】,
              只是更复杂、也更诚实。
        ⇒ 🔴 绝对不要因为"冻结版本更好写"就把 MoE_dir 这个臂删掉。""")
    if not Q4:
        print("""
    [X] Q4 不成立: 自由 MLP 比 MoE 明显更强。
        ⇒ 说明"K 个专家的凸包"这个约束【太紧了】, 真实的 g(w) 跑到了凸包外面。
        ⇒ 补救: 加大 K (试 K=16), 或者改成
                 g(w) = softplus( Σ_k α_k(w)·raw_k + MLP_residual(w) )
                 即"MoE 打底 + 自由残差修正", 保留可解释性的同时放开表达力。
        ⇒ 如果加大 K 也追不上, 就【如实报告】: 把 MoE 定位成
           "以极小的精度代价换取可解释的工况分区", 并给出精度差值。""")

    # ------------------------------------------------------------------
    # [6/6] 出图
    # ------------------------------------------------------------------
    print("\n[6/6] 出图 ...")
    try:
        # 图1: 门控权重 vs TRA (专家是否自动学出 TRA 分区) + 增益 vs TRA
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        tra_grid = np.linspace(np.quantile(tra, 0.01), np.quantile(tra, 0.99), 200)
        Wq = np.tile(W_all[~is_val].mean(axis=0), (200, 1))    # 其余工况固定在均值
        Wq[:, 0] = tra_grid
        with torch.no_grad():
            wz = to_t((Wq - w_mu) / w_sd)
            A = net.alphas(wz).numpy()                          # (200,K)
            G = net.gains(wz).numpy()                           # (200,9)

        ax = axes[0]
        for k in order:
            ax.plot(tra_grid, A[:, k], lw=2, label=f"专家{k}")
        ax.set_xlabel("TRA (油门杆角度)"); ax.set_ylabel("门控权重 alpha_k")
        ax.set_title("MoE 门控: 各专家负责的 TRA 区间\n(若自动分出高/低功率区 = 学到了物理)")
        ax.legend(); ax.grid(alpha=.3)

        ax = axes[1]
        for j in [THETA9.index(x) for x in ["fan_flow_mod", "LPT_eff_mod",
                                            "fan_eff_mod", "HPT_eff_mod"]]:
            ax.plot(tra_grid, G[:, j], lw=2, label=THETA9[j])
        ax.axhline(1.0, color="k", ls=":", lw=1, label="全局 H0 (g=1)")
        ax.set_xlabel("TRA (油门杆角度)"); ax.set_ylabel("增益 g_j(w)")
        ax.set_title("学出的工况增益\n((0)-d 手工分档: 随 TRA 单调递增, 波动 2.12 倍)")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig1_gate_physics.png"), dpi=150)
        plt.close(fig)

        # 图2 (主图3): 逐点占优 —— 技能分 vs lambda, 各臂一条线
        subs = tags
        fig, axes = plt.subplots(2, len(subs), figsize=(7.5*len(subs), 9), squeeze=False)
        for jj, tg in enumerate(subs):
            g_ = df[(df.subset == tg) & (df.arm != "Z_zero")]
            ax = axes[0][jj]
            for arm, sty in [("H0", "--"), ("TRA4", "-."), (MOE, "-"), ("FreeMLP", ":")]:
                gg = g_[g_.arm == arm].groupby("lam")["skill"].mean()
                if len(gg):
                    ax.plot(gg.index, gg.values, sty, marker="o", ms=4, lw=2, label=arm)
            ax.axhline(0, color="gray", ls=":", lw=1, label="全零哨兵 (S=0)")
            ax.set_xscale("log"); ax.set_ylim(-0.2, 1.0)
            ax.set_xlabel("正则化强度 lambda"); ax.set_ylabel("技能分 S")
            ax.set_title(f"{tg}  各臂的正则化路径")
            ax.legend(fontsize=8); ax.grid(alpha=.3)

            ax = axes[1][jj]
            for N in WINDOWS:
                gg = g_[(g_.N == N)].pivot_table(index="lam", columns="arm",
                                                 values="skill", aggfunc="mean")
                if MOE in gg and "H0" in gg:
                    ax.plot(gg.index, gg[MOE] - gg["H0"], marker="s", ms=4,
                            lw=2, label=f"N={N}")
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("正则化强度 lambda")
            ax.set_ylabel("技能分增益  MoE - H0")
            ax.set_title(f"{tg}  工况自适应的净贡献\n始终在 0 线之上 = 增益不依赖 lambda")
            ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.suptitle("脚本19 主图: 工况自适应 H(w) 的逐点占优", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig2_pointwise_moe.png"), dpi=150)
        plt.close(fig)
        print(f"    图已存到 {OUT_DIR}/")
    except Exception as e:
        print(f"    !! 出图失败 (不影响结论, 数据都在 csv 里): {e}")

    print("\n完成。产物: moe_raw.csv / moe_forward.csv / verdict19.json / fig1 / fig2")


if __name__ == "__main__":
    main()