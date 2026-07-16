# -*- coding: utf-8 -*-
"""
================================================================================
脚本 20：数据驱动对照臂 × 跨故障组合外推
          —— 回答"为什么不直接端到端训一个网络 r -> theta?"
================================================================================

【这个脚本回答的问题】

  审稿人/答辩老师必然会问:
    "现在深度学习这么强, 为什么不直接训一个网络, 输入残差 r(t), 输出 theta(t)?"

  本脚本正面回答: 【真的去训几个, 然后看它们在哪里崩】。

【实验设计的灵魂: 双重评测】

  训练集: DS01(HPT_eff) + DS04(fan x2) + DS05(HPC x2) + DS06(LPC) + DS07(LPT x2)
          —— 每个子集只有【单/双部件】故障
  评测1 (分布内):   训练子集的【留出发动机】(按 unit 切, 故障组合见过)
  评测2 (组合外推): DS02/DS03 —— HPT_eff + LPT_eff + LPT_flow【三部件同时退化】,
                    这个组合在训练中【从未出现过】, 且发动机也从未见过。

  核心报告量:  泛化落差 = 技能分(分布内) - 技能分(组合外推)

【为什么物理方法应该赢在外推】

  H 的线性叠加性是被物理保证的 (小扰动下各部件对残差的贡献线性可加),
  你的数据已经证实: DS02 三部件纯外推, 重度退化样本余弦相似度中位数 0.989。
  而神经网络学的是训练集里出现过的 r->theta 映射; "HPT+LPT 同时退化"这个
  组合它从没见过, 它【没有任何机制】知道两个故障的响应应该线性叠加。

【E_Lin 是关键控制组 —— 别小看这个最土的臂】

  E_Lin = 普通最小二乘学一个 13->9 的线性映射 (本质是"学出来的伪逆")。
  - 如果 E_Lin 外推得【好】而非线性网络外推得【差】:
      结论锐化为 "能外推的是【线性性】本身" —— 这正是 H 的物理内涵,
      也解释了为什么物理结构 (线性 H + 约束) 是对的。
  - 如果 E_Lin 外推也差:
      说明问题不止在非线性, 还在【没有约束/没有病态性处理】。
  两种情况都有明确的写法, 不存在"白跑"。

--------------------------------------------------------------------------------
【预注册判据（跑之前写死，跑完照着对，不许改）】

  E1【分布内的公平性】(前提检查, 不是胜负判据)
      在分布内留出单元上, 最好的数据驱动臂技能分 >= D - 0.05。
      → 不成立: 说明数据驱动臂没训好 (欠拟合/超参烂), 那么它们在外推上输了
        也不能算数 —— 必须先修训练再谈结论。⚠️ 这一条保护的是【对手】,
        防止我们打一个稻草人。

  E2【组合外推的胜负】★ 主判据
      在 DS02 和 DS03 上, D 的技能分 > 【每一个】数据驱动臂 (逐臂全胜)。
      → 成立: "数据驱动方法在故障组合的插值区间内有效, 但在组合外推时失效;
               物理结构化方法的泛化能力来自线性叠加原理本身。"
      → 不成立: 如实报告哪个臂赢了、赢多少。P1(约束占优)依然立着,
                但"物理必须进结构"要收缩为"物理结构提供精度与特异性优势"。

  E3【泛化落差】
      每个【非线性】数据驱动臂 (MLP/MixLinear/WPMixer) 的泛化落差
      > D 的泛化落差 + 0.05  (两个子集平均意义下)。
      → 这是"外推失效"的定量形式: 不光要输, 还要输在【落差】上
        (排除"它本来就弱"的解释)。

  E4【线性控制组】(描述性, 不作通过/失败判定)
      对比 E_Lin 与非线性臂的泛化落差。若 E_Lin 落差显著更小 ->
      "可外推的成分是线性映射; 非线性容量在组合外推时是负资产"。

  ⚠️ 全零哨兵在每个表里。⚠️ D 的 lambda 固定 31.6 (脚本18 已证成), 不再调 ——
     数据驱动臂没有 lambda, 所以这里不做逐点占优, 做的是【任务级】对比。

--------------------------------------------------------------------------------
运行:  python 20_datadriven_arms_extrapolation.py
       先 FAST_MODE=True 跑通 (几分钟), 再改 False 跑正式版。
依赖:  h5py numpy pandas scipy scikit-learn matplotlib torch
       (可选 pywt: 没有时 WPMixer 自动退化为手写 Haar 小波, 不影响运行)
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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False   # SimHei 没有 U+2212 字形

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("需要 PyTorch: pip install torch --index-url "
                     "https://download.pytorch.org/whl/cpu")

try:
    import pywt                      # WPMixer 的小波包分解
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False                 # 自动退化为手写 Haar (见 haar_wp)


# ============================================================================
# 配置 —— 数据管线部分与脚本 16~19 【完全一致】
# ============================================================================
DATA_DIR = "D:/N-CMPASS/data_set".strip()
OUT_DIR  = "./dd_out"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

FAST_MODE = True            # ★ 先 True 跑通, 再 False 跑正式版

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

# 分布内评测: 从这两个训练子集里各留出几台发动机 (dev split 里 unit 靠后的)
#   选 DS05(HPC) 和 DS07(LPT): 一个和验证故障无关, 一个部分相关, 覆盖两种情形
INDIST_FILES = {
    "DS05_id": ("N-CMAPSS_DS05.h5", ["HPC_eff_mod", "HPC_flow_mod"]),
    "DS07_id": ("N-CMAPSS_DS07.h5", ["LPT_eff_mod", "LPT_flow_mod"]),
}
N_INDIST_UNITS = 2          # 每个子集留出几台做分布内评测 (这些 unit 不进训练)

THETA_SPAN = {
    "fan_eff_mod":  0.223446, "fan_flow_mod": 0.121209, "LPC_eff_mod":  0.118767,
    "HPC_flow_mod": 0.070658, "LPC_flow_mod": 0.046187, "LPT_eff_mod":  0.036194,
    "LPT_flow_mod": 0.032908, "HPC_eff_mod":  0.025540, "HPT_eff_mod":  0.018668,
}
THETA9 = ["HPT_eff_mod", "fan_eff_mod", "fan_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
          "LPT_eff_mod", "LPT_flow_mod", "LPC_eff_mod", "LPC_flow_mod"]
OP_COLS = ["TRA", "Mach", "theta_c", "delta_c"]

WINDOWS       = [1, 10, 100, 1000] if not FAST_MODE else [10, 1000]
MAX_PER_CYCLE = 1000
N_REF_CYCLES  = 3
LAM_MAIN      = 31.6        # D 的 λ, 脚本18 已证成 (oracle 且可代理标定), 不再调
MAX_TEST_UNITS = None if not FAST_MODE else 2

# ---- 数据驱动臂的超参 ----
SEQ_LEN  = 16               # 时序臂 (MixLinear/WPMixer) 的输入窗口: 过去 L 个 cycle
HIDDEN   = 64
EPOCHS   = 40 if not FAST_MODE else 8
BATCH    = 256
LR       = 1e-3
WP_LEVEL = 2                # 小波包分解层数 -> 2^2 = 4 个子带

# ---- 预注册判据阈值 (写死) ----
E1_FAIRNESS_TOL = 0.05      # 分布内: 最好的数据驱动臂 >= D - 0.05
E3_GAP_MARGIN   = 0.05      # 泛化落差: 非线性臂落差 > D 落差 + 0.05

NONLIN_ARMS = ["E_MLP", "E_MixLinear", "E_WPMixer"]
DD_ARMS     = ["E_Lin"] + NONLIN_ARMS


# ============================================================================
# 数据读取 —— 与脚本 16~19 逐字一致的部分
# ============================================================================
def get_names(f, key):
    """h5 里列名存成字节串, 必须 decode"""
    arr = np.array(f[key]).ravel()
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def load_for_ident(path):
    with h5py.File(path, "r") as f:
        A_var, W_var = get_names(f, "A_var"), get_names(f, "W_var")
        Xs_var, T_var = get_names(f, "X_s_var"), get_names(f, "T_var")
        df_A = pd.DataFrame(np.array(f["A_dev"]), columns=A_var)
        ih  = np.where(df_A["hs"].values == 1)[0]
        idg = np.where(df_A["hs"].values == 0)[0]
        take = np.sort(np.concatenate([
            rng.choice(ih,  min(N_HEALTHY_ID,  len(ih)),  replace=False),
            rng.choice(idg, min(N_DEGRADED_ID, len(idg)), replace=False)]))
        W, Xs, T = f["W_dev"][take, :], f["X_s_dev"][take, :], f["T_dev"][take, :]
    return pd.concat([df_A.iloc[take].reset_index(drop=True),
                      pd.DataFrame(W,  columns=W_var),
                      pd.DataFrame(Xs, columns=Xs_var),
                      pd.DataFrame(T,  columns=T_var)], axis=1)


def load_per_cycle(path, split, only_units=None):
    """按 (unit, cycle) 分组抽样; only_units 可以只读指定的发动机 (省内存)"""
    with h5py.File(path, "r") as f:
        A_var, W_var = get_names(f, "A_var"), get_names(f, "W_var")
        Xs_var, T_var = get_names(f, "X_s_var"), get_names(f, "T_var")
        df_A = pd.DataFrame(np.array(f[f"A_{split}"]), columns=A_var)
        take = []
        for (u, c), idx in df_A.groupby(["unit", "cycle"]).indices.items():
            if only_units is not None and u not in only_units:
                continue
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
    for c in ["Nf", "Nc"]:
        df[f"{c}_c"] = df[c] / sq; cols.append(f"{c}_c")
    df["Wf_c"] = df["Wf"] / (df["delta_c"] * sq); cols.append("Wf_c")
    for c in ["T24", "T30", "T48", "T50"]:
        df[f"{c}_c"] = df[c] / df["T2"]; cols.append(f"{c}_c")
    for c in ["P15", "P21", "P24", "Ps30", "P40", "P50"]:
        df[f"{c}_c"] = df[c] / df["P2"]; cols.append(f"{c}_c")
    return df, cols


# ============================================================================
# 物理反演器 D (与脚本18 逐字一致)
# ============================================================================
def build_cum(T, n):
    Cum = np.zeros((T * n, T * n))
    for t in range(T):
        for i in range(t + 1):
            Cum[t*n:(t+1)*n, i*n:(i+1)*n] = np.eye(n)
    return Cum


def build_M(Hn, T):
    m, n = Hn.shape
    M = np.zeros((T * m, T * n))
    for t in range(T):
        for i in range(t + 1):
            M[t*m:(t+1)*m, i*n:(i+1)*n] = -Hn
    return M


def solve_D(M, Cum, R, lam, T, n):
    """约束+收缩: θ<=0 且单调, 重参数化成 NNLS。M/Cum 在循环外构造。"""
    M_aug = np.vstack([M, np.sqrt(lam) * Cum])
    y_aug = np.concatenate([R.reshape(-1), np.zeros(T * n)])
    d, _ = nnls(M_aug, y_aug, maxiter=50 * T * n)
    return -np.cumsum(d.reshape(T, n), axis=0)


# ============================================================================
# 数据驱动臂
# ============================================================================
class MLPArm(nn.Module):
    """E_MLP: 最朴素的端到端 —— 逐 cycle 把残差 r̄_t (13) 映到 theta_t (9)"""

    def __init__(self, m=13, n=9, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n))

    def forward(self, seq):
        # seq: (B, L, 13) —— 为了和时序臂共用一套训练循环, MLP 只取最后一帧
        return self.net(seq[:, -1, :])


class MixLinearArm(nn.Module):
    """
    E_MixLinear: DLinear/Mix-linear 式的轻量线性时序模型。
      1) 滑动平均把序列分成 趋势 + 残差 两个分量 (退化信号主要活在趋势里)
      2) 每个分量各自做【时间混合】(L->L 的线性层, 各通道共享)
      3) 拼回后做【通道混合】(13->9 的线性层, 各时刻共享), 取最后一帧输出
    全程无激活函数 —— 这是一个纯线性模型 (对输入而言), 但比 E_Lin 多了时间维。
    """

    def __init__(self, L=SEQ_LEN, m=13, n=9, ma=5):
        super().__init__()
        self.ma = ma
        self.time_trend = nn.Linear(L, L)
        self.time_resid = nn.Linear(L, L)
        self.chan = nn.Linear(m, n)

    def forward(self, seq):                      # (B, L, m)
        # 滑动平均 (因果的: 只看过去)。用 avg_pool1d 实现, 左侧复制填充。
        x = seq.transpose(1, 2)                  # (B, m, L)
        pad = F.pad(x, (self.ma - 1, 0), mode="replicate")
        trend = F.avg_pool1d(pad, self.ma, stride=1)      # (B, m, L)
        resid = x - trend
        y = self.time_trend(trend) + self.time_resid(resid)   # 时间混合
        y = y.transpose(1, 2)                    # (B, L, m)
        return self.chan(y[:, -1, :])            # 通道混合, 取最后一帧


def haar_wp(x, level):
    """
    手写 Haar 小波包分解 (pywt 不在时的退化方案)。
    x: (B, m, L) -> 返回 (B, m, 2^level, L/2^level)
    每一层把序列拆成 (低频=相邻均值, 高频=相邻差分) 两半, 递归 level 次。
    """
    bands = [x]
    for _ in range(level):
        nxt = []
        for b in bands:
            lo = (b[..., 0::2] + b[..., 1::2]) / np.sqrt(2.0)
            hi = (b[..., 0::2] - b[..., 1::2]) / np.sqrt(2.0)
            nxt += [lo, hi]
        bands = nxt
    return torch.stack(bands, dim=2)             # (B, m, 2^level, L')


class WPMixerArm(nn.Module):
    """
    E_WPMixer: 小波包 + Mixer (WPMixer 的轻量复刻, 保留其核心归纳偏置:
    "先把序列按频带分解, 再在 (子带 x 通道) 的 token 上做 MLP 混合")。

    ⚠️ 诚实声明 (论文里要写): 这不是原论文的全尺寸实现, 而是保留其
       核心结构 (小波包分解 + token 混合) 的轻量版。参数量对齐到与
       其他臂同一量级, 避免"你把对手做小了"的质疑 —— 训完会打印各臂参数量。
    """

    def __init__(self, L=SEQ_LEN, m=13, n=9, level=WP_LEVEL, hidden=HIDDEN):
        super().__init__()
        self.level = level
        n_band = 2 ** level
        Lp = L // n_band                          # 每个子带剩的长度
        self.token_dim = Lp
        self.n_token = m * n_band                 # (通道 x 子带) 个 token
        self.mix_token = nn.Sequential(           # token 间混合
            nn.Linear(self.n_token, hidden), nn.GELU(),
            nn.Linear(hidden, self.n_token))
        self.mix_dim = nn.Sequential(             # token 内混合
            nn.Linear(Lp, Lp), nn.GELU(),
            nn.Linear(Lp, Lp))
        self.head = nn.Linear(self.n_token * Lp, n)

    def forward(self, seq):                       # (B, L, m)
        x = seq.transpose(1, 2)                   # (B, m, L)
        if HAS_PYWT:
            # pywt 不能进计算图, 但小波包分解是【固定线性变换】, 不需要梯度
            # 穿过它 —— 我们把它当特征提取: 对输入 detach 后在 numpy 里做,
            # 梯度从分解后的系数开始传即可 (系数对网络参数才是变量)。
            # 为了简单和一致, 这里统一用 haar_wp (pywt 的 db1 = Haar, 完全等价)。
            pass
        w = haar_wp(x, self.level)                # (B, m, 2^lv, L')
        B = w.shape[0]
        tok = w.reshape(B, self.n_token, self.token_dim)     # (B, T, D)
        tok = tok + self.mix_dim(tok)                        # token 内
        t2 = tok.transpose(1, 2)                             # (B, D, T)
        t2 = t2 + self.mix_token(t2)                         # token 间
        return self.head(t2.transpose(1, 2).reshape(B, -1))


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 100)
    print("脚本 20: 数据驱动对照臂 x 跨故障组合外推")
    print("=" * 100)
    print(f"\n  FAST_MODE = {FAST_MODE}")
    print(f"  数据驱动臂 = {DD_ARMS}   (E_Lin 是线性控制组)")
    print(f"  物理臂 D 的 lambda = {LAM_MAIN} (脚本18 已证成, 固定不调)")
    print(f"  时序窗口 L = {SEQ_LEN}, 聚合窗口 N = {WINDOWS}")
    print(f"  pywt 可用 = {HAS_PYWT} (不可用时 WPMixer 用手写 Haar, 数学上等价于 db1)")
    print("\n  【预注册判据 (已写死, 跑完不许改)】")
    print(f"    E1 公平性:   分布内, 最好的数据驱动臂 >= D - {E1_FAIRNESS_TOL}")
    print( "                 (保护对手: 不成立则外推结论不能算数, 先修训练)")
    print( "    E2 主判据:   DS02 和 DS03 上, D > 每一个数据驱动臂 (逐臂全胜)")
    print(f"    E3 泛化落差: 每个非线性臂的落差 > D 的落差 + {E3_GAP_MARGIN}")
    print( "    E4 线性控制: 描述性 —— E_Lin 落差 vs 非线性臂落差")

    # ------------------------------------------------------------------
    # [1/6] 健康基准 + H (与脚本16~19 完全一致)
    # ------------------------------------------------------------------
    print("\n\n[1/6] 重建健康基准与影响系数矩阵 H ...")
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

    for fn in data:
        data[fn], corrected_cols = add_corrected(data[fn], ref)
    pool, _ = add_corrected(pool, ref)
    resid_cols = [f"r_{c}" for c in corrected_cols]

    base = {}
    for c in corrected_cols:
        m = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=SEED)
        m.fit(pool[OP_COLS].values, pool[c].values)   # ⚠️ 自变量绝不能加 Nf/Nc
        base[c] = m

    def residual(df):
        r = pd.DataFrame(index=df.index)
        for c in corrected_cols:
            r[f"r_{c}"] = df[c].values - base[c].predict(df[OP_COLS].values)
        return r

    resid_std = residual(pool).std()

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
            if p.startswith("LPC"):
                H_cols.setdefault(p, {})[rc] = lr.coef_[j]

    H = pd.DataFrame(H_cols).reindex(index=resid_cols)[THETA9]
    span = np.array([THETA_SPAN[c] for c in THETA9])
    Hn = H.values * span
    s = np.linalg.svd(Hn, compute_uv=False)
    print(f"    cond(H_n) = {s[0]/s[-1]:.1f}")

    # ------------------------------------------------------------------
    # [2/6] 构造【逐 cycle 序列】训练集 (含噪声增强)
    # ------------------------------------------------------------------
    # 数据驱动臂的输入是 cycle 级的残差序列, 所以训练集也必须是 cycle 级的。
    # 用 load_per_cycle 重新读 (load_for_ident 是随机行抽样, 不保证每个 cycle 完整)。
    #
    # ★ 噪声增强: 每个训练序列随机抽一个 N ∈ WINDOWS, 用 N 个样本的均值当 r̄_t。
    #   -> 一个模型对所有噪声水平都见过, 评测任何 N 都公平。
    #   -> 否则会有质疑: "数据驱动臂只在干净数据(N=1000)上训, 小 N 输了不奇怪"。
    print("\n[2/6] 构造 cycle 级训练序列 (噪声增强) ...")
    del data, pool     # 辨识用的行级数据不再需要, 释放内存

    def unit_cycle_table(df):
        """
        把一台发动机的数据整理成 cycle 级:
          返回 cycles 列表, 以及每个 cycle 的 {残差样本矩阵(k,13), theta(9)}
        零点 = 最初 N_REF_CYCLES 个 cycle (方法论纪律 #5, 不是健康期均值)
        """
        cycles = sorted(df["cycle"].unique())
        m_ref = df["cycle"].isin(cycles[:N_REF_CYCLES])
        b_unit = df.loc[m_ref, resid_cols].mean().values
        th0 = (df.loc[m_ref, THETA9].values / span).mean(axis=0)
        per = []
        for c in cycles:
            sub = df[df["cycle"] == c]
            per.append(dict(
                R_samples=sub[resid_cols].values - b_unit,        # (k,13)
                theta=sub[THETA9].values[0] / span - th0))        # (9,)
        return per

    train_seqs = []        # 每项: (per_cycle 列表, 子集名, unit)  —— 训练用
    indist_eval = {}       # 分布内评测: {tag: [(per, unit), ...]}

    for fn, params in IDENT_FILES.items():
        p = os.path.join(DATA_DIR, fn)
        # 只读退化机队的 dev split; unit 数量有限, 全读 cycle 级太贵 -> 限制台数
        with h5py.File(p, "r") as f:
            A_var = get_names(f, "A_var")
            units_all = np.unique(np.array(f["A_dev"])[:,
                                  A_var.index("unit")]).astype(int)
        n_load = (min(6, len(units_all)) if not FAST_MODE
                  else min(3, len(units_all)))
        use_units = set(units_all[:n_load].tolist())
        d = load_per_cycle(p, "dev", only_units=use_units)
        d, _ = add_corrected(d, ref)
        d = pd.concat([d, residual(d) / resid_std], axis=1)

        units = sorted(d["unit"].unique())
        # 留出规则: DS05/DS07 的【最后 N_INDIST_UNITS 台】做分布内评测, 不进训练
        tag_id = {"N-CMAPSS_DS05.h5": "DS05_id",
                  "N-CMAPSS_DS07.h5": "DS07_id"}.get(fn)
        held = set(units[-N_INDIST_UNITS:]) if tag_id else set()
        for u in units:
            per = unit_cycle_table(d[d["unit"] == u])
            if u in held:
                indist_eval.setdefault(tag_id, []).append((per, int(u)))
            else:
                train_seqs.append((per, fn, int(u)))
        print(f"    {fn}: 训练 {len(units)-len(held)} 台"
              f"{f' / 留出 {len(held)} 台 (分布内评测)' if held else ''}")

    def sample_epoch(seqs):
        """
        每个 epoch 重新采样一次训练张量 (噪声增强的实现):
        对每台机、每个 cycle 位置 t, 随机抽 N, 取 N 个样本均值组成 r̄ 序列,
        再切成 (SEQ_LEN, 13) -> theta_t 的监督对。序列开头零填充
        (早期 theta≈0、残差≈0, 零填充是自然的)。
        """
        X, Y = [], []
        for per, _, _ in seqs:
            Tn = len(per)
            N = int(rng.choice(WINDOWS))
            rbar = np.zeros((Tn, len(resid_cols)), dtype=np.float32)
            for t, pc in enumerate(per):
                k = min(N, len(pc["R_samples"]))
                idx = rng.choice(len(pc["R_samples"]), k, replace=False)
                rbar[t] = pc["R_samples"][idx].mean(axis=0)
            th = np.array([pc["theta"] for pc in per], dtype=np.float32)
            padded = np.vstack([np.zeros((SEQ_LEN - 1, rbar.shape[1]),
                                         dtype=np.float32), rbar])
            for t in range(Tn):
                X.append(padded[t:t + SEQ_LEN])
                Y.append(th[t])
        return (torch.tensor(np.array(X)), torch.tensor(np.array(Y)))

    # ------------------------------------------------------------------
    # [3/6] 训练数据驱动臂
    # ------------------------------------------------------------------
    print("\n[3/6] 训练数据驱动臂 ...")
    # 从训练机队再切一个小验证集 (按 unit!) 用于早停
    rng.shuffle(train_seqs)
    n_val = max(1, len(train_seqs) // 5)
    val_seqs, tr_seqs = train_seqs[:n_val], train_seqs[n_val:]
    print(f"    训练 {len(tr_seqs)} 台 / 早停验证 {len(val_seqs)} 台 (按发动机切分)")

    nets = {"E_MLP": MLPArm(), "E_MixLinear": MixLinearArm(),
            "E_WPMixer": WPMixerArm()}
    for name, net in nets.items():
        n_par = sum(p.numel() for p in net.parameters())
        print(f"    {name:<12s} 参数量 = {n_par}")

    Xva, Yva = sample_epoch(val_seqs)          # 验证集固定一次采样即可
    for name, net in nets.items():
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        best, best_state = 1e9, None
        for ep in range(EPOCHS):
            Xtr, Ytr = sample_epoch(tr_seqs)   # ★ 每个 epoch 重采样 = 噪声增强
            perm = torch.randperm(len(Xtr))
            net.train()
            for i in range(0, len(Xtr), BATCH):
                idx = perm[i:i + BATCH]
                loss = F.mse_loss(net(Xtr[idx]), Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = float(F.mse_loss(net(Xva), Yva))
            if v < best:
                best = v
                best_state = {k: t.detach().clone()
                              for k, t in net.state_dict().items()}
            if ep % max(1, EPOCHS // 4) == 0 or ep == EPOCHS - 1:
                print(f"    [{name:<12s}] ep {ep:>3d}  val MSE = {v:.5f}")
        net.load_state_dict(best_state)

    # ---- E_Lin: 线性控制组, 普通最小二乘 r̄ -> theta ("学出来的伪逆") ----
    Xtr, Ytr = sample_epoch(tr_seqs)
    A_lin = Xtr[:, -1, :].numpy()              # 只用当前帧 (和 E_MLP 输入一致)
    A_lin = np.hstack([A_lin, np.ones((len(A_lin), 1))])       # 带截距
    W_lin, *_ = np.linalg.lstsq(A_lin, Ytr.numpy(), rcond=None)
    print(f"    [E_Lin      ] 最小二乘解完成 (13+1 -> 9), 这是'学出来的伪逆'")

    def predict_dd(arm, rbar_seq):
        """rbar_seq: (T,13) 的 cycle 均值残差 -> (T,9) 的 theta 预测"""
        Tn = rbar_seq.shape[0]
        if arm == "E_Lin":
            A = np.hstack([rbar_seq, np.ones((Tn, 1))])
            return (A @ W_lin).astype(np.float64)
        padded = np.vstack([np.zeros((SEQ_LEN - 1, rbar_seq.shape[1]),
                                     dtype=np.float32),
                            rbar_seq.astype(np.float32)])
        X = np.stack([padded[t:t + SEQ_LEN] for t in range(Tn)])
        net = nets[arm]; net.eval()
        with torch.no_grad():
            return net(torch.tensor(X)).numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # [4/6] 评测 —— 分布内 (留出单元) + 组合外推 (DS02/DS03)
    # ------------------------------------------------------------------
    print("\n[4/6] 评测 ...")
    rows = []

    def eval_unit(per, tag, unit, true_faults, regime):
        """对一台发动机, 在每个 N 下评所有臂"""
        idx_tf = [THETA9.index(q) for q in true_faults]
        other  = [i for i in range(9) if i not in idx_tf]
        Tn = len(per)
        TH = np.array([pc["theta"] for pc in per])
        Cum = build_cum(Tn, 9)
        M   = build_M(Hn, Tn)
        n_early = max(1, int(0.3 * Tn))
        for N in WINDOWS:
            rbar = np.zeros((Tn, len(resid_cols)))
            for t, pc in enumerate(per):
                k = min(N, len(pc["R_samples"]))
                idx = rng.choice(len(pc["R_samples"]), k, replace=False)
                rbar[t] = pc["R_samples"][idx].mean(axis=0)
            mse_zero = float(np.mean(TH ** 2))
            rows.append(dict(regime=regime, subset=tag, unit=unit, N=N,
                             arm="Z_zero", skill=0.0,
                             rmse_early=float(np.sqrt(np.mean(TH[:n_early]**2))),
                             false_alarm=0.0, detect_corr=0.0))
            preds = {arm: predict_dd(arm, rbar) for arm in DD_ARMS}
            preds["D"] = solve_D(M, Cum, rbar, LAM_MAIN, Tn, 9)
            for arm, TH_hat in preds.items():
                err = TH_hat - TH
                mse = float(np.mean(err ** 2))
                skill = 1.0 - mse / max(mse_zero, 1e-12)   # 0=哨兵线
                cs = [np.corrcoef(TH_hat[:, i], TH[:, i])[0, 1]
                      for i in idx_tf
                      if np.std(TH[:, i]) > 1e-9 and np.std(TH_hat[:, i]) > 1e-9]
                rows.append(dict(
                    regime=regime, subset=tag, unit=unit, N=N, arm=arm,
                    skill=skill,
                    rmse_early=float(np.sqrt(np.mean(err[:n_early] ** 2))),
                    false_alarm=float(np.mean(np.abs(TH_hat[:, other]))),
                    detect_corr=float(np.mean(cs)) if cs else 0.0))

    # -- 分布内 --
    for tag, lst in indist_eval.items():
        faults = INDIST_FILES[tag][1]
        for per, u in lst:
            eval_unit(per, tag, u, faults, regime="in_dist")
        print(f"    分布内 {tag}: {len(lst)} 台完成")

    # -- 组合外推 --
    for tag, (fn, true_faults) in VALID_FILES.items():
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"    !! 缺 {fn}, 跳过 {tag}")
            continue
        d = load_per_cycle(p, "test")
        d, _ = add_corrected(d, ref)
        d = pd.concat([d, residual(d) / resid_std], axis=1)
        units = sorted(d["unit"].unique())
        if MAX_TEST_UNITS is not None:
            units = units[:MAX_TEST_UNITS]
        for u in units:
            eval_unit(unit_cycle_table(d[d["unit"] == u]),
                      tag, int(u), true_faults, regime="ood_combo")
        print(f"    组合外推 {tag}: {len(units)} 台完成")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "dd_raw.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  原始结果已存: dd_raw.csv ({len(df)} 行)")

    # ------------------------------------------------------------------
    # [5/6] 裁决 (严格按预注册判据)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("[5/6] 裁  决 (严格按预注册判据)")
    print("=" * 100)

    ALL_ARMS = ["D"] + DD_ARMS
    piv = df[df.arm != "Z_zero"].pivot_table(
        index=["regime", "subset"], columns="arm", values="skill", aggfunc="mean")
    print("\n  技能分总表 (0 = 全零哨兵线):")
    print(piv.round(4).to_string())

    # 分布内/外推的臂级平均 (跨子集)
    sk_in  = piv.loc["in_dist"].mean(axis=0)   if "in_dist"   in piv.index else None
    sk_ood = piv.loc["ood_combo"].mean(axis=0) if "ood_combo" in piv.index else None

    # ---- E1 公平性 ----
    print("\n  --- E1: 分布内的公平性 (保护对手, 防止打稻草人) ---")
    best_dd_in = max(sk_in[a] for a in DD_ARMS)
    E1 = best_dd_in >= sk_in["D"] - E1_FAIRNESS_TOL
    print(f"    分布内: D = {sk_in['D']:.4f} / 最好的数据驱动臂 = {best_dd_in:.4f}"
          f"   (阈值 >= D - {E1_FAIRNESS_TOL})  {'[OK]' if E1 else '[X]'}")
    if not E1:
        print("    !! 数据驱动臂在分布内都没训到位 -> 外推结论【不能算数】,")
        print("       先修训练 (加 EPOCHS / 调 LR / 查数据), 再重跑本脚本。")

    # ---- E2 主判据 ----
    print("\n  --- E2: 组合外推, D 是否逐臂全胜 (主判据) ---")
    E2_ok = []
    for tg in [t for t in piv.index.get_level_values(1).unique()
               if ("ood_combo", t) in piv.index]:
        row = piv.loc[("ood_combo", tg)]
        for arm in DD_ARMS:
            ok = row["D"] > row[arm]
            E2_ok.append(ok)
            print(f"    {tg}: D({row['D']:.4f}) vs {arm}({row[arm]:.4f})"
                  f"   {'[OK]' if ok else '[X] 数据驱动臂赢了'}")
    E2 = all(E2_ok) and len(E2_ok) > 0

    # ---- E3 泛化落差 ----
    print("\n  --- E3: 泛化落差 (分布内技能分 - 外推技能分) ---")
    gap = {a: float(sk_in[a] - sk_ood[a]) for a in ALL_ARMS}
    print(f"    {'臂':<14s} {'分布内':>9s} {'组合外推':>9s} {'落差':>9s}")
    print("    " + "-" * 46)
    for a in ALL_ARMS:
        print(f"    {a:<14s} {sk_in[a]:>9.4f} {sk_ood[a]:>9.4f} {gap[a]:>+9.4f}")
    E3_ok = [gap[a] > gap["D"] + E3_GAP_MARGIN for a in NONLIN_ARMS]
    E3 = all(E3_ok)
    for a, ok in zip(NONLIN_ARMS, E3_ok):
        print(f"    {a}: 落差 {gap[a]:+.4f} vs D 落差 {gap['D']:+.4f} + {E3_GAP_MARGIN}"
              f"   {'[OK]' if ok else '[X]'}")

    # ---- E4 线性控制组 (描述性) ----
    print("\n  --- E4: 线性控制组 (描述性, 无通过/失败) ---")
    nl_gap = float(np.mean([gap[a] for a in NONLIN_ARMS]))
    print(f"    E_Lin 落差 = {gap['E_Lin']:+.4f}   非线性臂平均落差 = {nl_gap:+.4f}")
    if gap["E_Lin"] < nl_gap - 0.03:
        print("    => E_Lin 落差显著更小: 【可外推的成分是线性映射】,")
        print("       非线性容量在组合外推时是负资产。这锐化了物理结构的论证:")
        print("       H 的价值 = 线性叠加原理 + 病态性处理 + 可行域约束。")
    elif gap["E_Lin"] > nl_gap + 0.03:
        print("    => E_Lin 落差反而更大: 线性性不是外推的关键, 结论要重想。")
    else:
        print("    => E_Lin 与非线性臂落差相近, 线性/非线性之分不是主因,")
        print("       外推优势更多来自【约束与结构】而非线性性本身。")

    # ---- 虚警对比 (次要, 呼应 P3) ----
    print("\n  --- 附: 组合外推时的虚警幅度 (呼应脚本18 的 P3) ---")
    fa = df[(df.regime == "ood_combo") & (df.arm != "Z_zero")].pivot_table(
        index="subset", columns="arm", values="false_alarm", aggfunc="mean")
    print(fa.round(4).to_string())

    verdict = {"E1": bool(E1), "E2": bool(E2), "E3": bool(E3),
               "skill_table": piv.round(4).reset_index().to_dict(orient="records"),
               "gaps": {k: round(v, 4) for k, v in gap.items()},
               "false_alarm_ood": fa.round(4).reset_index().to_dict(orient="records")}
    with open(os.path.join(OUT_DIR, "verdict20.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)

    print("\n\n" + "=" * 100)
    if E1 and E2 and E3:
        print("[PASS] 全部判据通过。可以写:")
        print("  '数据驱动方法在故障组合的插值区间内有效, 但在组合外推时失效;")
        print("   物理结构化方法的泛化能力来自线性叠加原理本身, 而非训练数据的覆盖度。")
        print("   这正是在故障样本天然稀缺的航空发动机场景下, 物理必须进入模型结构的原因。'")
    else:
        print("[MIXED] 部分判据未通过 —— 逐条处理, 不许粉饰:")
        if not E1:
            print("  [X] E1: 对手没训好, 外推结论作废, 先修训练。")
        if not E2:
            print("  [X] E2: 有数据驱动臂在组合外推上打赢了 D。如实报告是哪个、赢多少。")
            print("      检查它是不是 E_Lin —— 若是, 结论改写为'线性映射可外推,")
            print("      物理结构的净价值在约束(虚警/单调性)而非线性性', 看附表虚警。")
        if not E3:
            print("  [X] E3: 非线性臂的落差没有显著大于 D。'组合外推失效'的定量主张")
            print("      不成立, 从论文中删除该表述, 保留 E2 的直接对比。")
    print("=" * 100)

    # ------------------------------------------------------------------
    # [6/6] 出图: 分布内 vs 组合外推 的成对条形图 (第3章主图2)
    # ------------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        arms_plot = ALL_ARMS
        x = np.arange(len(arms_plot))
        wdt = 0.38
        v_in  = [sk_in[a]  for a in arms_plot]
        v_ood = [sk_ood[a] for a in arms_plot]
        ax.bar(x - wdt/2, v_in,  wdt, label="分布内 (留出发动机, 故障组合见过)")
        ax.bar(x + wdt/2, v_ood, wdt, label="组合外推 (DS02/03, 三部件同时退化, 从未见过)")
        ax.axhline(0, color="k", lw=1)
        ax.axhline(0, color="gray", ls=":", lw=1)
        for i, a in enumerate(arms_plot):
            ax.annotate(f"{gap[a]:+.2f}", (x[i], max(v_in[i], 0) + 0.03),
                        ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(arms_plot)
        ax.set_ylabel("技能分 S  (0 = 全零哨兵)")
        ax.set_title("脚本20 主图: 谁能外推到从未见过的故障组合?\n(柱顶数字 = 泛化落差, 越小越好)")
        ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig1_extrapolation.png"), dpi=150)
        plt.close(fig)
        print(f"\n[6/6] 图已存到 {OUT_DIR}/fig1_extrapolation.png")
    except Exception as e:
        print(f"\n[6/6] 出图失败 (不影响结论, 数据在 csv 里): {e}")

    print("\n完成。产物: dd_raw.csv / verdict20.json / fig1_extrapolation.png")


if __name__ == "__main__":
    main()