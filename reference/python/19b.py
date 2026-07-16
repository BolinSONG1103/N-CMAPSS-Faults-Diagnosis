# -*- coding: utf-8 -*-
"""
================================================================================
脚本 19 v2：工况自适应 H(w) = H0 · diag(g(w))  [MoE 门控]
            —— 双窗口模式评测: 整周期平均 vs 高功率段
================================================================================

【v1 (FAST_MODE 烟测) 查明了什么 —— 本版为什么存在】

  1. ✅ Q3 干净通过: 给了 MoE_dir 改方向的自由, 它自己选择不改
       ||dH||_F/||H0||_F = 0.0031, 扰动后方向余弦最差 0.9999, 技能分反而略降。
     ⇒ "H0 方向冻结"被证成: 方向是部件故障的物理签名, 工况只调制幅值。

  2. ✅ Q5 干净通过: 门控自主重新发现了 (0)-d 的物理规律
       按加权平均 TRA 排序 (49.6 -> 75.1), 专家增益单调递增;
       高功率专家 (TRA≈75, 占比43%) 增益全部 >1, 低功率专家全部 <1。
       TRA4 硬分档臂也复现了 (0)-d: fan_flow 学出 0.50->1.22 (参考 0.59->1.24)。

  3. ❌ Q1/Q2 挂了, 但挂得【有规律】—— 诊断端所有臂挤在 ±0.002 之内。
     机制 (精确的, 不是猜的):
       诊断窗口 = 整周期内随机抽 N 个样本取均值。窗口平均后的等效矩阵是
       H0·diag(g_bar_t), 其中 g_bar_t 是窗口内样本增益的均值。
       每个 cycle 都是一段完整飞行 (爬升+巡航+下降), 工况分布逐 cycle
       几乎相同, 于是:
           g_bar_t ≈ E_整个包线[g(w)] ≈ 常数 (且接近 1)   对所有 t
       ⇒ 工况变化被【整周期平均】抵消了。剩下的只是逐部件百分之几的常数
         幅值偏差, 对技能分的影响量级 (几%)^2 ≈ 1e-3 —— 正好是测到的 gap。
     ⇒ 结论不是"工况自适应没用", 而是:
       【整周期平均的诊断方案对工况变化天然免疫】。
       这本身是一个值得写的发现 —— 它同时解释了为什么全局 H0 在
       之前所有脚本里都够用。

【H(w) 的价值在哪里? 在窗口不覆盖整个包线的时候】

  工程上更真实的场景恰恰如此: 机队监控普遍用【巡航稳态报文】或
  【起飞快照】做趋势分析, 不会等一个整周期的均值。
  如果窗口只取高功率段: g_bar_t ≈ E[g | 高TRA] ≈ 1.1+,
  而 H0 是按全包线混合标定的 -> 系统性 ~10% 的幅值偏差
  -> 这时 H(w) 才有活干。

  ⇒ 本版把评测改成【双模式】(训练完全不变, 训练是样本级的, 与窗口无关):
       mode="full"    整周期平均 (v1 的做法, 保留作对照)
       mode="highTRA" 每个 cycle 只用 TRA 前 25% 的样本
     两个模式并排, 就是"什么时候需要工况自适应"的完整回答。

  ⚠️ 分段模式的一个致命坑 (v2 已修): 零点 b_unit 必须在【同一段】上计算。
     若零点用全周期算、窗口用高功率段算, 段与全周期的基线错位会制造假信号
     (残差里混进一个与退化无关的常数偏移), 高估 H(w) 的价值。

--------------------------------------------------------------------------------
【预注册判据（跑之前写死，跑完照着对，不许改）】

  Q1 【前向有效性, 样本级】 (与 v1 相同, 样本级不受窗口平均影响)
      验证集上 MoE 的前向 NRMSE 相对 H0 降低 >= 5%。
      (v1 烟测: 训练验证集上降了 8.8%, 但那是 6 epoch; 正式版重新裁决)

  Q2f【整周期模式, 描述性 —— 机制的验证, 不作胜负判定】
      预测: full 模式下 MoE-H0 的中位 gap 量级 ~1e-3 (即"被平均洗掉")。
      如实报告数字。若 gap 反而 >= 0.01, 说明上面的机制解释是错的, 要重想。

  Q2s【分段模式的逐点占优】★ 本版主判据
      highTRA 模式下, 所有 (子集 x N x lambda) 网格点上:
        (a) MoE > H0 的占优率 >= 95%
        (b) gap 的中位数 >= 0.01
      → 成立可写: "当诊断窗口不覆盖整个飞行包线时 (工程常态),
        工况自适应带来一个数量级于 0.01 技能分的一致增益,
        且该增益在整条正则化路径上存在, 不由 lambda 的选择产生。"

  Q3 【H0 冻结的合法性】 (同 v1, 在两个模式的主 lambda 上都要成立)
      MoE_dir 相对 MoE 的额外技能分增益 < 0.01。

  Q4 【MoE vs 自由 MLP】 (同 v1, 主 lambda, 两个模式)
      MoE >= FreeMLP - 0.005。

  Q6 【TRA4 硬分档对照 —— 防止高估 MoE, 本版新增】
      highTRA 模式主 lambda 上, 比较 MoE 与 TRA4:
        若 MoE - TRA4 < 0.005 (两个子集): 诚实结论是
          "按段重标定 (硬分档) 即可吃掉大部分收益, 连续门控的增量价值有限;
           MoE 的独特卖点是【一个模型服务所有段】+ 可解释分区"。
        若 MoE - TRA4 >= 0.005: 连续门控本身有超出分档的价值, 如实报告。
      (两种结果都有明确写法, 不存在白跑。)

  Q5 【门控物理性】 (同 v1, 描述性): 门控分区 + 增益单调性 vs (0)-d。

  ⚠️ 全零哨兵在每个表里。⚠️ 图注按数据说话, 不写"始终在0线之上"这类
     先于数据的断言 (v1 的 fig2 犯了这个错, 本版删掉)。

--------------------------------------------------------------------------------
运行:  python 19b_moe_dual_window.py
       先 FAST_MODE=True 跑通 (几分钟), 再改 False 跑正式版。
耗时:  正式版 NNLS 次数 = v1 的 2 倍 (两个窗口模式)。
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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False   # SimHei 没有 U+2212 字形

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("需要 PyTorch: pip install torch --index-url "
                     "https://download.pytorch.org/whl/cpu")


# ============================================================================
# 配置 —— 数据管线部分与脚本 16~19 【完全一致】
# ============================================================================
DATA_DIR = "D:/N-CMPASS/data_set".strip()
OUT_DIR  = "./moe2_out"
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

THETA_SPAN = {
    "fan_eff_mod":  0.223446, "fan_flow_mod": 0.121209, "LPC_eff_mod":  0.118767,
    "HPC_flow_mod": 0.070658, "LPC_flow_mod": 0.046187, "LPT_eff_mod":  0.036194,
    "LPT_flow_mod": 0.032908, "HPC_eff_mod":  0.025540, "HPT_eff_mod":  0.018668,
}
THETA9 = ["HPT_eff_mod", "fan_eff_mod", "fan_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
          "LPT_eff_mod", "LPT_flow_mod", "LPC_eff_mod", "LPC_flow_mod"]

# ★ 门控输入 = 第2章健康基准的自变量, 完全一致 (信息泄漏纪律)
OP_COLS = ["TRA", "Mach", "theta_c", "delta_c"]

WINDOWS       = [1, 10, 100, 1000] if not FAST_MODE else [100, 1000]
MAX_PER_CYCLE = 1000
N_REF_CYCLES  = 3
LAMBDA_GRID   = np.logspace(-1, 3, 9) if not FAST_MODE else np.logspace(0, 2, 5)
LAM_MAIN      = 31.6        # 脚本17/18 的 oracle λ
MAX_TEST_UNITS = None if not FAST_MODE else 2

# ---- ★ 双窗口模式 ----
WINDOW_MODES = ["full", "highTRA"]
SEG_Q        = 0.75         # highTRA: 每个 cycle 只用 TRA >= 该 cycle 75 分位的样本
SEG_MIN      = 5            # 段内可用样本少于这个数就退回全周期 (并计数报告)

# ---- MoE 超参 (与 v1 一致) ----
K_MAIN   = 4
K_ABLATE = [1, 2, 4, 8] if not FAST_MODE else [1, 4]
HIDDEN   = 32
EPOCHS   = 30 if not FAST_MODE else 6
BATCH    = 4096
LR       = 3e-3
N_TRAIN_MAX = 300000 if not FAST_MODE else 60000
DIR_PENALTY = 1e-3
SOFTPLUS_INV_1 = float(np.log(np.e - 1.0))

# ---- 预注册判据阈值 (写死) ----
Q1_FWD_GAIN_MIN  = 0.05     # 样本级前向 NRMSE 至少降 5%
Q2S_DOMINANCE    = 0.95     # 分段模式: 占优率 >= 95%
Q2S_MEDGAP_MIN   = 0.01     # 分段模式: 中位 gap >= 0.01
Q3_DIR_GAIN_MAX  = 0.01
Q4_MOE_TOL       = 0.005
Q6_MOE_VS_TRA4   = 0.005    # MoE 相对硬分档的增量价值判别线

TRA_BIN_REF = {
    "fan_flow_mod": (0.59, 1.24), "LPT_eff_mod": (0.66, 1.26),
    "fan_eff_mod":  (0.66, 1.21), "HPT_eff_mod": (0.91, 1.10),
}


# ============================================================================
# 数据读取 (与脚本 16~19 逐字一致)
# ============================================================================
def get_names(f, key):
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


def load_per_cycle(path, split):
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
    for c in ["Nf", "Nc"]:
        df[f"{c}_c"] = df[c] / sq; cols.append(f"{c}_c")
    df["Wf_c"] = df["Wf"] / (df["delta_c"] * sq); cols.append("Wf_c")
    for c in ["T24", "T30", "T48", "T50"]:
        df[f"{c}_c"] = df[c] / df["T2"]; cols.append(f"{c}_c")
    for c in ["P15", "P21", "P24", "Ps30", "P40", "P50"]:
        df[f"{c}_c"] = df[c] / df["P2"]; cols.append(f"{c}_c")
    return df, cols


# ============================================================================
# MoE 门控 (与 v1 相同)
# ============================================================================
class GainNet(nn.Module):
    """mode='moe'/'free'/'const'; learn_dir=True 时额外学 13x9 的 dH (挑战冻结假设)"""

    def __init__(self, K, mode="moe", learn_dir=False, m=13, n=9,
                 n_in=4, hidden=HIDDEN):
        super().__init__()
        self.mode, self.K, self.m, self.n = mode, K, m, n
        self.learn_dir = learn_dir
        if mode == "moe":
            self.gate = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                      nn.Linear(hidden, K))
            self.experts_raw = nn.Parameter(torch.full((K, n), SOFTPLUS_INV_1))
        elif mode == "free":
            self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                     nn.Linear(hidden, n))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, SOFTPLUS_INV_1)
        self.bias = nn.Parameter(torch.zeros(m))
        if learn_dir:
            self.dH = nn.Parameter(torch.zeros(m, n))

    def gains(self, w):
        if self.mode == "const":
            return torch.ones(w.shape[0], self.n, device=w.device)
        if self.mode == "free":
            return F.softplus(self.net(w))
        a = torch.softmax(self.gate(w), dim=-1)
        s = F.softplus(self.experts_raw)
        return a @ s

    def alphas(self, w):
        return torch.softmax(self.gate(w), dim=-1) if self.mode == "moe" else None

    def H_eff(self, Hn):
        return Hn + self.dH if self.learn_dir else Hn

    def forward(self, w, th_n, Hn):
        g = self.gains(w)
        return (g * th_n) @ self.H_eff(Hn).T + self.bias


def fit_gain_ls(Hn, TH_n, R, mask=None):
    """TRA 硬分档臂: 在样本子集上最小二乘解常数增益 g (对 g 线性, 正规方程)"""
    if mask is not None:
        TH_n, R = TH_n[mask], R[mask]
    m, n = Hn.shape
    A = (TH_n[:, None, :] * Hn[None, :, :]).reshape(-1, n)
    y = R.reshape(-1)
    AtA = A.T @ A
    dead = np.diag(AtA) < 1e-8 * max(np.diag(AtA).max(), 1e-12)
    g = np.ones(n)
    ridge = 1e-6 * np.trace(AtA) / n
    try:
        g = np.linalg.solve(AtA + ridge * np.eye(n), A.T @ y)
    except np.linalg.LinAlgError:
        pass
    g[dead] = 1.0
    return np.clip(g, 0.05, 5.0)


# ============================================================================
# 反演器 (支持逐 cycle 不同的 H; 与 v1 相同)
# ============================================================================
def build_cum(T, n):
    Cum = np.zeros((T * n, T * n))
    for t in range(T):
        for i in range(t + 1):
            Cum[t*n:(t+1)*n, i*n:(i+1)*n] = np.eye(n)
    return Cum


def build_M_var(Hn_list, T):
    """r(t) = Hn_t·θ(t): 设计矩阵 (t,i) 块 (i<=t) = -Hn_t (下标是 t 不是 i)"""
    m, n = Hn_list[0].shape
    M = np.zeros((T * m, T * n))
    for t in range(T):
        Ht = -Hn_list[t]
        for i in range(t + 1):
            M[t*m:(t+1)*m, i*n:(i+1)*n] = Ht
    return M


def solve_D_var(M, Cum, R, lam, T, n):
    M_aug = np.vstack([M, np.sqrt(lam) * Cum])
    y_aug = np.concatenate([R.reshape(-1), np.zeros(T * n)])
    d, _ = nnls(M_aug, y_aug, maxiter=50 * T * n)
    return -np.cumsum(d.reshape(T, n), axis=0)


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 100)
    print("脚本 19 v2: 工况自适应 H(w)  [MoE 门控]  —— 双窗口模式评测")
    print("=" * 100)
    print(f"\n  FAST_MODE = {FAST_MODE}")
    print(f"  窗口模式  = {WINDOW_MODES}  (highTRA: 每 cycle 只用 TRA >= 该 cycle "
          f"{SEG_Q:.0%} 分位的样本)")
    print(f"  门控输入  = {OP_COLS}, K = {K_MAIN} (消融 {K_ABLATE})")
    print(f"  lambda 网格 = {len(LAMBDA_GRID)} 点, 主工作点 {LAM_MAIN}; 窗口 N = {WINDOWS}")
    print("\n  【预注册判据 (已写死, 跑完不许改)】")
    print(f"    Q1  样本级前向: MoE 相对 H0 的 NRMSE 降低 >= {Q1_FWD_GAIN_MIN:.0%}")
    print( "    Q2f 整周期模式 (描述性): 预测 MoE-H0 中位 gap ~1e-3 (被平均洗掉);")
    print( "        若 >= 0.01 则机制解释错了, 要重想")
    print(f"    Q2s 分段模式 (主判据): 占优率 >= {Q2S_DOMINANCE:.0%} 且中位 gap >= {Q2S_MEDGAP_MIN}")
    print(f"    Q3  冻结合法: MoE_dir - MoE < {Q3_DIR_GAIN_MAX} (两模式主 lambda)")
    print(f"    Q4  结构代价: MoE >= FreeMLP - {Q4_MOE_TOL} (两模式主 lambda)")
    print(f"    Q6  硬分档对照: MoE - TRA4 与 {Q6_MOE_VS_TRA4} 比较, 两种结果都有写法")
    print( "    Q5  门控物理性 (描述性)")

    # ------------------------------------------------------------------
    # [1/6] 健康基准 + H0 (与 16~19 完全一致)
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
    for fn in data:
        data[fn], corrected_cols = add_corrected(data[fn], ref)
    pool, _ = add_corrected(pool, ref)
    resid_cols = [f"r_{c}" for c in corrected_cols]

    base = {}
    for c in corrected_cols:
        m = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=SEED)
        m.fit(pool[OP_COLS].values, pool[c].values)   # ⚠️ 绝不能加 Nf/Nc
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
    Hn_t = torch.tensor(Hn, dtype=torch.float32)

    # ------------------------------------------------------------------
    # [2/6] 门控训练集 (样本级正向模型; 与 v1 相同 —— 训练与窗口模式无关)
    # ------------------------------------------------------------------
    print("\n[2/6] 构造门控训练集 ...")
    Ws, THs, Rs, UNITs = [], [], [], []
    for fi, (fn, params) in enumerate(IDENT_FILES.items()):
        df = data[fn]
        df = df[df["hs"] == 0]
        if len(df) == 0:
            continue
        Ws.append(df[OP_COLS].values)
        THs.append(df[THETA9].values / span)
        Rs.append((residual(df) / resid_std).values)
        UNITs.append(df["unit"].values + 1000 * fi)
    W_all  = np.concatenate(Ws).astype(np.float32)
    TH_all = np.concatenate(THs).astype(np.float32)
    R_all  = np.concatenate(Rs).astype(np.float32)
    U_all  = np.concatenate(UNITs)
    del data, pool, Ws, THs, Rs

    if len(W_all) > N_TRAIN_MAX:
        sel = rng.choice(len(W_all), N_TRAIN_MAX, replace=False)
        W_all, TH_all, R_all, U_all = W_all[sel], TH_all[sel], R_all[sel], U_all[sel]

    units = np.unique(U_all)
    rng.shuffle(units)
    n_val = max(1, int(0.25 * len(units)))
    val_units = set(units[:n_val].tolist())
    is_val = np.array([u in val_units for u in U_all])
    print(f"    训练 {(~is_val).sum()} 行 / 验证 {is_val.sum()} 行"
          f"  ({len(units)-n_val} / {n_val} 台, 按发动机切分)")

    w_mu = W_all[~is_val].mean(axis=0)
    w_sd = W_all[~is_val].std(axis=0) + 1e-8
    Wz_all = (W_all - w_mu) / w_sd

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32)

    Wtr, THtr, Rtr = to_t(Wz_all[~is_val]), to_t(TH_all[~is_val]), to_t(R_all[~is_val])
    Wva, THva, Rva = to_t(Wz_all[is_val]),  to_t(TH_all[is_val]),  to_t(R_all[is_val])

    # ------------------------------------------------------------------
    # [3/6] 训练 (与 v1 相同)
    # ------------------------------------------------------------------
    print("\n[3/6] 训练门控 ...")

    @torch.no_grad()
    def eval_forward(net, W, TH, R):
        net.eval()
        pred = net(W, TH, Hn_t)
        return float(torch.norm(pred - R) / (torch.norm(R) + 1e-12))

    def train_arm(name, K, mode, learn_dir=False):
        net = GainNet(K=K, mode=mode, learn_dir=learn_dir)
        if mode == "const" and not learn_dir:
            with torch.no_grad():
                net.bias.copy_((Rtr - THtr @ Hn_t.T).mean(dim=0))
            v = eval_forward(net, Wva, THva, Rva)
            print(f"    [{name:<10s}] (无需训练)  val NRMSE = {v:.4f}")
            return net, v
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        n = len(Wtr)
        best, best_state = 1e9, None
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            net.train()
            for i in range(0, n, BATCH):
                idx = perm[i:i+BATCH]
                loss = F.mse_loss(net(Wtr[idx], THtr[idx], Hn_t), Rtr[idx])
                if learn_dir:
                    loss = loss + DIR_PENALTY * (net.dH ** 2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
            v = eval_forward(net, Wva, THva, Rva)
            if v < best:
                best = v
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            if ep % max(1, EPOCHS // 5) == 0 or ep == EPOCHS - 1:
                print(f"    [{name:<10s}] ep {ep:>3d}  val NRMSE = {v:.4f}")
        net.load_state_dict(best_state)
        print(f"    [{name:<10s}] ==> best val NRMSE = {best:.4f}")
        return net, best

    arms = {}
    arms["H0"], nrmse_H0 = train_arm("H0", 1, "const")
    nrmse_moe = None
    for K in K_ABLATE:
        arms[f"MoE_K{K}"], v = train_arm(f"MoE_K{K}", K, "moe")
        if K == K_MAIN:
            nrmse_moe = v
    arms["FreeMLP"], _ = train_arm("FreeMLP", 1, "free")
    arms["MoE_dir"], _ = train_arm("MoE_dir", K_MAIN, "moe", learn_dir=True)
    MOE = f"MoE_K{K_MAIN}"

    with torch.no_grad():
        dH = arms["MoE_dir"].dH.numpy()
        Hp = Hn + dH
        col_cos = [float(np.dot(Hn[:, j], Hp[:, j]) /
                         (np.linalg.norm(Hn[:, j]) * np.linalg.norm(Hp[:, j]) + 1e-12))
                   for j in range(9)]
    print(f"\n    MoE_dir 扰动: ||dH||_F/||H0||_F = {np.linalg.norm(dH)/np.linalg.norm(Hn):.4f}"
          f", 方向余弦最差 = {min(col_cos):.4f}")

    # TRA 硬分档臂 (Q6 的对照)
    print("\n    [TRA4     ] 手工 TRA 四分位硬分档 (非学习)")
    tra_tr = W_all[~is_val][:, 0]
    tra_q = np.quantile(tra_tr, [0.25, 0.5, 0.75])
    g_bins = np.array([fit_gain_ls(Hn, TH_all[~is_val], R_all[~is_val],
                                   (tra_tr > (-np.inf if b == 0 else tra_q[b-1])) &
                                   (tra_tr <= (np.inf if b == 3 else tra_q[b])))
                       for b in range(4)])
    for j, name in enumerate(THETA9):
        if name in TRA_BIN_REF:
            lo, hi = TRA_BIN_REF[name]
            print(f"      {name:<14s} 学出 {g_bins[0,j]:.2f} -> {g_bins[3,j]:.2f}"
                  f"   | (0)-d 参考 {lo:.2f} -> {hi:.2f}")

    @torch.no_grad()
    def gains_of(arm, W_raw):
        if arm == "TRA4":
            return g_bins[np.digitize(W_raw[:, 0], tra_q)]
        net = arms[arm]; net.eval()
        return net.gains(to_t((W_raw - w_mu) / w_sd)).numpy()

    # ------------------------------------------------------------------
    # [4/6] 双模式评测
    # ------------------------------------------------------------------
    print("\n[4/6] 双窗口模式评测 (full / highTRA) ...")

    seg_fallback = {"full": 0, "highTRA": 0}   # 段内样本不足退回全周期的次数

    def seg_index(sub, mode):
        """返回该 cycle 内可用样本的行号 (相对 sub)。highTRA 段太小则退回全周期。"""
        if mode == "full":
            return np.arange(len(sub))
        thr = sub["TRA"].quantile(SEG_Q)
        idx = np.where(sub["TRA"].values >= thr)[0]
        if len(idx) < SEG_MIN:
            seg_fallback[mode] += 1
            return np.arange(len(sub))
        return idx

    def build_windows(df, unit, mode):
        """
        返回 {N: (R, TH, OPS)}。
        ★ v2 的关键修正: 零点 b_unit 在【同一段】上计算。
          若零点用全周期、窗口用高功率段, 基线错位会制造与退化无关的
          常数偏移 -> 高估 H(w) 的价值。这里参考 cycle 也先过 seg_index。
        """
        du = df[df["unit"] == unit].copy()
        cycles = sorted(du["cycle"].unique())

        ref_rows = []
        for c in cycles[:N_REF_CYCLES]:
            sub = du[du["cycle"] == c]
            ref_rows.append(sub[resid_cols].values[seg_index(sub, mode)])
        b_unit = np.concatenate(ref_rows).mean(axis=0)
        m_ref = du["cycle"].isin(cycles[:N_REF_CYCLES])
        th0 = (du.loc[m_ref, THETA9].values / span).mean(axis=0)   # θ 逐 cycle 常数, 不受段影响

        out = {}
        for N in WINDOWS:
            R_list, TH_list, OP_list = [], [], []
            for c in cycles:
                sub = du[du["cycle"] == c]
                pool_idx = seg_index(sub, mode)
                k = min(N, len(pool_idx))
                idx = rng.choice(pool_idx, k, replace=False)
                R_list.append(sub[resid_cols].values[idx].mean(axis=0) - b_unit)
                TH_list.append(sub[THETA9].values[0] / span - th0)
                OP_list.append(sub[OP_COLS].values[idx])   # ★ 增益均值用同一批样本
            out[N] = (np.array(R_list), np.array(TH_list), OP_list)
        return out

    rows = []
    eval_arms = ["H0", "TRA4", MOE, "FreeMLP", "MoE_dir"]

    for tag, (fn, true_faults) in VALID_FILES.items():
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"  !! 缺 {fn}, 跳过 {tag}")
            continue
        print("\n" + "#" * 100)
        print(f"# {tag}    真实退化部件: {true_faults}")
        print("#" * 100)
        d = load_per_cycle(p, "test")
        d, _ = add_corrected(d, ref)
        d = pd.concat([d, residual(d) / resid_std], axis=1)
        idx_tf = [THETA9.index(q) for q in true_faults]
        other  = [i for i in range(9) if i not in idx_tf]

        test_units = sorted(d["unit"].unique())
        if MAX_TEST_UNITS is not None:
            test_units = test_units[:MAX_TEST_UNITS]
        n_solve = len(test_units) * len(WINDOWS) * len(LAMBDA_GRID) \
                  * len(eval_arms) * len(WINDOW_MODES)
        print(f"  test 单元: {[int(u) for u in test_units]},"
              f" 共需 {n_solve} 次 NNLS, 请耐心 ...")

        for u in test_units:
            for mode in WINDOW_MODES:
                wins = build_windows(d, u, mode)
                for N in WINDOWS:
                    R, TH, OPS = wins[N]
                    T = len(R)
                    Cum = build_cum(T, 9)
                    mse_zero = float(np.mean(TH ** 2))
                    n_early = max(1, int(0.3 * T))
                    rows.append(dict(mode=mode, subset=tag, unit=int(u), N=N,
                                     arm="Z_zero", lam=np.nan, skill=0.0,
                                     rmse_early=float(np.sqrt(np.mean(TH[:n_early]**2))),
                                     false_alarm=0.0, detect_corr=0.0))
                    for arm in eval_arms:
                        Hbase = Hn if arm == "TRA4" else \
                                arms[arm].H_eff(Hn_t).detach().numpy()
                        Hn_list = [Hbase * gains_of(arm, OPS[t]).mean(axis=0)[None, :]
                                   for t in range(T)]
                        M = build_M_var(Hn_list, T)
                        for lam in LAMBDA_GRID:
                            TH_hat = solve_D_var(M, Cum, R, lam, T, 9)
                            err = TH_hat - TH
                            mse = float(np.mean(err ** 2))
                            skill = 1.0 - mse / max(mse_zero, 1e-12)
                            cs = [np.corrcoef(TH_hat[:, i], TH[:, i])[0, 1]
                                  for i in idx_tf
                                  if np.std(TH[:, i]) > 1e-9 and np.std(TH_hat[:, i]) > 1e-9]
                            rows.append(dict(
                                mode=mode, subset=tag, unit=int(u), N=N, arm=arm,
                                lam=float(lam), skill=skill,
                                rmse_early=float(np.sqrt(np.mean(err[:n_early] ** 2))),
                                false_alarm=float(np.mean(np.abs(TH_hat[:, other]))),
                                detect_corr=float(np.mean(cs)) if cs else 0.0))
            print(f"    unit {int(u)} 完成 (两个模式)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "moe2_raw.csv"), index=False, encoding="utf-8-sig")
    print(f"\n  原始结果已存: moe2_raw.csv ({len(df)} 行)")
    print(f"  highTRA 段样本不足退回全周期的 cycle 数: {seg_fallback['highTRA']}"
          f" (若占比高, 分段结论的代表性要打折, 如实报告)")

    # ------------------------------------------------------------------
    # [5/6] 裁决
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("[5/6] 裁  决 (严格按预注册判据)")
    print("=" * 100)

    tags = sorted(df["subset"].unique())
    lam_pick = float(LAMBDA_GRID[np.argmin(np.abs(LAMBDA_GRID - LAM_MAIN))])

    # ---- Q1: 样本级前向 ----
    print("\n  --- Q1: 样本级前向 (验证集) ---")
    gain_fwd = 1.0 - nrmse_moe / nrmse_H0
    Q1 = gain_fwd >= Q1_FWD_GAIN_MIN
    print(f"    H0 = {nrmse_H0:.4f}, MoE = {nrmse_moe:.4f}, 降低 {gain_fwd:+.1%}"
          f"  (阈值 {Q1_FWD_GAIN_MIN:.0%})  {'[OK]' if Q1 else '[X]'}")

    # ---- Q2f / Q2s: 双模式逐点占优 ----
    verdicts_mode = {}
    for mode in WINDOW_MODES:
        sub = df[(df["mode"] == mode) & df.arm.isin([MOE, "H0"])]
        grid = sub.pivot_table(index=["subset", "N", "lam"], columns="arm",
                               values="skill", aggfunc="mean")
        grid["gap"] = grid[MOE] - grid["H0"]
        rate = float((grid["gap"] > 0).mean())
        med  = float(grid["gap"].median())
        verdicts_mode[mode] = (rate, med, grid)
        label = "Q2f (整周期, 描述性)" if mode == "full" else "Q2s (分段, 主判据)"
        print(f"\n  --- {label} ---")
        print(f"    {'子集':<8s}{'N':>6s}{'点数':>6s}{'占优':>6s}{'占优率':>8s}"
              f"{'gap中位':>12s}{'gap最小':>12s}")
        for (tg, N), g in grid.groupby(level=[0, 1]):
            w = int((g['gap'] > 0).sum())
            print(f"    {tg:<8s}{N:>6d}{len(g):>6d}{w:>6d}{w/len(g):>7.1%}"
                  f"{g['gap'].median():>+12.4f}{g['gap'].min():>+12.4f}")
        print(f"    总计: 占优率 {rate:.1%}, 中位 gap {med:+.4f}")

    rate_f, med_f, _ = verdicts_mode["full"]
    rate_s, med_s, _ = verdicts_mode["highTRA"]
    Q2f_mech_ok = med_f < 0.01     # 机制预测: full 模式 gap 应该很小
    Q2s = (rate_s >= Q2S_DOMINANCE) and (med_s >= Q2S_MEDGAP_MIN)
    print(f"\n  Q2f: full 中位 gap = {med_f:+.4f} "
          f"{'(与机制预测 ~1e-3 一致)' if Q2f_mech_ok else '(!! >= 0.01, 机制解释错了, 重想)'}")
    print(f"  Q2s: 占优率 {rate_s:.1%} (>= {Q2S_DOMINANCE:.0%}?) 且"
          f" 中位 gap {med_s:+.4f} (>= {Q2S_MEDGAP_MIN}?)"
          f"  ==> {'成立' if Q2s else '不成立'}")

    # ---- Q3/Q4/Q6: 两个模式的主 lambda 表 ----
    main_t = df[((df.lam == lam_pick) | (df.arm == "Z_zero"))]
    tbl = main_t.pivot_table(index=["mode", "subset", "arm"],
                             values=["skill", "false_alarm", "detect_corr",
                                     "rmse_early"], aggfunc="mean")
    for mode in WINDOW_MODES:
        print(f"\n  --- 主 lambda = {lam_pick:.3g}, 模式 = {mode} ---")
        print(f"    {'臂':<10s}{'技能分':>9s}{'虚警':>9s}{'检出':>9s}{'早期RMSE':>10s}")
        for tg in tags:
            print(f"    [{tg}]")
            for arm in ["Z_zero", "H0", "TRA4", MOE, "FreeMLP", "MoE_dir"]:
                if (mode, tg, arm) not in tbl.index:
                    continue
                r_ = tbl.loc[(mode, tg, arm)]
                print(f"      {arm:<10s}{r_['skill']:>9.4f}{r_['false_alarm']:>9.4f}"
                      f"{r_['detect_corr']:>9.4f}{r_['rmse_early']:>10.4f}")

    def sk(mode, tg, arm):
        return float(tbl.loc[(mode, tg, arm), "skill"])

    print("\n  --- Q3: 冻结合法性 (两模式) ---")
    Q3_ok = []
    for mode in WINDOW_MODES:
        for tg in tags:
            extra = sk(mode, tg, "MoE_dir") - sk(mode, tg, MOE)
            ok = extra < Q3_DIR_GAIN_MAX
            Q3_ok.append(ok)
            print(f"    [{mode}/{tg}] MoE_dir - MoE = {extra:+.4f}  {'[OK]' if ok else '[X]'}")
    Q3 = all(Q3_ok)

    print("\n  --- Q4: MoE vs FreeMLP (两模式) ---")
    Q4_ok = []
    for mode in WINDOW_MODES:
        for tg in tags:
            d_ = sk(mode, tg, MOE) - sk(mode, tg, "FreeMLP")
            ok = d_ >= -Q4_MOE_TOL
            Q4_ok.append(ok)
            print(f"    [{mode}/{tg}] MoE - FreeMLP = {d_:+.4f}  {'[OK]' if ok else '[X]'}")
    Q4 = all(Q4_ok)

    print("\n  --- Q6: MoE vs TRA4 硬分档 (分段模式, 防止高估连续门控) ---")
    q6_vals = []
    for tg in tags:
        d_ = sk("highTRA", tg, MOE) - sk("highTRA", tg, "TRA4")
        q6_vals.append(d_)
        print(f"    [{tg}] MoE - TRA4 = {d_:+.4f}")
    if all(v < Q6_MOE_VS_TRA4 for v in q6_vals):
        q6_txt = ("硬分档吃掉了大部分收益 -> 写法: '按段重标定即可获得主要增益; "
                  "连续门控的卖点是一个模型服务所有段 + 可解释分区'")
    elif all(v >= Q6_MOE_VS_TRA4 for v in q6_vals):
        q6_txt = "连续门控有超出硬分档的净价值 -> 如实报告数值"
    else:
        q6_txt = "两个子集结论不一致 -> 如实报告, 不外推"
    print(f"    ==> Q6: {q6_txt}")

    # ---- Q5: 门控物理性 (描述性, 与 v1 相同) ----
    print("\n  --- Q5: 门控学到了什么 ---")
    net = arms[MOE]
    with torch.no_grad():
        s_k = F.softplus(net.experts_raw).numpy()
        a_k = net.alphas(to_t(Wz_all[~is_val])).numpy()
    tra = W_all[~is_val][:, 0]
    order = np.argsort([(a_k[:, k] * tra).sum() / max(a_k[:, k].sum(), 1e-12)
                        for k in range(K_MAIN)])
    for k in order:
        wgt = a_k[:, k]
        tra_k = float((wgt * tra).sum() / (wgt.sum() + 1e-12))
        gg = "  ".join(f"{THETA9[j][:8]}={s_k[k, j]:.2f}" for j in [1, 2, 0, 5])
        print(f"    专家{k}: 占比 {wgt.mean():.1%}, 加权TRA {tra_k:.1f}   {gg}")

    verdict = {"Q1": bool(Q1), "Q2f_mech_ok": bool(Q2f_mech_ok),
               "Q2f_median_gap": med_f, "Q2s": bool(Q2s),
               "Q2s_rate": rate_s, "Q2s_median_gap": med_s,
               "Q3": bool(Q3), "Q4": bool(Q4),
               "Q6_vals": {tg: round(v, 4) for tg, v in zip(tags, q6_vals)},
               "seg_fallback": seg_fallback,
               "main_table": tbl.round(4).reset_index().to_dict(orient="records")}
    with open(os.path.join(OUT_DIR, "verdict19b.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 100)
    core = [Q1, Q2s, Q3, Q4]
    print(("[PASS] 核心判据 (Q1/Q2s/Q3/Q4) 全部通过。" if all(core) else
           "[MIXED] 部分核心判据未通过 —— 逐条处理, 不许粉饰:"))
    if not Q1:
        print("  [X] Q1: 样本级前向增益不足 5% -> 查收敛(加EPOCHS)/按退化程度分层再看;"
              " 若仍不行, 如实报告工况自适应在本数据集收益有限。")
    if not Q2s:
        print("  [X] Q2s: 分段模式下增益仍不显著。可能性:"
              "\n      (a) 段内工况仍然混合得太开 -> 收紧 SEG_Q 到 0.9 试一次 (要注明是事后分析);"
              "\n      (b) 幅值失配确实被 lambda 收缩兜住了 -> 那就写"
              "\n          '正则化对适度的 H 幅值失配有鲁棒性', 也是有价值的结论。")
    if not Q3:
        print("  [X] Q3: 方向也需要自适应 -> 按 v1 的处理预案: 改写创新点为"
              " H(w)=[H0+低秩扰动(w)]·diag(g(w)), 如实报告方向自适应的增量。")
    if not Q4:
        print("  [X] Q4: FreeMLP 更强 -> 加大 K 或 'MoE打底+自由残差'; 追不上就把 MoE"
              " 定位成'以极小精度代价换可解释分区', 给出差值。")
    print("=" * 100)

    # ------------------------------------------------------------------
    # [6/6] 出图: 双模式对比 (图注按数据说话, 不写先于数据的断言)
    # ------------------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, mode, ttl in [(axes[0], "full", "整周期平均窗口"),
                              (axes[1], "highTRA", f"高功率段窗口 (TRA前{100-int(SEG_Q*100)}%)")]:
            _, _, grid = verdicts_mode[mode]
            for tg in tags:
                for N in WINDOWS:
                    try:
                        g = grid.xs((tg, N), level=(0, 1))
                    except KeyError:
                        continue
                    ax.plot(g.index, g["gap"], marker="o", ms=4, lw=1.8,
                            label=f"{tg} N={N}")
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("正则化强度 lambda")
            ax.set_ylabel("技能分增益  MoE - H0")
            med = verdicts_mode[mode][1]
            ax.set_title(f"{ttl}\n中位增益 = {med:+.4f}")
            ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.suptitle("脚本19v2: 工况自适应的价值取决于诊断窗口是否覆盖整个包线", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig1_dual_mode.png"), dpi=150)
        plt.close(fig)

        # 门控物理图 (同 v1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        tra_grid = np.linspace(np.quantile(tra, 0.01), np.quantile(tra, 0.99), 200)
        Wq = np.tile(W_all[~is_val].mean(axis=0), (200, 1))
        Wq[:, 0] = tra_grid
        with torch.no_grad():
            wz = to_t((Wq - w_mu) / w_sd)
            A = net.alphas(wz).numpy()
            G = net.gains(wz).numpy()
        ax = axes[0]
        for k in order:
            ax.plot(tra_grid, A[:, k], lw=2, label=f"专家{k}")
        ax.set_xlabel("TRA"); ax.set_ylabel("门控权重 alpha_k")
        ax.set_title("MoE 门控: 各专家负责的 TRA 区间")
        ax.legend(); ax.grid(alpha=.3)
        ax = axes[1]
        for j in [THETA9.index(x) for x in ["fan_flow_mod", "LPT_eff_mod",
                                            "fan_eff_mod", "HPT_eff_mod"]]:
            ax.plot(tra_grid, G[:, j], lw=2, label=THETA9[j])
        ax.axhline(1.0, color="k", ls=":", lw=1, label="全局 H0 (g=1)")
        ax.set_xlabel("TRA"); ax.set_ylabel("增益 g_j(w)")
        ax.set_title("学出的工况增益")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig2_gate_physics.png"), dpi=150)
        plt.close(fig)
        print(f"\n[6/6] 图已存到 {OUT_DIR}/ (fig1_dual_mode / fig2_gate_physics)")
    except Exception as e:
        print(f"\n[6/6] 出图失败 (不影响结论, 数据在 csv 里): {e}")

    print("\n完成。产物: moe2_raw.csv / verdict19b.json / fig1 / fig2")


if __name__ == "__main__":
    main()