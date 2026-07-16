# -*- coding: utf-8 -*-
"""
================================================================================
脚本 17：λ 的 oracle 扫描 —— D 到底是【方法输了】还是【λ 选歪了】？
================================================================================

【为什么必须做这个实验】

  脚本16 的 2×2 消融给出：
      DS02:  D − B = +0.021   （D 微赢）
      DS03:  D − B = −0.170   （D 惨输）

  但看一眼 λ 的选择就会觉得非常可疑：
      DS03:  λ_B* = 3.16e+01     λ_D* = 1.00e-01      ← 差了 300 倍
      DS02:  λ_B* = 3.16e+01     λ_D* = 3.16e+01      ← 一模一样

  λ 是在 dev（DS03 的 unit 1/2/3）上调的，评测在 test（unit 10~15）上。
  所以 D 在 DS03 上的崩溃，有两种完全不同的可能：

    情况① 【方法真的不行】
          即使作弊、直接在 test 上把 λ 调到最优，D 的最优点仍然打不过 B。
          → 说明约束的解族本身就装不下真值，学 λ 也救不回来。
          → "物理约束提高精度" 这个卖点彻底作废。

    情况② 【λ 选歪了，方法没输】
          D 的 oracle 最优点其实高于 B，只是 dev 上选出来的 λ 没泛化到 test。
          → 问题在【超参选择的泛化】，不在【方法本身】。
          → 换一个不需要真值的 λ 选择准则（L-curve / GCV / Morozov）就能救活。
          → 而且"带约束的正则化问题，其 λ 无法用监督方式在 dev 上选"
            这件事本身就是一个可写的方法论贡献。

  ⚠️ 这两种情况的【论文写法完全不同】，不查清楚不能往下走。

--------------------------------------------------------------------------------
【本脚本做什么】

  在 test 集上把整个 λ 网格【扫一遍】，画出 B 和 D 各自的 "oracle λ 曲线"
  （技能分 vs λ）。这是【作弊】的——用 test 调 λ 在正式实验里绝对不允许。
  但作为【诊断实验】，它恰恰是唯一能回答"方法有没有上限"的手段。

  注意区分：
    · Skill(λ_dev)   = 用 dev 选出的 λ，在 test 上的表现   ← 这是【诚实】的数字
    · max_λ Skill    = 在 test 上把 λ 扫遍能达到的最好表现 ← 这是【作弊】的上界

  正式论文里只能报前者。后者只用来做本脚本的【诊断裁决】。

--------------------------------------------------------------------------------
【预注册判据（跑之前写死，跑完照着对，不许改）】

  R1 （→ 情况①）：
      若  max_λ Skill(D) − max_λ Skill(B) < 0.02
      在【DS02 和 DS03 两个子集上都成立】
      → D 真的不行。物理约束在精度上没有上限优势。

      ⚠️ 是【两个子集都】，不是"任一个"。
         脚本16 的 Q6 写成了 max(g2,g3) < 0.02（取最大 = "只要有一个赢就算赢"），
         那是 cherry-picking。这次不许再犯。

  R2 （→ 情况②）：
      若  在 DS03 上  max_λ Skill(D) > max_λ Skill(B)
          且          Skill(D, λ_dev) < Skill(B, λ_dev)
      → 方法本身没输，问题在 λ 的泛化。换无监督准则。

  R1 和 R2 可能都不触发（比如 D 的上界高一点点但 dev 也没选错），
  那属于灰区，脚本会如实报告"灰区"，不硬套结论。

--------------------------------------------------------------------------------
【附带产出（不参与裁决，但直接决定情况②怎么救）】

  额外计算三种【不需要真值 θ】的 λ 选择准则，看它们落在 oracle 曲线的哪里：
    · L-curve           (Hansen 1992, SIAM Review 34(4):561–580)
    · 广义交叉验证 GCV  (Golub, Heath & Wahba 1979, Technometrics 21(2):215–223)
    · Morozov 偏差原理  (噪声水平已知时；我们知道 resid_std，所以可用)

  如果情况② 成立，而且某个无监督准则选出的 λ 接近 oracle 最优点，
  → 那就直接构成第3章的一个方法论贡献，不需要额外发明新东西。

--------------------------------------------------------------------------------
运行:  python 17_lambda_oracle_scan.py
依赖:  和脚本16 完全相同（h5py / numpy / pandas / scipy / sklearn / matplotlib）
耗时:  比脚本16 长（test 上要对每个 λ 都解一次 NNLS）。
       如果太慢，把下面的 FAST_MODE 改成 True。
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
matplotlib.use("Agg")                    # 无显示环境也能存图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 配置区  —— 这一段和脚本16 保持【完全一致】，否则结果不可比
# ============================================================================
DATA_DIR = "D:/N-CMPASS/data_set".strip()   # ⚠️ 是 CMPASS 不是 CMAPSS；末尾不能有空格
OUT_DIR  = "./oracle_out"                   # 本脚本的输出目录
os.makedirs(OUT_DIR, exist_ok=True)         # 目录不存在就建一个

SEED = 42                                   # ⚠️ 必须和脚本16 一样，否则抽样不同、结果不可比
rng = np.random.default_rng(SEED)           # numpy 的新版随机数生成器

# ---- FAST_MODE: 先跑一遍看看流程通不通，再关掉跑完整版 ----
# True  → 只跑 N=100/1000 两个窗口、λ 网格 11 个点、每个子集只用 3 台 test 机（快，约 1/6 时间）
# False → 完整版（4 个窗口 × 21 个 λ × 全部 test 机）
FAST_MODE = False

# 用来【辨识 H】的子集：键=文件名，值=该子集里真正退化的健康参数
IDENT_FILES = {
    "N-CMAPSS_DS01-005.h5": ["HPT_eff_mod"],
    "N-CMAPSS_DS04.h5":     ["fan_eff_mod", "fan_flow_mod"],
    "N-CMAPSS_DS05.h5":     ["HPC_eff_mod", "HPC_flow_mod"],
    "N-CMAPSS_DS07.h5":     ["LPT_eff_mod", "LPT_flow_mod"],
    "N-CMAPSS_DS06.h5":     ["LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod"],
}
N_HEALTHY_ID, N_DEGRADED_ID = 60000, 150000   # 辨识 H 时每个子集抽多少健康/退化样本

# 用来【验证】的子集：H 从来没在这两个子集上训练过
VALID_FILES = {
    "DS02": ("N-CMAPSS_DS02-006.h5", ["HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod"]),
    "DS03": ("N-CMAPSS_DS03-012.h5", ["HPT_eff_mod", "LPT_eff_mod", "LPT_flow_mod"]),
}

# 各健康参数的【真实退化量程】(|θ| 的 99.9 分位数，跨子集取最大)。
# 用途：H 的列必须先乘以量程，条件数才有物理意义（各参数量程相差 12 倍）。
THETA_SPAN = {
    "fan_eff_mod":  0.223446, "fan_flow_mod": 0.121209, "LPC_eff_mod":  0.118767,
    "HPC_flow_mod": 0.070658, "LPC_flow_mod": 0.046187, "LPT_eff_mod":  0.036194,
    "LPT_flow_mod": 0.032908, "HPC_eff_mod":  0.025540, "HPT_eff_mod":  0.018668,
}
# 9 个健康参数的【固定顺序】。HPT_flow_mod 只在 DS08 退化，干净子集拿不到，故不含。
THETA9 = ["HPT_eff_mod", "fan_eff_mod", "fan_flow_mod", "HPC_eff_mod", "HPC_flow_mod",
          "LPT_eff_mod", "LPT_flow_mod", "LPC_eff_mod", "LPC_flow_mod"]

# 健康基准 g(·) 的自变量。⚠️ 绝不能加 Nf/Nc —— 转速本身会被退化污染，逻辑上循环。
OP_COLS = ["TRA", "Mach", "theta_c", "delta_c"]

WINDOWS = [1, 10, 100, 1000] if not FAST_MODE else [100, 1000]   # 每个cycle平均多少个样本
MAX_PER_CYCLE = 1000            # 每个 (unit,cycle) 最多读多少行，防止内存爆
N_REF_CYCLES  = 3               # 逐单元零点 = 最初 3 个 cycle（不是健康期均值！）

# λ 网格。B 和 D 必须共用【同一个网格】，否则不公平。
LAMBDA_GRID = np.logspace(-6, 4, 21) if not FAST_MODE else np.logspace(-6, 4, 11)

MAX_TEST_UNITS = None if not FAST_MODE else 3   # FAST_MODE 下只用前 3 台 test 机

# 预注册判据的阈值（写死，不许跑完再改）
R1_THRESHOLD = 0.02


# ============================================================================
# 数据读取（和脚本16 逐字一致）
# ============================================================================
def get_names(f, key):
    """
    h5 文件里，数值矩阵和列名是【分开存】的，而且列名存成【字节串】(bytes)。
    必须 .decode() 成普通字符串，否则后面 DataFrame 的列名会变成 b'T24' 这种鬼东西。
    """
    arr = np.array(f[key]).ravel()          # ravel(): 拉成一维
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def load_for_ident(path):
    """
    读【辨识 H】用的数据：随机抽样（健康 6 万 + 退化 15 万）。
    技巧：先读体积最小的 A_dev 拿到 hs 列（健康标志），算出要哪些行号，
          再只从 W/X_s/T 里读这些行 —— 避免把 27GB 全读进内存。
    """
    with h5py.File(path, "r") as f:
        A_var, W_var = get_names(f, "A_var"), get_names(f, "W_var")
        Xs_var, T_var = get_names(f, "X_s_var"), get_names(f, "T_var")

        df_A = pd.DataFrame(np.array(f["A_dev"]), columns=A_var)
        ih  = np.where(df_A["hs"].values == 1)[0]      # hs=1 → 健康样本的行号
        idg = np.where(df_A["hs"].values == 0)[0]      # hs=0 → 退化样本的行号

        # rng.choice(总体, 抽几个, replace=False) → 不放回抽样
        take = np.sort(np.concatenate([
            rng.choice(ih,  min(N_HEALTHY_ID,  len(ih)),  replace=False),
            rng.choice(idg, min(N_DEGRADED_ID, len(idg)), replace=False)]))

        # h5py 支持"花式索引"：f["W_dev"][take, :] 只把 take 这些行读进来
        W, Xs, T = f["W_dev"][take, :], f["X_s_dev"][take, :], f["T_dev"][take, :]

    return pd.concat([df_A.iloc[take].reset_index(drop=True),
                      pd.DataFrame(W,  columns=W_var),
                      pd.DataFrame(Xs, columns=Xs_var),
                      pd.DataFrame(T,  columns=T_var)], axis=1)


def load_per_cycle(path, split):
    """
    读【验证】用的数据：按 (unit, cycle) 分组抽样。
    为什么不能像上面那样随机抽？因为后面的"聚合窗口 N"是【在同一个 cycle 内部】
    平均 N 个样本，所以必须保证每个 cycle 都有足够多的样本。
    """
    with h5py.File(path, "r") as f:
        A_var, W_var = get_names(f, "A_var"), get_names(f, "W_var")
        Xs_var, T_var = get_names(f, "X_s_var"), get_names(f, "T_var")

        df_A = pd.DataFrame(np.array(f[f"A_{split}"]), columns=A_var)

        take = []
        # groupby([...]).indices 返回 {(unit,cycle): 行号数组}
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
    """
    相似理论修正参数 (Walsh & Fletcher, Gas Turbine Performance, 2nd ed., 2004)

    目的：把"工况"的影响先用【已知的物理规律】除掉，剩下的才可能是退化。
      · theta_c = T2/T_ref   无量纲总温
      · delta_c = P2/P_ref   无量纲总压
      · 转速   → N/√theta_c            （修正转速）
      · 燃油   → Wf/(delta_c·√theta_c) （修正燃油）
      · 温度   → T24/T2 等              （温比，无量纲）
      · 压力   → P24/P2 等              （压比，无量纲）
    结果：13 个修正通道（P2 当了分母，自己消掉了，所以是 13 不是 14）
    """
    df = df.copy()
    df["theta_c"] = df["T2"] / ref["T_ref"]
    df["delta_c"] = df["P2"] / ref["P_ref"]
    sq = np.sqrt(df["theta_c"])

    cols = []
    for c in ["Nf", "Nc"]:                                    # 修正转速
        df[f"{c}_c"] = df[c] / sq
        cols.append(f"{c}_c")

    df["Wf_c"] = df["Wf"] / (df["delta_c"] * sq)              # 修正燃油
    cols.append("Wf_c")

    for c in ["T24", "T30", "T48", "T50"]:                    # 温比
        df[f"{c}_c"] = df[c] / df["T2"]
        cols.append(f"{c}_c")

    for c in ["P15", "P21", "P24", "Ps30", "P40", "P50"]:     # 压比
        df[f"{c}_c"] = df[c] / df["P2"]
        cols.append(f"{c}_c")

    return df, cols


# ============================================================================
# 反演器：B(岭回归) / D(约束+收缩) / Z(全零哨兵)
# ============================================================================
def build_cum(T, n):
    """
    "累加矩阵" Cum，形状 (T*n, T*n)。
    作用：把【每步增量 d】映射成【累计退化 θ】。
        θ(t) = −Σ_{i≤t} d_i    ⟺    θ = −Cum·d
    结构：分块下三角，每个块是 n×n 单位阵
        [ I  0  0 ]
        [ I  I  0 ]
        [ I  I  I ]
    """
    Cum = np.zeros((T * n, T * n))
    for t in range(T):
        for i in range(t + 1):
            Cum[t*n:(t+1)*n, i*n:(i+1)*n] = np.eye(n)
    return Cum


def build_M(Hn, T):
    """
    设计矩阵 M，形状 (13T, 9T)。
    预测残差:  r̂(t) = Hn·θ(t) = Hn·(−Σ_{i≤t} d_i) = −Σ_{i≤t} (Hn·d_i)
    所以 M[第t个13行块, 第i个9列块] = −Hn  当 i ≤ t，否则 0。
    """
    m, n = Hn.shape                       # m=13（残差通道数），n=9（健康参数数）
    M = np.zeros((T * m, T * n))
    for t in range(T):
        for i in range(t + 1):
            M[t*m:(t+1)*m, i*n:(i+1)*n] = -Hn
    return M


def solve_B(Hn, R, lam):
    """
    方法 B —— Tikhonov 岭回归（Doel 1994, TEMPER 的做法）
        θ̂ = (HᵀH + λI)⁻¹ Hᵀ r
    没有任何约束，逐 cycle 独立求解（所以特别快）。
    ⚠️ 用 np.linalg.solve 而不是先求逆再乘，数值上更稳。
    """
    A = Hn.T @ Hn + lam * np.eye(Hn.shape[1])   # (9,9)
    Binv = np.linalg.solve(A, Hn.T)             # (9,13)
    return R @ Binv.T                           # (T,13) @ (13,9) = (T,9)


def solve_D_cached(M, Cum, R, lam, T, n):
    """
    方法 D —— ★本文方法：约束 + 收缩
        min ‖Hn·θ − r‖² + λ‖θ‖²   s.t.  θ ≤ 0 ,  θ(t+1) ≤ θ(t)

    【怎么把 Tikhonov 项塞进 NNLS（不需要装 cvxpy）】
      重参数化后 θ = −Cum·d，d ≥ 0，两个约束【自动满足】。
      又因为负号不影响范数：‖θ‖² = ‖Cum·d‖²
      于是目标 = ‖M·d − y‖² + λ‖Cum·d‖²
              = ‖ [    M    ]      [ y ] ‖²      ← 把 √λ·Cum 【增广】到 M 底下
                ‖ [ √λ·Cum ]·d  −  [ 0 ] ‖        把 0     增广到 y 底下
      这就是一个标准的非负最小二乘 → scipy.optimize.nnls 精确求解。
      （这是岭回归"数据增广"技巧的标准做法，见任何统计教材）

    ⚠️ _cached 版本：M 和 Cum 【不依赖 λ】，所以在扫描 λ 时只构造一次，
       否则每个 λ 都重建一遍 (13T×9T) 的矩阵，会慢得离谱。
    """
    # np.vstack: 沿行方向堆叠 → 增广设计矩阵 (13T+9T, 9T)
    M_aug = np.vstack([M, np.sqrt(lam) * Cum])
    # np.concatenate: 拼接向量 → 增广观测 (13T+9T,)
    y_aug = np.concatenate([R.reshape(-1), np.zeros(T * n)])

    d, _ = nnls(M_aug, y_aug, maxiter=50 * T * n)
    # cumsum: 累加 → θ(t) = −Σ_{i≤t} d_i
    return -np.cumsum(d.reshape(T, n), axis=0)


def solve_Z(Hn, R):
    """
    ★ 全零哨兵：什么都不预测，恒输出 0。
    纪律：任何方法只要在某个指标上【跟哨兵打平】，那个指标就是坏的
         （因为它在奖励"什么都不做"）。
    脚本14/15 就是栽在这上面：|v·(θ̂−θ_true)| 这个指标被哨兵白嫖了。
    """
    return np.zeros((R.shape[0], Hn.shape[1]))


# ============================================================================
# 无监督 λ 选择准则（不需要真值 θ！）—— 附带产出，不参与裁决
# ============================================================================
def unsupervised_lambda_selection(Hn, R, lam_grid, M, Cum, T, n, method="B"):
    """
    三种【只看残差、不看真值】的 λ 选择准则。
    如果脚本判定为"情况②（λ 选歪了）"，这三个准则就是救活方法 D 的药。

    对每个 λ，先求出 θ̂(λ)，然后计算：
      · 残差范数     res(λ) = ‖Hn·θ̂ − r‖
      · 解范数       sol(λ) = ‖θ̂‖

    ---- 1. L-curve (Hansen 1992) ----
      在 log-log 平面上画 (log res, log sol)，这条曲线通常是个 "L" 形。
      拐角（曲率最大处）就是"欠拟合与过拟合的最佳折中点"。
      我们用离散曲率公式找拐角。

    ---- 2. GCV（广义交叉验证, Golub-Heath-Wahba 1979）----
      GCV(λ) = ‖Hn·θ̂ − r‖² / (m − tr(A_λ))²
      其中 A_λ = Hn(HnᵀHn+λI)⁻¹Hnᵀ 是"帽子矩阵"，tr(A_λ) 是【有效自由度】。
      直觉：它近似"留一交叉验证误差"，但不用真的做 LOO。取 GCV 最小的 λ。
      ⚠️ 只对【线性】估计器（方法B）有严格定义。对 D（带不等式约束，非线性）
         我们用一个近似：把 tr(A_λ) 换成"活跃约束外的自由参数个数"。
         这只是启发式，论文里要如实标注。

    ---- 3. Morozov 偏差原理 ----
      我们【知道】噪声水平（残差已经用健康样本残差标准差归一化过，所以 σ≈1，
      而聚合 N 个样本后噪声降到 1/√N）。
      偏差原理说：选一个 λ，使得 ‖Hn·θ̂ − r‖ ≈ 期望的噪声范数。
      拟合得比噪声还准 = 在拟合噪声（过拟合）；差太远 = 欠拟合。
    """
    m = Hn.shape[0]                # 13 个残差通道
    res_norms, sol_norms, gcvs = [], [], []

    for lam in lam_grid:
        if method == "B":
            th = solve_B(Hn, R, lam)
        else:
            th = solve_D_cached(M, Cum, R, lam, T, n)

        pred = th @ Hn.T                                    # 预测残差 (T,13)
        res = float(np.linalg.norm(pred - R))               # ‖Hn·θ̂ − r‖
        sol = float(np.linalg.norm(th))                     # ‖θ̂‖
        res_norms.append(max(res, 1e-12))
        sol_norms.append(max(sol, 1e-12))

        # --- GCV ---
        # 有效自由度 tr(A_λ) = Σ_j s_j²/(s_j²+λ)，s_j 是 Hn 的奇异值
        s = np.linalg.svd(Hn, compute_uv=False)
        dof = float(np.sum(s**2 / (s**2 + lam)))
        if method == "D":
            # 约束把一部分参数"钉"在边界上（d_i=0），有效自由度更低。
            # 用活跃比例做个粗糙修正。⚠️ 这是启发式，不是严格 GCV。
            active_frac = float(np.mean(np.abs(np.diff(th, axis=0, prepend=0)) > 1e-9))
            dof = dof * max(active_frac, 1e-3)
        denom = max(m - dof, 1e-6)
        gcvs.append(res**2 / (denom ** 2))

    res_norms = np.array(res_norms)
    sol_norms = np.array(sol_norms)
    gcvs = np.array(gcvs)

    # ---- L-curve 拐角：log-log 平面上的最大曲率点 ----
    x, y = np.log(res_norms), np.log(sol_norms)
    # np.gradient 做数值微分（一阶导、二阶导）
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    # 平面曲线的曲率公式 κ = |x'y'' − y'x''| / (x'²+y'²)^{3/2}
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5) + 1e-12
    lam_lcurve = float(lam_grid[int(np.argmax(curvature))])

    # ---- GCV：取最小 ----
    lam_gcv = float(lam_grid[int(np.argmin(gcvs))])

    # ---- Morozov：残差范数最接近期望噪声范数的那个 λ ----
    # 残差已归一化 → 单样本噪声 σ≈1；聚合 N 个样本后 σ≈1/√N。
    # 这里我们从 R 本身估计：用一个非常小的 λ 解出来的残差范数当作"噪声地板"。
    # （更严谨的做法是从健康期样本估计，这里先给一个可用的近似）
    noise_floor = res_norms[0]                       # λ 最小时的残差范数
    target = noise_floor * 1.1                       # 允许比噪声地板高 10%
    lam_morozov = float(lam_grid[int(np.argmin(np.abs(res_norms - target)))])

    return {
        "L_curve":  lam_lcurve,
        "GCV":      lam_gcv,
        "Morozov":  lam_morozov,
        "res_norms": res_norms.tolist(),
        "sol_norms": sol_norms.tolist(),
    }


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 92)
    print("脚本 17：λ 的 oracle 扫描 —— D 是【方法输了】还是【λ 选歪了】？")
    print("=" * 92)
    print(f"\n  FAST_MODE = {FAST_MODE}")
    print(f"  λ 网格    = {len(LAMBDA_GRID)} 个点, 从 {LAMBDA_GRID[0]:.1e} 到 {LAMBDA_GRID[-1]:.1e}")
    print(f"  聚合窗口  = {WINDOWS}")
    print(f"\n  【预注册判据（已写死，跑完不许改）】")
    print(f"    R1: max_λ S(D) − max_λ S(B) < {R1_THRESHOLD}  在【两个子集上都成立】 → 情况①（方法真输）")
    print(f"    R2: DS03 上 max_λ S(D) > max_λ S(B) 且 S(D,λ_dev) < S(B,λ_dev) → 情况②（λ 选歪）")

    # ------------------------------------------------------------------
    # [1/4] 重建健康基准 g(·) 与影响系数矩阵 H
    #       （完全复刻脚本 12/13/15/16 的标准流程，7 步）
    # ------------------------------------------------------------------
    print("\n\n[1/4] 重建健康基准与影响系数矩阵 H ...")

    # 步骤1：读 5 个辨识子集
    data = {}
    for fn in IDENT_FILES:
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"  !! 找不到文件: {p}")
            return
        data[fn] = load_for_ident(p)
        print(f"    读入 {fn}: {len(data[fn])} 行")

    # 步骤2：全局参考状态（⚠️ 必须统一，各子集不能各算各的，否则残差不在同一尺度）
    pool = pd.concat([d[d["hs"] == 1] for d in data.values()], ignore_index=True)
    ref = {"T_ref": pool["T2"].median(), "P_ref": pool["P2"].median()}
    print(f"    全局参考: T_ref={ref['T_ref']:.2f}, P_ref={ref['P_ref']:.2f}")

    # 步骤3：加相似理论修正参数 → 13 个修正通道
    for fn in data:
        data[fn], corrected_cols = add_corrected(data[fn], ref)
    pool, _ = add_corrected(pool, ref)
    resid_cols = [f"r_{c}" for c in corrected_cols]

    # 步骤4：用【全部健康样本】拟合 g(TRA, Mach, theta_c, delta_c) → 13 个 GBM
    base = {}
    for c in corrected_cols:
        m = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=SEED)
        m.fit(pool[OP_COLS].values, pool[c].values)
        base[c] = m
    print(f"    健康基准: {len(base)} 个 GBM 拟合完成")

    # 步骤5：残差 r = x_c − g(w)，再除以健康样本残差的标准差
    def residual(df):
        r = pd.DataFrame(index=df.index)
        for c in corrected_cols:
            r[f"r_{c}"] = df[c].values - base[c].predict(df[OP_COLS].values)
        return r

    resid_std = residual(pool).std()

    # 步骤6：逐子集把归一化残差回归到该子集退化的 θ 上 → H 的各列
    #        ⚠️ LinearRegression 默认 fit_intercept=True。脚本09 的 bug 是
    #           拟合时有截距、预测时把截距丢了。这里我们只取 coef_ 用来构造 H，
    #           而 H 的用途是"残差对 θ 的敏感度"，截距（≈0.148σ）会在
    #           【逐单元 baselining】那一步被减掉，所以是自洽的。
    H_cols = {}
    for fn in ["N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS04.h5",
               "N-CMAPSS_DS05.h5", "N-CMAPSS_DS07.h5"]:
        df, params = data[fn], IDENT_FILES[fn]
        r = residual(df) / resid_std
        for rc in resid_cols:
            lr = LinearRegression().fit(df[params].values, r[rc].values)
            for j, p in enumerate(params):
                H_cols.setdefault(p, {})[rc] = lr.coef_[j]

    # LPC 两列只能从 DS06 拿（DS06 里 LPC 和 HPC 同时退化）
    df6 = data["N-CMAPSS_DS06.h5"]
    df6 = df6[df6["hs"] == 0]                       # 只用退化样本
    r6 = residual(df6) / resid_std
    p6 = IDENT_FILES["N-CMAPSS_DS06.h5"]
    for rc in resid_cols:
        lr = LinearRegression().fit(df6[p6].values, r6[rc].values)
        for j, p in enumerate(p6):
            if p.startswith("LPC"):                 # 只取 LPC 两列；HPC 用 DS05 的干净结果
                H_cols.setdefault(p, {})[rc] = lr.coef_[j]

    # 步骤7：量程归一化 H_n = H × diag(真实退化量程)   ← 这一步不能省
    H = pd.DataFrame(H_cols).reindex(index=resid_cols)[THETA9]
    span = np.array([THETA_SPAN[c] for c in THETA9])
    Hn = H.values * span                            # (13, 9)

    # SVD：拿到条件数和零空间方向
    U, s, Vt = np.linalg.svd(Hn)
    v_null = Vt[-1] / np.linalg.norm(Vt[-1])        # 最小奇异值对应的右奇异向量
    print(f"    cond(H_n) = {s[0]/s[-1]:.1f} ,  σ_min = {s[-1]:.4f}σ")
    print(f"    零空间方向: " + ", ".join(
        f"{THETA9[i]}={v_null[i]:+.3f}" for i in np.argsort(-np.abs(v_null))[:3]))

    del data, pool                                  # 释放内存

    # ------------------------------------------------------------------
    # 窗口构造（和脚本16 一致）
    # ------------------------------------------------------------------
    def build_windows(df, unit):
        """
        对一台发动机，构造 {N: (R, TH)}：
          R  形状 (T, 13)  —— 每个 cycle 的归一化残差（相对于该单元零点）
          TH 形状 (T, 9)   —— 每个 cycle 的真值 θ/量程（相对于该单元零点）

        ⚠️ 零点 = 该单元【最初 3 个 cycle】，不是健康期均值。
           脚本14 的 bug：用健康期均值当零点，而健康期内 θ 本就在缓慢下降，
           导致健康期前半段 θ_relative > 0，而方法 C/D 强制 θ ≤ 0
           → 真值落在可行域【外面】→ C 被自己判了死刑。
        """
        du = df[df["unit"] == unit].copy()
        cycles = sorted(du["cycle"].unique())
        m_ref = du["cycle"].isin(cycles[:N_REF_CYCLES])
        b_unit = du.loc[m_ref, resid_cols].mean().values           # 残差零点
        th0 = (du.loc[m_ref, THETA9].values / span).mean(axis=0)   # θ 零点

        out = {}
        for N in WINDOWS:
            R_list, TH_list = [], []
            for c in cycles:
                sub = du[du["cycle"] == c]
                k = min(N, len(sub))
                idx = rng.choice(len(sub), k, replace=False)
                # 平均 N 个样本 → 噪声按 1/√N 衰减
                R_list.append(sub[resid_cols].values[idx].mean(axis=0) - b_unit)
                TH_list.append(sub[THETA9].values[0] / span - th0)
            out[N] = (np.array(R_list), np.array(TH_list))
        return out

    # ------------------------------------------------------------------
    # [2/4] 逐子集：dev 上选 λ（诚实的做法） + test 上 oracle 扫描（作弊的诊断）
    # ------------------------------------------------------------------
    all_rows = []          # 每一行 = (子集, N, λ, 方法, 单元, 技能分, 虚警, ...)
    lam_dev_record = {}    # 记录 dev 选出的 λ（用于 R2）
    unsup_record = {}      # 记录无监督准则选出的 λ

    for tag, (fn, true_faults) in VALID_FILES.items():
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"\n  !! 缺文件 {fn}，跳过 {tag}")
            continue

        print("\n\n" + "#" * 92)
        print(f"# {tag}    真实退化部件: {true_faults}")
        print("#" * 92)

        # 读 dev 和 test
        ds = {}
        for split in ["dev", "test"]:
            d = load_per_cycle(p, split)
            d, _ = add_corrected(d, ref)
            ds[split] = pd.concat([d, residual(d) / resid_std], axis=1)

        # 零空间重合度：真实故障方向 与 H 零空间方向 的夹角余弦
        th_final = ds["dev"][THETA9].min().values / span
        align = abs(float(v_null @ th_final)) / np.linalg.norm(th_final)
        print(f"\n  零空间重合度 = {align*100:.1f}%")

        idx_tf = [THETA9.index(q) for q in true_faults]      # 真退化部件的下标
        other = [i for i in range(9) if i not in idx_tf]     # 未退化部件（虚警要看这些）

        # ------------------------------------------------------
        # (a) 在 dev 上选 λ —— 这是【诚实】的做法，和脚本16 完全一致
        # ------------------------------------------------------
        dev_units = sorted(ds["dev"]["unit"].unique())[:3]
        dev_wins = {u: build_windows(ds["dev"], u) for u in dev_units}

        print(f"\n  --- (a) 在 {tag}-dev 上选 λ（诚实做法，B 和 D 用同一网格、同一批数据）---")
        lamB_dev, lamD_dev = {}, {}
        for N in WINDOWS:
            errB, errD = [], []
            for lam in LAMBDA_GRID:
                eB, eD = [], []
                for u in dev_units:
                    R, TH = dev_wins[u][N]
                    T = len(R)
                    M = build_M(Hn, T)
                    Cum = build_cum(T, 9)
                    eB.append(np.mean((solve_B(Hn, R, lam) - TH) ** 2))
                    eD.append(np.mean((solve_D_cached(M, Cum, R, lam, T, 9) - TH) ** 2))
                errB.append(np.mean(eB))
                errD.append(np.mean(eD))
            lamB_dev[N] = float(LAMBDA_GRID[int(np.argmin(errB))])
            lamD_dev[N] = float(LAMBDA_GRID[int(np.argmin(errD))])
            print(f"    N={N:>5d}   λ_B(dev)={lamB_dev[N]:.2e}   λ_D(dev)={lamD_dev[N]:.2e}")
        lam_dev_record[tag] = {"B": lamB_dev, "D": lamD_dev}

        # ------------------------------------------------------
        # (b) 在 test 上做 oracle 扫描 —— 这是【作弊】的，只用于诊断
        # ------------------------------------------------------
        test_units = sorted(ds["test"]["unit"].unique())
        if MAX_TEST_UNITS is not None:
            test_units = test_units[:MAX_TEST_UNITS]

        print(f"\n  --- (b) 在 {tag}-test 上做 λ 的 oracle 扫描（作弊，仅用于诊断）---")
        print(f"      test 单元: {[int(u) for u in test_units]}")
        print(f"      共需求解 {len(test_units)}×{len(WINDOWS)}×{len(LAMBDA_GRID)} "
              f"= {len(test_units)*len(WINDOWS)*len(LAMBDA_GRID)} 次 NNLS，请耐心 ...")

        # 先把每台机的窗口都建好（避免重复）
        test_wins = {u: build_windows(ds["test"], u) for u in test_units}

        for u in test_units:
            for N in WINDOWS:
                R, TH = test_wins[u][N]
                T = len(R)

                # ⚠️ M 和 Cum 不依赖 λ → 每个 (unit,N) 只构造一次，在 λ 循环外面
                M = build_M(Hn, T)
                Cum = build_cum(T, 9)

                mse_zero = float(np.mean(TH ** 2))     # 技能分的分母 = 全零哨兵的 MSE

                # --- 全零哨兵（λ 无关，只记一次）---
                TH_z = solve_Z(Hn, R)
                all_rows.append({
                    "subset": tag, "unit": int(u), "N": N,
                    "method": "Z_全零哨兵", "lam": np.nan,
                    "skill": 0.0,                       # 定义上恒为 0
                    "rmse": float(np.sqrt(mse_zero)),
                    "false_alarm": 0.0,                 # 全零 → 虚警恒为 0
                    "detect_corr": 0.0,
                })

                # --- 扫 λ ---
                for lam in LAMBDA_GRID:
                    for name, TH_hat in [
                        ("B_岭回归",     solve_B(Hn, R, lam)),
                        ("D_约束+收缩★", solve_D_cached(M, Cum, R, lam, T, 9)),
                    ]:
                        err = TH_hat - TH
                        mse = float(np.mean(err ** 2))

                        # 技能分 S = 1 − MSE/MSE(全零)
                        #   1  = 完美
                        #   0  = 跟"什么都不预测"一样  ← 哨兵在这条线上
                        #   <0 = 比什么都不做还糟
                        skill = 1.0 - mse / max(mse_zero, 1e-12)

                        # 虚警幅度：6 个【没退化】的部件上 |θ̂| 的均值（真值恒为 0，应≈0）
                        false_alarm = float(np.mean(np.abs(TH_hat[:, other])))

                        # 检出相关性：3 个【真退化】部件上，θ̂ 与真值的相关系数
                        corrs = []
                        for i in idx_tf:
                            if np.std(TH[:, i]) > 1e-9 and np.std(TH_hat[:, i]) > 1e-9:
                                corrs.append(np.corrcoef(TH_hat[:, i], TH[:, i])[0, 1])
                        detect = float(np.mean(corrs)) if corrs else 0.0

                        all_rows.append({
                            "subset": tag, "unit": int(u), "N": N,
                            "method": name, "lam": float(lam),
                            "skill": skill, "rmse": float(np.sqrt(mse)),
                            "false_alarm": false_alarm, "detect_corr": detect,
                        })

            print(f"      unit {int(u)} 完成")

        # ------------------------------------------------------
        # (c) 无监督 λ 选择准则（附带产出，不参与裁决）
        # ------------------------------------------------------
        print(f"\n  --- (c) 无监督 λ 准则（不看真值 θ，只看残差）---")
        unsup_record[tag] = {}
        u0 = test_units[0]                       # 用第一台 test 机做示例
        N0 = WINDOWS[-1]                         # 用最大窗口（噪声最小）
        R0, TH0 = test_wins[u0][N0]
        T0 = len(R0)
        M0, Cum0 = build_M(Hn, T0), build_cum(T0, 9)
        for meth in ["B", "D"]:
            sel = unsupervised_lambda_selection(Hn, R0, LAMBDA_GRID, M0, Cum0, T0, 9, meth)
            unsup_record[tag][meth] = {k: sel[k] for k in ["L_curve", "GCV", "Morozov"]}
            print(f"    方法{meth}: L-curve={sel['L_curve']:.2e}  "
                  f"GCV={sel['GCV']:.2e}  Morozov={sel['Morozov']:.2e}")

    # ------------------------------------------------------------------
    # [3/4] 汇总与裁决
    # ------------------------------------------------------------------
    df_r = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "lambda_oracle_scan.csv")
    df_r.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n\n[3/4] 原始结果已存: {csv_path}   ({len(df_r)} 行)")

    print("\n\n" + "=" * 92)
    print("λ 的 oracle 曲线（技能分，跨 test 单元与窗口 N 取平均）")
    print("=" * 92)

    verdict = {}
    for tag in df_r["subset"].unique():
        sub = df_r[(df_r["subset"] == tag) & (df_r["method"] != "Z_全零哨兵")]

        # 对每个 (方法, λ)，跨 unit 和 N 取平均技能分
        pv = sub.pivot_table(index="lam", columns="method", values="skill", aggfunc="mean")
        print(f"\n\n{'#'*92}\n# {tag}\n{'#'*92}")
        print("\n  --- 技能分 S vs λ（★ 标记 = 该方法的 oracle 最优点）---")

        colB, colD = "B_岭回归", "D_约束+收缩★"
        best_lamB = float(pv[colB].idxmax()); best_SB = float(pv[colB].max())
        best_lamD = float(pv[colD].idxmax()); best_SD = float(pv[colD].max())

        print(f"\n  {'λ':>10s}  {'B_岭回归':>12s}  {'D_约束+收缩':>14s}")
        print("  " + "-" * 42)
        for lam in pv.index:
            mB = " ★" if abs(lam - best_lamB) < 1e-12 else "  "
            mD = " ★" if abs(lam - best_lamD) < 1e-12 else "  "
            print(f"  {lam:>10.2e}  {pv.loc[lam, colB]:>10.4f}{mB} "
                  f"{pv.loc[lam, colD]:>12.4f}{mD}")

        # ---- 关键数字 ----
        # oracle 上界（作弊，用 test 调 λ）
        oracle_gap = best_SD - best_SB

        # 诚实数字（用 dev 选的 λ，在 test 上的表现）
        #   注意：λ_dev 是逐 N 的，所以要逐 N 取出来再平均
        sub_full = df_r[(df_r["subset"] == tag)]
        S_dev_B, S_dev_D = [], []
        for N in WINDOWS:
            lb = lam_dev_record[tag]["B"][N]
            ld = lam_dev_record[tag]["D"][N]
            mB = sub_full[(sub_full["method"] == colB) & (sub_full["N"] == N) &
                          (np.isclose(sub_full["lam"], lb))]["skill"].mean()
            mD = sub_full[(sub_full["method"] == colD) & (sub_full["N"] == N) &
                          (np.isclose(sub_full["lam"], ld))]["skill"].mean()
            S_dev_B.append(mB); S_dev_D.append(mD)
        S_dev_B = float(np.nanmean(S_dev_B))
        S_dev_D = float(np.nanmean(S_dev_D))

        verdict[tag] = {
            "oracle_S_B": best_SB, "oracle_lam_B": best_lamB,
            "oracle_S_D": best_SD, "oracle_lam_D": best_lamD,
            "oracle_gap_D_minus_B": oracle_gap,
            "dev_S_B": S_dev_B, "dev_S_D": S_dev_D,
            "dev_gap_D_minus_B": S_dev_D - S_dev_B,
        }

        print(f"\n  ★ oracle 上界（在 test 上把 λ 扫遍 —— 作弊，仅诊断用）:")
        print(f"      B: S = {best_SB:+.4f}  (λ = {best_lamB:.2e})")
        print(f"      D: S = {best_SD:+.4f}  (λ = {best_lamD:.2e})")
        print(f"      ★ oracle 差距 D − B = {oracle_gap:+.4f}")
        print(f"\n  ○ 诚实数字（用 dev 选的 λ，在 test 上评测）:")
        print(f"      B: S = {S_dev_B:+.4f}")
        print(f"      D: S = {S_dev_D:+.4f}")
        print(f"      ○ 诚实差距 D − B = {S_dev_D - S_dev_B:+.4f}")
        print(f"\n  Δ（oracle − 诚实）: B 损失 {best_SB - S_dev_B:+.4f} ; "
              f"D 损失 {best_SD - S_dev_D:+.4f}")
        print(f"      → D 的损失如果远大于 B，说明【D 的 λ 特别难选】，这本身就是发现。")

    # ------------------------------------------------------------------
    # [4/4] 预注册判据的裁决
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 92)
    print("# 裁　决（严格按预注册判据，不许事后改）")
    print("#" * 92)

    tags = list(verdict.keys())
    if len(tags) < 2:
        print("\n  ⚠️ 只有一个子集有结果，无法执行『两个子集都成立』的判据。请检查数据。")
    else:
        gaps = {t: verdict[t]["oracle_gap_D_minus_B"] for t in tags}
        print("\n  oracle 差距 (D − B)：")
        for t in tags:
            print(f"    {t}: {gaps[t]:+.4f}")

        # ---- R1 ----
        # ⚠️ all(...) 而不是 any(...)。脚本16 的 max() 是 cherry-picking，这次不犯。
        R1_hit = all(gaps[t] < R1_THRESHOLD for t in tags)

        print(f"\n  --- R1: 所有子集的 oracle 差距都 < {R1_THRESHOLD} ? ---")
        print(f"      {'✅ 成立' if R1_hit else '❌ 不成立'}"
              f"   (逐子集: " + ", ".join(f"{t}={gaps[t]:+.4f}" for t in tags) + ")")

        # ---- R2 ----
        R2_hit = False
        if "DS03" in verdict:
            v = verdict["DS03"]
            cond_a = v["oracle_gap_D_minus_B"] > 0            # oracle 上 D 赢
            cond_b = v["dev_gap_D_minus_B"] < 0               # 诚实上 D 输
            R2_hit = cond_a and cond_b
            print(f"\n  --- R2: DS03 上 oracle 时 D 赢，但 dev 选 λ 时 D 输 ? ---")
            print(f"      oracle D−B = {v['oracle_gap_D_minus_B']:+.4f}  "
                  f"({'D 赢 ✓' if cond_a else 'D 没赢 ✗'})")
            print(f"      dev    D−B = {v['dev_gap_D_minus_B']:+.4f}  "
                  f"({'D 输 ✓' if cond_b else 'D 没输 ✗'})")
            print(f"      {'✅ R2 成立' if R2_hit else '❌ R2 不成立'}")

        # ---- 结论 ----
        print("\n\n  " + "=" * 88)
        if R1_hit:
            print("  🔴 判定：【情况①】—— D 真的不行。")
            print("  " + "=" * 88)
            print("""
    即使作弊、直接用 test 把 λ 调到最优，D 的上界仍然打不过 B。
    这说明约束的解族本身就装不下真值，换任何 λ 选择准则都救不回来。

    ⇒ "物理约束提高退化量估计精度" 这个卖点【彻底作废】，不要再试图抢救。
    ⇒ 第3章主线改成（这三条都有独立实验依据，不依赖本实验）：
         1. 工况自适应影响系数矩阵  H(w) = H₀·diag(g_φ(w))
            （方向余弦≥0.9865 稳，幅值波动2.12倍且随TRA单调 —— ⓪-d 已测）
         2. 零空间感知的可辨识性量化
            （σ_min=0.183σ, cond=1562.8, 零空间方向在 7列/9列 版本下一致）
         3. 诚实的负面结论 + 作用边界：
            物理约束在【精度】上不优于 Tikhonov，
            但在【虚警抑制】上有 4–9 倍优势且不损失检出。
            ★ 约束的价值在【特异性】（说对是哪个部件坏），不在【精度】（算准坏了多少）。

    ⚠️ 注意：即便 R1 成立，"约束降低虚警" 这个结论【依然成立】，
       它来自脚本16 的独立测量（DS02: B 0.0089 → D 0.0023；DS03: 0.0085 → 0.0017），
       和本脚本的 λ 扫描无关。不要因为 R1 成立就把它一起扔掉。
            """)
        elif R2_hit:
            print("  🟢 判定：【情况②】—— 方法没输，是 λ 选歪了。")
            print("  " + "=" * 88)
            print(f"""
    D 的 oracle 上界高于 B，但 dev 上选出的 λ 没能泛化到 test。
    问题在【超参选择的泛化】，不在【方法本身】。

    ⇒ 下一步：换掉监督式的 λ 选择，用【无监督准则】（不需要真值 θ）：
         · L-curve          (Hansen 1992, SIAM Review 34(4):561–580)
         · GCV              (Golub, Heath & Wahba 1979, Technometrics 21(2):215–223)
         · Morozov 偏差原理 (噪声水平已知；我们知道 resid_std，所以可用)
       本脚本已经把这三个准则选出的 λ 算好了（见上面 (c) 部分和图上的竖线），
       直接对照 oracle 曲线看它们落在哪里。

    ⇒ 更进一步（这才是第3章的算法创新）：
       把整个约束反演展开成【可微网络层】(deep unfolding)，
       让 λ 从"要泛化的超参"变成"要学习的参数"，
       甚至让 λ = λ_ψ(w) 随工况变化、或每个健康参数一个 λ_j
       （因为各 θ 量程差 12 倍，用同一个 λ 本来就不合理）。
       文献: Gregor & LeCun 2010 (LISTA);
             Monga, Li & Eldar 2021, "Algorithm Unrolling",
             IEEE Signal Processing Magazine 38(2):18–44.

    ⇒ 而且"带约束的正则化问题，其 λ 无法用监督方式在 dev 上选"
       这件事【本身就是一个可写的方法论贡献】。
            """)
        else:
            print("  🟡 判定：【灰区】—— R1 和 R2 都不触发。")
            print("  " + "=" * 88)
            print("""
    可能的情形：
      (i)  D 的 oracle 上界确实高于 B（某个子集 gap ≥ 0.02），
           但 dev 选的 λ 也没选错（诚实差距也是正的）。
           → 那说明脚本16 的结论可能受抽样噪声影响，值得换 SEED 复跑一次确认。
      (ii) 各子集之间不一致（一个赢一个输）。
           → 说明约束的价值【依赖故障方向】。去看零空间重合度这个自变量。
               ⚠️ 但交接文档已记录：重合度规律被证伪，而且方向是反的
                  (DS02 重合度 2.5% → D−B=+0.021；DS03 重合度 27.9% → D−B=−0.170)。
                  如果这次还是这样，不要再试图救这条规律。

    无论哪种，都【不要】为了让结论好看而改判据。
    照实报告灰区，并在论文里说明"约束的作用边界尚不清晰"，比编一个规律强。
            """)

    # 存裁决结果
    with open(os.path.join(OUT_DIR, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict,
                   "lam_dev": lam_dev_record,
                   "unsupervised_lambda": unsup_record,
                   "R1_threshold": R1_THRESHOLD}, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 主图：oracle λ 曲线
    # ------------------------------------------------------------------
    try:
        subs = [t for t in df_r["subset"].unique()]
        fig, axes = plt.subplots(2, len(subs), figsize=(7.5 * len(subs), 10), squeeze=False)

        for j, tag in enumerate(subs):
            sub = df_r[(df_r["subset"] == tag) & (df_r["method"] != "Z_全零哨兵")]

            # ---- 上排：技能分 vs λ（oracle 曲线）----
            ax = axes[0][j]
            pv = sub.pivot_table(index="lam", columns="method", values="skill", aggfunc="mean")
            for meth, col in [("B_岭回归", "#DD8452"), ("D_约束+收缩★", "#4C72B0")]:
                if meth in pv.columns:
                    ax.plot(pv.index, pv[meth], marker="o", color=col, lw=2, label=meth)
                    # 标出 oracle 最优点
                    bl = pv[meth].idxmax()
                    ax.plot(bl, pv[meth].max(), marker="*", ms=20, color=col,
                            markeredgecolor="k", zorder=5)

            # 全零哨兵：技能分恒为 0 的水平线
            ax.axhline(0, color="gray", ls="--", lw=1.5, label="Z_全零哨兵 (S=0)")

            # dev 选出的 λ：用竖线标出来（取各 N 的中位数，只是示意）
            if tag in lam_dev_record:
                lb = np.median(list(lam_dev_record[tag]["B"].values()))
                ld = np.median(list(lam_dev_record[tag]["D"].values()))
                ax.axvline(lb, color="#DD8452", ls=":", lw=1.5, alpha=.8,
                           label=f"λ_B(dev)={lb:.1e}")
                ax.axvline(ld, color="#4C72B0", ls=":", lw=1.5, alpha=.8,
                           label=f"λ_D(dev)={ld:.1e}")

            ax.set_xscale("log")
            ax.set_ylim(-0.6, 1.05)
            ax.set_xlabel("正则化强度 λ")
            ax.set_ylabel("技能分 S")
            ax.set_title(f"{tag} — λ 的 oracle 曲线\n★=在test上调λ的上界(作弊) ┆ 虚线=dev选的λ(诚实)")
            ax.grid(alpha=.3, which="both")
            ax.legend(fontsize=8)

            # ---- 下排：虚警幅度 vs λ ----
            ax2 = axes[1][j]
            pv2 = sub.pivot_table(index="lam", columns="method",
                                  values="false_alarm", aggfunc="mean")
            for meth, col in [("B_岭回归", "#DD8452"), ("D_约束+收缩★", "#4C72B0")]:
                if meth in pv2.columns:
                    ax2.plot(pv2.index, pv2[meth], marker="s", color=col, lw=2, label=meth)
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel("正则化强度 λ")
            ax2.set_ylabel("虚警幅度 (越小越好)")
            ax2.set_title(f"{tag} — 虚警幅度 vs λ\n(6个未退化部件上 |θ̂| 的均值，真值恒为0)")
            ax2.grid(alpha=.3, which="both")
            ax2.legend(fontsize=8)

        fig.suptitle("脚本17：λ 的 oracle 扫描 —— D 是【方法输了】还是【λ 选歪了】？",
                     fontsize=14)
        fig.tight_layout()
        png = os.path.join(OUT_DIR, "lambda_oracle_scan.png")
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"\n  ⭐ 主图已存: {png}")
        print("     这张图可以直接进论文（第3章 3.5.3 节：'传统做法的失败'）")
    except Exception as e:
        print("  (画图失败)", e)

    print(f"\n{'='*92}\n✅ 完成。输出目录: {os.path.abspath(OUT_DIR)}\n{'='*92}")


if __name__ == "__main__":
    main()