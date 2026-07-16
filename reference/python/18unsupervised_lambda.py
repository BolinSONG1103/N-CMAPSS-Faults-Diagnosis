# -*- coding: utf-8 -*-
"""
================================================================================
脚本 18：无监督 λ 选择（Morozov / L-curve）+ 逐点占优 + 在新 λ 下完整重评
================================================================================

【脚本17 查明了什么】

  1. 🔴 R1 被否决 —— 物理约束【没输】, 而且赢得比想象的干净:
       DS02: oracle D−B = +0.0372
       DS03: oracle D−B = +0.0384
     更重要的是【逐点占优】: 在 21 个 λ × 2 个子集 = 42 个对比点上,
     D 的技能分【100% 全部】高于 B。不存在任何一个 λ 让 B 追平 D。
     ⇒ 这个结论完全免疫"你只是调参调得好"的质疑。

  2. ✅ R2 成立 —— DS03 的崩溃是【λ 选歪了】, 不是方法输了:
       DS03  oracle λ_D = 31.6 → S = 0.9514
             dev 选的 λ_D = 0.1  → S = 0.7471    (损失 0.204!)
       而 B 的 dev 选择很稳 (31.6 → 0.9165, 损失只有 0.004)。
     机制: 约束本身已经承担了大部分正则化作用, 所以 D 的 dev-MSE 曲面
           在小 λ 区【极其平坦】(DS03: λ 从 1e-6 到 1e-2, S 只从 0.117 爬到 0.140),
           argmin 被噪声主导, 随便挑一个点。
           而 B 在小 λ 区直接崩到 −119, argmin 想跑都跑不掉。

  3. 🎯 最值钱的意外发现 —— Morozov 偏差原理几乎完美复现了 oracle:
       DS02: Morozov 选 λ=31.6 = oracle λ, 损失 0.0000
       DS03: Morozov 选 λ=10.0, S=0.9503, 距 oracle(0.9514) 只差 0.0011
     ⇒ 把 DS03 的 D 从 0.7471 救回 0.9503, 提升 +0.203, 且【完全不看真值 θ】。

  4. ❌ 零空间重合度的规律【彻底作废】:
       DS02 重合度 2.5%  → 增益 +0.0372
       DS03 重合度 27.9% → 增益 +0.0384      (几乎一模一样)
     ⇒ 增益与故障方向【无关】。不要再救这条规律。
       但这反而是更好的结论: 约束的净贡献是一个【稳定的常数】。

--------------------------------------------------------------------------------
【脚本17 的两个必须修掉的缺陷】

  缺陷A: oracle 算错了粒度。
    脚本17 的 oracle 是"跨所有 N 共用一个 λ"取最优, 而诚实数字是
    "每个 N 各用各的 λ_dev"再平均 → 出现了 B 在 DS02 上
    "诚实分 0.9102 > oracle 0.9029" 的荒谬情况。
    ⇒ 本脚本改为【逐 N 的 oracle】。

  缺陷B: Morozov 的实现是启发式的, 不能写进论文。
    脚本17 用"λ 最小时的残差范数"当噪声地板。
    这对 D 恰好能用(约束挡住了过拟合, 残差范数留在真实噪声水平),
    但对 B 完全失效(λ→0 时残差被过拟合到近乎 0, 地板是假的)。
    ⇒ 本脚本改用【教科书版】: 从 cycle 内样本直接估计噪声标准误差。

--------------------------------------------------------------------------------
【本脚本做什么】

  (1) 严格估计噪声水平 δ
      每个 (unit, cycle, 通道) 的残差是 N 个样本的均值,
      所以它的标准误差 se = (该 cycle 内的样本标准差) / √N。
      整个残差矩阵 (T×13) 的期望噪声范数:
          δ = sqrt( Σ_{t,c} se[t,c]² )

  (2) Morozov 偏差原理 (Morozov 1966; Hansen 1998, Rank-Deficient... , SIAM)
      选一个 λ, 使得   ‖Hn·θ̂(λ) − r‖_F  ≈  τ·δ
      直觉: 拟合得【比噪声还准】= 在拟合噪声(过拟合);
            差【太远】= 欠拟合。
      τ 是安全系数 (τ≥1), 用来吸收模型误差(H 的辨识误差 + 线性化误差)。
      本脚本扫 τ ∈ {1.0, 1.5, 2.0, 3.0} 做敏感性, 主结果用 τ=1.0。

  (3) L-curve (Hansen 1992, SIAM Review 34(4):561–580)
      在 (log‖残差‖, log‖解‖) 平面上找曲率最大的拐角。

  (4) 逐点占优表 —— 第3章的主图1
      在每个 (子集, N, λ) 网格点上比较 D 和 B 的平均技能分,
      统计 D 占优的比例。

  (5) ★ 在【无监督选出的 λ】下重新测 虚警 / 检出 / 早期RMSE
      ⚠️ 这一步非做不可: 脚本16 的"虚警降低 4–9 倍"是在 λ_dev 下测的,
         现在 λ 变了(D 从 0.1 变成 ~10~31.6), 那些数字【必然会变】。
         不重测就写进论文 = 数据造假。

--------------------------------------------------------------------------------
【预注册判据（跑之前写死，跑完照着对，不许改）】

  P1【约束的独立有效性】—— 第3章的核心主张
      在所有 (子集 × N × λ) 网格点上, D 的平均技能分 > B 的比例 ≥ 95%。
      → 成立则可写: "在完全相同的收缩强度下, 物理约束在整个正则化路径上一致占优。"

  P2【Morozov 可用性】—— 决定第3章有没有一个可落地的 λ 选择方案
      同时满足以下【全部】三条:
        (a) 两个子集上, D 用 Morozov(τ=1.0) 选 λ 的诚实技能分,
            相对【逐 N oracle】的损失 < 0.03
        (b) 两个子集上, D(Morozov) 的技能分 > B(用它自己最好的无监督准则)
        (c) DS03 上 D(Morozov) 的技能分 > D(λ_dev)  —— 即确实救回来了

  P3【虚警优势在新 λ 下仍然成立】—— 决定"约束的价值在特异性"这句话还能不能写
      在各自选出的 λ 下:
        (a) 两个子集上, D 的虚警幅度 < B 的虚警幅度
        (b) 两个子集上, D 的检出相关性 ≥ B 的检出相关性 − 0.01 (不显著变差)
      ⚠️ 如果 P3(a) 不成立, 说明"虚警优势"是【小 λ 的副产品】而不是【约束的功劳】,
         那么脚本16 的那段结论必须【整段删掉】。这是本脚本最可能打脸的地方。

--------------------------------------------------------------------------------
运行:  python 18_unsupervised_lambda.py
耗时:  和脚本17 差不多(同样是每个 (unit,N,λ) 解一次 NNLS)。
       先把 FAST_MODE 设 True 跑通流程, 再改 False 跑正式版。
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

# ⚠️ 字体：SimHei 没有 U+2212(真减号)和组合抑扬符(θ̂)的字形。
#    解决办法：(1) axes.unicode_minus=False 让 matplotlib 用 ASCII 减号;
#             (2) 所有标题/标签里【只用 ASCII 的 - 】, 不用 −;
#             (3) 不写 θ̂, 改写 theta_hat。
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 配置 —— 与脚本 16/17 【完全一致】, 否则结果不可比
# ============================================================================
DATA_DIR = "D:/N-CMPASS/data_set".strip()
OUT_DIR  = "./unsup_out"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

FAST_MODE = False          # True: 2个窗口 × 11个λ × 3台test机（先跑通用）

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

OP_COLS = ["TRA", "Mach", "theta_c", "delta_c"]

WINDOWS = [1, 10, 100, 1000] if not FAST_MODE else [100, 1000]
MAX_PER_CYCLE = 1000
N_REF_CYCLES  = 3
LAMBDA_GRID = np.logspace(-6, 4, 21) if not FAST_MODE else np.logspace(-6, 4, 11)
MAX_TEST_UNITS = None if not FAST_MODE else 3

# Morozov 的安全系数 τ。主结果用 1.0, 其余做敏感性。
TAU_LIST = [1.0, 1.5, 2.0, 3.0]
TAU_MAIN = 1.0

# 预注册判据的阈值（写死）
P1_THRESHOLD = 0.95      # D 占优比例 ≥ 95%
P2_LOSS_MAX  = 0.03      # Morozov 相对 oracle 的技能分损失上限
P3_DETECT_TOL = 0.01     # 检出相关性允许比 B 低多少


# ============================================================================
# 数据读取（与脚本16/17 逐字一致）
# ============================================================================
def get_names(f, key):
    """h5 里列名存成字节串, 必须 decode 成普通字符串"""
    arr = np.array(f[key]).ravel()
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def load_for_ident(path):
    """辨识 H 用: 先读小的 A_dev 拿行号, 再只读需要的行（避免把 27GB 全读进内存）"""
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
    """验证用: 按 (unit, cycle) 分组抽样（因为聚合窗口是 cycle 内部的）"""
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
# 反演器
# ============================================================================
def build_cum(T, n):
    """累加矩阵: theta = -Cum @ d, 把增量 d 变成累计退化 theta"""
    Cum = np.zeros((T * n, T * n))
    for t in range(T):
        for i in range(t + 1):
            Cum[t*n:(t+1)*n, i*n:(i+1)*n] = np.eye(n)
    return Cum


def build_M(Hn, T):
    """设计矩阵: r_hat(t) = -sum_{i<=t} Hn @ d_i"""
    m, n = Hn.shape
    M = np.zeros((T * m, T * n))
    for t in range(T):
        for i in range(t + 1):
            M[t*m:(t+1)*m, i*n:(i+1)*n] = -Hn
    return M


def solve_B(Hn, R, lam):
    """B: Tikhonov 岭回归 (Doel 1994)。无约束, 逐 cycle 独立解。"""
    A = Hn.T @ Hn + lam * np.eye(Hn.shape[1])
    Binv = np.linalg.solve(A, Hn.T)
    return R @ Binv.T


def solve_D(M, Cum, R, lam, T, n):
    """
    D: 约束 + 收缩（本文方法）
        min ||Hn@theta - r||^2 + lam*||theta||^2
        s.t. theta <= 0 , theta(t+1) <= theta(t)
    重参数化 theta = -Cum@d, d>=0 -> 约束自动满足
    Tikhonov 项用【数据增广】塞进 NNLS:
        min || [M ; sqrt(lam)*Cum] @ d - [r ; 0] ||^2   s.t. d >= 0
    M 和 Cum 不依赖 lam -> 在 lam 循环外面构造好传进来
    """
    M_aug = np.vstack([M, np.sqrt(lam) * Cum])
    y_aug = np.concatenate([R.reshape(-1), np.zeros(T * n)])
    d, _ = nnls(M_aug, y_aug, maxiter=50 * T * n)
    return -np.cumsum(d.reshape(T, n), axis=0)


def solve_Z(Hn, R):
    """全零哨兵: 什么都不预测。任何方法跟它打平, 那个指标就是坏的。"""
    return np.zeros((R.shape[0], Hn.shape[1]))


# ============================================================================
# 无监督 λ 选择（不看真值 theta）
# ============================================================================
def pick_morozov(lam_grid, res_norms, delta, tau):
    """
    Morozov 偏差原理 (Morozov 1966)

    残差范数 res(lam) 是 lam 的【单调增】函数:
      lam 很小 -> 拼命拟合 -> 残差小（在拟合噪声）
      lam 很大 -> 解被压扁 -> 残差大（欠拟合）

    偏差原理: 选【最小的 lam】使得 res(lam) >= tau * delta
      delta = 期望的噪声范数（我们从 cycle 内样本直接估出来）
      tau   = 安全系数 >= 1, 吸收模型误差（H 的辨识误差 + 线性化误差）

    直觉: 不要把残差拟合到比噪声还小 —— 那是在拟合噪声。
          在"刚好达到噪声水平"的地方停下, 是最大的正则化, 也是最稳的解。
    """
    target = tau * delta
    ok = np.where(res_norms >= target)[0]     # np.where 返回满足条件的下标
    if len(ok) == 0:
        return float(lam_grid[-1]), int(len(lam_grid) - 1)   # 全都没达到 -> 取最大 lam
    i = int(ok[0])                            # 第一个（=最小的 lam）
    return float(lam_grid[i]), i


def pick_lcurve(lam_grid, res_norms, sol_norms):
    """
    L-curve (Hansen 1992, SIAM Review 34(4):561-580)

    在 (log||res||, log||sol||) 平面上, 这条曲线通常是个 "L" 形:
      左上竖直段 = lam 太小, 解范数爆炸（噪声被放大）
      右下水平段 = lam 太大, 残差爆炸（欠拟合）
      拐角       = 两者的最佳折中

    用离散曲率公式找拐角: kappa = |x'y'' - y'x''| / (x'^2+y'^2)^{3/2}
    """
    x, y = np.log(np.maximum(res_norms, 1e-300)), np.log(np.maximum(sol_norms, 1e-300))
    dx, dy   = np.gradient(x), np.gradient(y)        # 一阶数值导数
    ddx, ddy = np.gradient(dx), np.gradient(dy)      # 二阶数值导数
    kappa = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5) + 1e-12
    i = int(np.argmax(kappa))
    return float(lam_grid[i]), i


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 96)
    print("脚本 18：无监督 lambda 选择 + 逐点占优 + 在新 lambda 下完整重评")
    print("=" * 96)
    print(f"\n  FAST_MODE = {FAST_MODE}")
    print(f"  lambda 网格 = {len(LAMBDA_GRID)} 点, {LAMBDA_GRID[0]:.1e} ~ {LAMBDA_GRID[-1]:.1e}")
    print(f"  聚合窗口    = {WINDOWS}")
    print(f"  Morozov tau = {TAU_LIST} (主结果用 tau={TAU_MAIN})")
    print("\n  【预注册判据（已写死，跑完不许改）】")
    print(f"    P1: 所有 (子集 x N x lambda) 网格点上, D 占优比例 >= {P1_THRESHOLD:.0%}")
    print(f"    P2: (a) Morozov 相对逐N-oracle 的损失 < {P2_LOSS_MAX}  在两个子集都成立")
    print( "        (b) D(Morozov) > B(其最好的无监督准则)  在两个子集都成立")
    print( "        (c) DS03 上 D(Morozov) > D(lambda_dev)   —— 确实救回来了")
    print(f"    P3: (a) D 的虚警 < B 的虚警  在两个子集都成立")
    print(f"        (b) D 的检出 >= B 的检出 - {P3_DETECT_TOL}  在两个子集都成立")
    print( "        !! P3 是最可能打脸的地方: 若 (a) 不成立, 说明'虚警优势'是小 lambda")
    print( "           的副产品而非约束的功劳, 脚本16 的那段结论必须整段删掉。")

    # ------------------------------------------------------------------
    # [1/5] 重建健康基准与 H（标准 7 步流程，与脚本16/17 一致）
    # ------------------------------------------------------------------
    print("\n\n[1/5] 重建健康基准与影响系数矩阵 H ...")
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
        m.fit(pool[OP_COLS].values, pool[c].values)
        base[c] = m
    print(f"    健康基准: {len(base)} 个 GBM 拟合完成")

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
    U, s, Vt = np.linalg.svd(Hn)
    print(f"    cond(H_n) = {s[0]/s[-1]:.1f} , sigma_min = {s[-1]:.4f} sigma")
    del data, pool

    # ------------------------------------------------------------------
    # 窗口构造 —— ★ 新增：同时返回【噪声标准误差 SE】
    # ------------------------------------------------------------------
    def build_windows(df, unit):
        """
        返回 {N: (R, TH, SE, delta)}
          R     (T,13)  归一化残差（相对该单元零点）
          TH    (T,9)   真值 theta/量程（相对该单元零点）
          SE    (T,13)  每个残差元素的【标准误差】= cycle内样本std / sqrt(N)
          delta 标量     整个残差矩阵的期望噪声范数 = sqrt(sum(SE^2))

        ★ 为什么 SE 这么算:
          R[t,c] 是 N 个样本的【均值】。
          由中心极限定理, 均值的标准误差 = 单样本标准差 / sqrt(N)。
          单样本标准差用【该 cycle 内所有可用样本】估计（不只是抽中的 N 个,
          这样 N=1 时也有定义, 而且估计更稳）。

        ⚠️ 已知的近似（论文里要写清楚）:
          零点 b_unit 本身也有噪声, 严格说应该并进来。但 b_unit 是
          3 个 cycle × 上千样本的均值, 其标准误差比单个 cycle 小一个量级,
          忽略是安全的。

        ⚠️ 零点 = 最初 N_REF_CYCLES 个 cycle, 不是"健康期均值"。
          脚本14 的 bug: 健康期内 theta 本就在缓慢下降, 用均值当零点会让
          健康期前半段 theta_relative > 0, 而 C/D 强制 theta <= 0
          -> 真值落在可行域【外面】-> 方法被自己判了死刑。
        """
        du = df[df["unit"] == unit].copy()
        cycles = sorted(du["cycle"].unique())
        m_ref = du["cycle"].isin(cycles[:N_REF_CYCLES])
        b_unit = du.loc[m_ref, resid_cols].mean().values
        th0 = (du.loc[m_ref, THETA9].values / span).mean(axis=0)

        # 先算每个 cycle 内、每个通道的【单样本标准差】（用全部可用样本）
        std_per_cycle = {}
        for c in cycles:
            sub = du[du["cycle"] == c]
            # ddof=1 -> 样本标准差（无偏）。样本数<2 时用 0 兜底。
            sd = sub[resid_cols].values.std(axis=0, ddof=1) if len(sub) > 1 \
                 else np.zeros(len(resid_cols))
            std_per_cycle[c] = np.nan_to_num(sd, nan=0.0)

        out = {}
        for N in WINDOWS:
            R_list, TH_list, SE_list = [], [], []
            for c in cycles:
                sub = du[du["cycle"] == c]
                k = min(N, len(sub))                       # 实际能抽到几个
                idx = rng.choice(len(sub), k, replace=False)
                R_list.append(sub[resid_cols].values[idx].mean(axis=0) - b_unit)
                TH_list.append(sub[THETA9].values[0] / span - th0)
                # 标准误差 = 单样本std / sqrt(实际抽样数)
                SE_list.append(std_per_cycle[c] / np.sqrt(max(k, 1)))
            R  = np.array(R_list)
            TH = np.array(TH_list)
            SE = np.array(SE_list)
            delta = float(np.sqrt(np.sum(SE ** 2)))        # 期望噪声范数（Frobenius）
            out[N] = (R, TH, SE, delta)
        return out

    # ------------------------------------------------------------------
    # [2/5] 逐子集：dev 选 λ（作对照） + test 全 λ 扫描（记录一切）
    # ------------------------------------------------------------------
    rows = []            # 逐 (子集,unit,N,方法,λ) 的完整记录
    lam_dev_rec = {}     # dev 选出的 λ（脚本16/17 的做法，作为对照）
    delta_rec = {}       # 噪声水平记录

    for tag, (fn, true_faults) in VALID_FILES.items():
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"\n  !! 缺 {fn}, 跳过 {tag}")
            continue

        print("\n\n" + "#" * 96)
        print(f"# {tag}    真实退化部件: {true_faults}")
        print("#" * 96)

        ds = {}
        for split in ["dev", "test"]:
            d = load_per_cycle(p, split)
            d, _ = add_corrected(d, ref)
            ds[split] = pd.concat([d, residual(d) / resid_std], axis=1)

        idx_tf = [THETA9.index(q) for q in true_faults]     # 真退化部件下标
        other  = [i for i in range(9) if i not in idx_tf]   # 未退化部件（看虚警）

        # ---------- (a) dev 上选 λ（对照组：脚本16/17 的做法）----------
        dev_units = sorted(ds["dev"]["unit"].unique())[:3]
        dev_wins = {u: build_windows(ds["dev"], u) for u in dev_units}
        print(f"\n  --- (a) 对照组: 在 {tag}-dev 上监督式选 lambda ---")
        lamB_dev, lamD_dev = {}, {}
        for N in WINDOWS:
            errB, errD = [], []
            for lam in LAMBDA_GRID:
                eB, eD = [], []
                for u in dev_units:
                    R, TH, _, _ = dev_wins[u][N]
                    T = len(R)
                    M, Cum = build_M(Hn, T), build_cum(T, 9)
                    eB.append(np.mean((solve_B(Hn, R, lam) - TH) ** 2))
                    eD.append(np.mean((solve_D(M, Cum, R, lam, T, 9) - TH) ** 2))
                errB.append(np.mean(eB)); errD.append(np.mean(eD))
            lamB_dev[N] = float(LAMBDA_GRID[int(np.argmin(errB))])
            lamD_dev[N] = float(LAMBDA_GRID[int(np.argmin(errD))])
            print(f"    N={N:>5d}   lam_B(dev)={lamB_dev[N]:.2e}   lam_D(dev)={lamD_dev[N]:.2e}")
        lam_dev_rec[tag] = {"B": lamB_dev, "D": lamD_dev}

        # ---------- (b) test 全 λ 扫描 ----------
        test_units = sorted(ds["test"]["unit"].unique())
        if MAX_TEST_UNITS is not None:
            test_units = test_units[:MAX_TEST_UNITS]

        n_solve = len(test_units) * len(WINDOWS) * len(LAMBDA_GRID)
        print(f"\n  --- (b) 在 {tag}-test 上扫描全部 lambda ---")
        print(f"      test 单元: {[int(u) for u in test_units]}")
        print(f"      共需 {n_solve} 次 NNLS, 请耐心 ...")

        test_wins = {u: build_windows(ds["test"], u) for u in test_units}
        delta_rec[tag] = {}

        for u in test_units:
            for N in WINDOWS:
                R, TH, SE, delta = test_wins[u][N]
                T = len(R)
                M, Cum = build_M(Hn, T), build_cum(T, 9)

                delta_rec[tag].setdefault(str(N), []).append(delta)

                mse_zero = float(np.mean(TH ** 2))          # 全零哨兵的 MSE
                n_early = max(1, int(0.3 * T))              # 早期 = 前 30% cycle

                # ---- 全零哨兵（λ 无关，只记一次）----
                TH_z = solve_Z(Hn, R)
                rmse_early_z = float(np.sqrt(np.mean((TH_z - TH)[:n_early] ** 2)))
                rows.append(dict(
                    subset=tag, unit=int(u), N=N, method="Z_zero", lam=np.nan,
                    skill=0.0, rmse=float(np.sqrt(mse_zero)),
                    rmse_early=rmse_early_z, false_alarm=0.0, detect_corr=0.0,
                    res_norm=float(np.linalg.norm(R)), sol_norm=0.0, delta=delta))

                # ---- 扫 λ ----
                for lam in LAMBDA_GRID:
                    for name, TH_hat in [("B", solve_B(Hn, R, lam)),
                                         ("D", solve_D(M, Cum, R, lam, T, 9))]:
                        err = TH_hat - TH
                        mse = float(np.mean(err ** 2))

                        # 技能分 S = 1 - MSE/MSE(全零)
                        #   1  = 完美 ; 0 = 跟"什么都不做"一样(哨兵线) ; <0 = 更糟
                        skill = 1.0 - mse / max(mse_zero, 1e-12)

                        # 早期用【绝对RMSE】不用技能分:
                        #   早期 theta_true ≈ 0 -> 技能分分母趋零 -> 被除爆, 指标无定义
                        rmse_early = float(np.sqrt(np.mean(err[:n_early] ** 2)))

                        # 虚警幅度: 6 个未退化部件上 |theta_hat| 的均值（真值恒为 0）
                        false_alarm = float(np.mean(np.abs(TH_hat[:, other])))

                        # 检出相关性: 3 个真退化部件上 theta_hat 与真值的相关系数
                        cs = []
                        for i in idx_tf:
                            if np.std(TH[:, i]) > 1e-9 and np.std(TH_hat[:, i]) > 1e-9:
                                cs.append(np.corrcoef(TH_hat[:, i], TH[:, i])[0, 1])
                        detect = float(np.mean(cs)) if cs else 0.0

                        # 无监督准则需要的两个范数（不看真值！）
                        res_norm = float(np.linalg.norm(TH_hat @ Hn.T - R))
                        sol_norm = float(np.linalg.norm(TH_hat))

                        rows.append(dict(
                            subset=tag, unit=int(u), N=N, method=name, lam=float(lam),
                            skill=skill, rmse=float(np.sqrt(mse)), rmse_early=rmse_early,
                            false_alarm=false_alarm, detect_corr=detect,
                            res_norm=res_norm, sol_norm=sol_norm, delta=delta))
            print(f"      unit {int(u)} 完成")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "raw_scan.csv"), index=False, encoding="utf-8-sig")
    print(f"\n\n[2/5] 原始扫描已存: raw_scan.csv  ({len(df)} 行)")

    # ------------------------------------------------------------------
    # [3/5] 逐 (unit, N) 应用无监督准则选 λ，并汇总
    # ------------------------------------------------------------------
    print("\n[3/5] 应用无监督准则选 lambda ...")

    sel_rows = []
    for (tag, u, N, meth), g in df[df["method"] != "Z_zero"].groupby(
            ["subset", "unit", "N", "method"]):
        g = g.sort_values("lam")
        lam_grid = g["lam"].values
        res = g["res_norm"].values
        sol = g["sol_norm"].values
        delta = float(g["delta"].iloc[0])

        # --- L-curve ---
        lam_lc, i_lc = pick_lcurve(lam_grid, res, sol)

        # --- Morozov（各 τ）---
        picks = {"L_curve": (lam_lc, i_lc)}
        for tau in TAU_LIST:
            lam_mz, i_mz = pick_morozov(lam_grid, res, delta, tau)
            picks[f"Morozov_tau{tau}"] = (lam_mz, i_mz)

        # --- 逐 N 的 oracle（作弊，只作上界参考）---
        i_or = int(np.argmax(g["skill"].values))
        picks["oracle"] = (float(lam_grid[i_or]), i_or)

        # --- dev 选的 λ（对照）---
        if tag in lam_dev_rec:
            lam_dv = lam_dev_rec[tag][meth][N]
            i_dv = int(np.argmin(np.abs(lam_grid - lam_dv)))
            picks["lam_dev"] = (float(lam_grid[i_dv]), i_dv)

        for crit, (lam_sel, i_sel) in picks.items():
            r = g.iloc[i_sel]
            sel_rows.append(dict(
                subset=tag, unit=int(u), N=N, method=meth, criterion=crit,
                lam=lam_sel, skill=float(r["skill"]), rmse=float(r["rmse"]),
                rmse_early=float(r["rmse_early"]),
                false_alarm=float(r["false_alarm"]),
                detect_corr=float(r["detect_corr"]), delta=delta))

    sel = pd.DataFrame(sel_rows)
    sel.to_csv(os.path.join(OUT_DIR, "selected_lambda.csv"),
               index=False, encoding="utf-8-sig")

    # 噪声水平报告
    print("\n  --- 估计的噪声水平 delta（残差矩阵的期望噪声 Frobenius 范数）---")
    for tag in delta_rec:
        for N in sorted(delta_rec[tag], key=int):
            v = np.mean(delta_rec[tag][N])
            print(f"    {tag}  N={int(N):>5d}   delta = {v:.4f}")
    print("    （N 每 x10, delta 应约 /3.16 = sqrt(10)。若不是, 说明噪声估计有问题。）")

    # ------------------------------------------------------------------
    # [4/5] 报告
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 96)
    print("[4/5] 结　果")
    print("=" * 96)

    # ---- 表1: 逐点占优 ----
    print("\n\n" + "#" * 96)
    print("# 表1  逐点占优 —— 在每个 (子集, N, lambda) 上, D 是否优于 B ?")
    print("#      ★ 这是第3章的主图1: 证明约束的价值【不是调参调出来的】")
    print("#" * 96)

    grid = df[df["method"] != "Z_zero"].pivot_table(
        index=["subset", "N", "lam"], columns="method", values="skill", aggfunc="mean")
    grid = grid.dropna()
    grid["D_win"] = grid["D"] > grid["B"]
    grid["gap"] = grid["D"] - grid["B"]

    print(f"\n  {'子集':<6s} {'N':>6s}  {'网格点数':>8s}  {'D占优数':>8s}  {'占优率':>8s}  "
          f"{'gap中位数':>10s}  {'gap最小':>10s}")
    print("  " + "-" * 70)
    for (tag, N), gg in grid.groupby(level=[0, 1]):
        nw = int(gg["D_win"].sum()); nt = len(gg)
        print(f"  {tag:<6s} {N:>6d}  {nt:>8d}  {nw:>8d}  {nw/nt:>7.1%}  "
              f"{gg['gap'].median():>+10.4f}  {gg['gap'].min():>+10.4f}")

    n_win_all = int(grid["D_win"].sum()); n_tot_all = len(grid)
    win_rate = n_win_all / n_tot_all
    print("  " + "-" * 70)
    print(f"  {'总计':<6s} {'':>6s}  {n_tot_all:>8d}  {n_win_all:>8d}  {win_rate:>7.1%}")

    # ---- 表2: 各 λ 选择准则的诚实技能分 ----
    print("\n\n" + "#" * 96)
    print("# 表2  各 lambda 选择准则下的【诚实】技能分（跨 test 单元与 N 取平均）")
    print("#      oracle 是作弊的上界, 仅作参考; 其余都是可落地的")
    print("#" * 96)

    crit_order = (["oracle", "lam_dev", "L_curve"] +
                  [f"Morozov_tau{t}" for t in TAU_LIST])
    pv = sel.pivot_table(index=["subset", "criterion"], columns="method",
                         values="skill", aggfunc="mean")

    for tag in sel["subset"].unique():
        print(f"\n  --- {tag} ---")
        print(f"    {'准则':<16s} {'B_岭回归':>10s} {'D_约束+收缩':>12s} "
              f"{'D-B':>9s} {'D距oracle':>11s}")
        print("    " + "-" * 62)
        d_oracle = pv.loc[(tag, "oracle"), "D"]
        for c in crit_order:
            if (tag, c) not in pv.index:
                continue
            b = pv.loc[(tag, c), "B"]; d = pv.loc[(tag, c), "D"]
            mark = "  <- 上界(作弊)" if c == "oracle" else ""
            mark = "  <- 脚本16/17的做法" if c == "lam_dev" else mark
            mark = "  ★主结果" if c == f"Morozov_tau{TAU_MAIN}" else mark
            print(f"    {c:<16s} {b:>10.4f} {d:>12.4f} {d-b:>+9.4f} "
                  f"{d-d_oracle:>+11.4f}{mark}")

    # ---- 表3: 在新 λ 下重测虚警/检出/早期 ----
    print("\n\n" + "#" * 96)
    print("# 表3  ★ 在【无监督选出的 lambda】下重测 虚警 / 检出 / 早期RMSE")
    print("#      !! 脚本16 的'虚警降 4-9 倍'是在 lam_dev 下测的, lambda 变了必须重测")
    print("#" * 96)

    zero_ref = df[df["method"] == "Z_zero"].groupby("subset")["rmse_early"].mean()

    for tag in sel["subset"].unique():
        print(f"\n  --- {tag}  (全零哨兵的早期RMSE = {zero_ref[tag]:.4f}) ---")
        print(f"    {'准则':<16s} {'方法':<4s} {'虚警幅度':>10s} {'检出相关':>10s} "
              f"{'早期RMSE':>10s} {'技能分':>9s}")
        print("    " + "-" * 66)
        for c in [f"Morozov_tau{TAU_MAIN}", "L_curve", "lam_dev", "oracle"]:
            for meth in ["B", "D"]:
                m = sel[(sel.subset == tag) & (sel.criterion == c) &
                        (sel.method == meth)]
                if len(m) == 0:
                    continue
                flag = ""
                if m["rmse_early"].mean() > zero_ref[tag]:
                    flag = "  !!比哨兵还差"
                print(f"    {c:<16s} {meth:<4s} {m['false_alarm'].mean():>10.4f} "
                      f"{m['detect_corr'].mean():>10.4f} {m['rmse_early'].mean():>10.4f} "
                      f"{m['skill'].mean():>9.4f}{flag}")

    # ------------------------------------------------------------------
    # [5/5] 裁决
    # ------------------------------------------------------------------
    print("\n\n" + "#" * 96)
    print("# [5/5] 裁　决（严格按预注册判据，不许事后改）")
    print("#" * 96)

    tags = sorted(sel["subset"].unique())
    verdict = {}

    # ---- P1 ----
    P1 = win_rate >= P1_THRESHOLD
    print(f"\n  --- P1: D 在所有 (子集 x N x lambda) 网格点上占优比例 >= {P1_THRESHOLD:.0%} ? ---")
    print(f"      实际占优率 = {win_rate:.1%}  ({n_win_all}/{n_tot_all})")
    print(f"      最小 gap = {grid['gap'].min():+.4f}   中位 gap = {grid['gap'].median():+.4f}")
    print(f"      {'[OK] P1 成立' if P1 else '[X] P1 不成立'}")

    # ---- P2 ----
    mz = f"Morozov_tau{TAU_MAIN}"
    p2a, p2b, p2c = True, True, True
    print(f"\n  --- P2: Morozov(tau={TAU_MAIN}) 的可用性 ---")
    for tag in tags:
        d_mz = pv.loc[(tag, mz), "D"]
        d_or = pv.loc[(tag, "oracle"), "D"]
        loss = d_or - d_mz
        ok = loss < P2_LOSS_MAX
        p2a &= ok
        print(f"      (a) {tag}: D(Morozov)={d_mz:.4f}  oracle={d_or:.4f}  "
              f"损失={loss:.4f}  {'[OK]' if ok else '[X]'}")

    for tag in tags:
        d_mz = pv.loc[(tag, mz), "D"]
        # B 的"最好的无监督准则"（不含 oracle 和 lam_dev）
        b_unsup = max(pv.loc[(tag, c), "B"] for c in crit_order
                      if c not in ("oracle", "lam_dev") and (tag, c) in pv.index)
        ok = d_mz > b_unsup
        p2b &= ok
        print(f"      (b) {tag}: D(Morozov)={d_mz:.4f}  vs  B(最好无监督)={b_unsup:.4f}  "
              f"{'[OK]' if ok else '[X]'}")

    if "DS03" in tags:
        d_mz = pv.loc[("DS03", mz), "D"]
        d_dv = pv.loc[("DS03", "lam_dev"), "D"]
        p2c = d_mz > d_dv
        print(f"      (c) DS03: D(Morozov)={d_mz:.4f}  vs  D(lam_dev)={d_dv:.4f}  "
              f"提升={d_mz-d_dv:+.4f}  {'[OK]' if p2c else '[X]'}")

    P2 = p2a and p2b and p2c
    print(f"      {'[OK] P2 成立' if P2 else '[X] P2 不成立'}")

    # ---- P3 ----
    p3a, p3b = True, True
    print(f"\n  --- P3: 虚警优势在【新 lambda】下是否仍然成立 ? （最可能打脸）---")
    for tag in tags:
        fa_B = sel[(sel.subset == tag) & (sel.criterion == mz) &
                   (sel.method == "B")]["false_alarm"].mean()
        fa_D = sel[(sel.subset == tag) & (sel.criterion == mz) &
                   (sel.method == "D")]["false_alarm"].mean()
        dc_B = sel[(sel.subset == tag) & (sel.criterion == mz) &
                   (sel.method == "B")]["detect_corr"].mean()
        dc_D = sel[(sel.subset == tag) & (sel.criterion == mz) &
                   (sel.method == "D")]["detect_corr"].mean()
        oa = fa_D < fa_B
        ob = dc_D >= dc_B - P3_DETECT_TOL
        p3a &= oa; p3b &= ob
        ratio = fa_B / max(fa_D, 1e-12)
        print(f"      {tag}: 虚警 B={fa_B:.4f} -> D={fa_D:.4f}  "
              f"(D 是 B 的 1/{ratio:.1f})  {'[OK]' if oa else '[X]'}")
        print(f"             检出 B={dc_B:.4f} -> D={dc_D:.4f}  {'[OK]' if ob else '[X]'}")
    P3 = p3a and p3b
    print(f"      {'[OK] P3 成立' if P3 else '[X] P3 不成立'}")

    verdict = {"P1": bool(P1), "P1_win_rate": float(win_rate),
               "P2": bool(P2), "P2a": bool(p2a), "P2b": bool(p2b), "P2c": bool(p2c),
               "P3": bool(P3), "P3a": bool(p3a), "P3b": bool(p3b),
               "min_gap": float(grid["gap"].min()),
               "median_gap": float(grid["gap"].median())}

    # ---- 结论 ----
    print("\n\n  " + "=" * 92)
    if P1 and P2 and P3:
        print("  [GREEN] P1 + P2 + P3 全部成立 —— 第3章的三条主张全部立住")
        print("  " + "=" * 92)
        print(f"""
    可以直接写进论文的三句话:

    1. 【约束的独立有效性】
       在完全相同的收缩强度下, 物理约束(非正性 + 单调性)在【整个正则化路径上
       一致占优】: {n_tot_all} 个 (子集 x 窗口 x lambda) 网格点中, D 优于 B 的
       比例为 {win_rate:.1%}, 最小增益 {grid['gap'].min():+.4f}。
       ⇒ 该增益【不可能】归因于超参数调优, 只能归因于约束本身。

    2. 【可落地的 lambda 选择方案】
       约束会使监督式 dev 选择失效(DS03 上损失 0.204), 但基于偏差原理
       (Morozov)的【无监督】选择几乎复现 oracle(损失 < {P2_LOSS_MAX}), 且完全不需要真值。
       ⇒ 这是本文提出的、可直接部署的诊断流程。

    3. 【约束的作用边界】
       在无监督选出的 lambda 下, 约束仍然显著降低虚警且不损失检出。
       ⇒ 约束的价值【同时】体现在精度和特异性上。

    下一步 -> 做 H(w) = H0 * diag(g_phi(w)) 的工况自适应实验（脚本19）。
        """)
    else:
        print("  [MIXED] 部分判据未通过 —— 逐条处理, 不许粉饰")
        print("  " + "=" * 92)
        if not P1:
            print("""
    [X] P1 不成立: 逐点占优率 < 95%。
        ⇒ "约束在整个正则化路径上一致占优" 这句话【不能写】。
        ⇒ 退而求其次: 只报 oracle 差距 (+0.037/+0.038), 但必须承认
           这依赖 lambda 的选择, 说服力弱一个量级。
        ⇒ 先去看逐 N 的占优率: 是不是只有 N=1(噪声最大)那一档在拖后腿?
           如果是, 就限定"在 N>=10 的实用工况下一致占优", 并说明原因。
            """)
        if not P2:
            print(f"""
    [X] P2 不成立: Morozov 不好用。
        ⇒ 先看 tau 的敏感性表(表2): 是不是 tau=1.0 太激进?
           模型误差(H 的辨识误差 + 线性化误差)会抬高残差地板,
           所以真实的 tau 可能需要 1.5~3.0。如果某个 tau 好用, 就用那个,
           但【必须在论文里说明 tau 是怎么定的, 不能事后挑】。
           诚实做法: 在 dev 上定 tau(只有一个标量, 泛化风险远小于选 lambda),
                     然后 test 上用。这仍然比监督式选 lambda 干净得多。
        ⇒ 如果所有 tau 都不行, 改用 L-curve; 如果 L-curve 也不行,
           那就必须上 deep unfolding (让 lambda 端到端学出来)。
            """)
        if not P3:
            print("""
    [X] P3 不成立: 虚警优势在新 lambda 下消失了!
        ⇒ 这说明脚本16 里"虚警降低 4-9 倍"是【小 lambda 的副产品】,
           不是【约束的功劳】—— 因为脚本16 里 D 用的 lambda(DS03 上是 0.1)
           比 B 的(31.6)小 300 倍, 小 lambda 本身就会让解更稀疏、更接近 0。
        ⇒ 🔴 脚本16 关于虚警的那整段结论【必须从论文里删掉】。
           不要试图保留, 不要试图换个说法保留。
        ⇒ 但 P1(逐点占优) 如果成立, 第3章依然立得住 ——
           只是主张从"约束提升特异性"收缩为"约束提升精度"。
            """)

    with open(os.path.join(OUT_DIR, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict,
                   "lam_dev": lam_dev_rec,
                   # 修复: pv 是 MultiIndex(subset, criterion)，直接 to_dict() 会产生元组 key，
                   #       JSON 不支持。reset_index() 把两级索引降成普通列，
                   #       再用 orient="records" 导出成 [{subset:..., criterion:..., B:..., D:...}, ...]
                   #       —— 既能被 JSON 序列化，人读起来也更清楚。
                   "skill_table": pv.round(4).reset_index().to_dict(orient="records")},
                  f, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------
    # 图1（主图）: 逐点占优
    # ------------------------------------------------------------------
    try:
        subs = tags
        fig, axes = plt.subplots(2, len(subs), figsize=(7.5 * len(subs), 10), squeeze=False)

        for j, tag in enumerate(subs):
            g = df[(df.subset == tag) & (df.method != "Z_zero")]

            # -- 上: 技能分 vs lambda, 各 N 一条线, B 虚线 D 实线 --
            ax = axes[0][j]
            cmap = plt.get_cmap("viridis")
            for k, N in enumerate(WINDOWS):
                col = cmap(k / max(len(WINDOWS) - 1, 1))
                gg = g[g.N == N].pivot_table(index="lam", columns="method",
                                             values="skill", aggfunc="mean")
                ax.plot(gg.index, gg["B"], ls="--", color=col, marker="o", ms=4,
                        alpha=.85, label=f"B  N={N}")
                ax.plot(gg.index, gg["D"], ls="-", color=col, marker="^", ms=5,
                        lw=2, label=f"D  N={N}")
            ax.axhline(0, color="gray", ls=":", lw=1.5, label="全零哨兵 (S=0)")
            ax.set_xscale("log")
            ax.set_ylim(-0.3, 1.02)
            ax.set_xlabel("正则化强度 lambda")
            ax.set_ylabel("技能分 S")
            ax.set_title(f"{tag}  逐点占优\n实线=D(约束+收缩)  虚线=B(纯收缩)  "
                         f"D 在每个 lambda 上都在 B 之上")
            ax.grid(alpha=.3, which="both")
            ax.legend(fontsize=7, ncol=2)

            # -- 下: gap = D - B, 跨 N --
            ax2 = axes[1][j]
            for k, N in enumerate(WINDOWS):
                col = cmap(k / max(len(WINDOWS) - 1, 1))
                gg = g[g.N == N].pivot_table(index="lam", columns="method",
                                             values="skill", aggfunc="mean")
                ax2.plot(gg.index, gg["D"] - gg["B"], color=col, marker="s", ms=5,
                         lw=2, label=f"N={N}")
            ax2.axhline(0, color="k", ls="--", lw=1.5)
            ax2.set_xscale("log")
            ax2.set_yscale("symlog", linthresh=0.1)   # symlog 能显示负值
            ax2.set_xlabel("正则化强度 lambda")
            ax2.set_ylabel("技能分增益  D - B")
            ax2.set_title(f"{tag}  约束的净贡献 (D - B)\n始终在 0 线之上 = 约束一致有效")
            ax2.grid(alpha=.3, which="both")
            ax2.legend(fontsize=8)

        fig.suptitle("脚本18 图1：物理约束的逐点占优 —— 增益不依赖 lambda 的选择",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig1_pointwise_dominance.png"), dpi=150)
        plt.close(fig)
        print("\n  [*] 图1 (主图): fig1_pointwise_dominance.png")
    except Exception as e:
        print("  (图1 失败)", e)

    # ------------------------------------------------------------------
    # 图2: 各 λ 选择准则的对比
    # ------------------------------------------------------------------
    try:
        fig, axes = plt.subplots(1, len(tags), figsize=(7 * len(tags), 5.5), squeeze=False)
        crits = ["oracle", "lam_dev", "L_curve"] + [f"Morozov_tau{t}" for t in TAU_LIST]
        x = np.arange(len(crits))
        w = 0.36
        for j, tag in enumerate(tags):
            ax = axes[0][j]
            vb = [pv.loc[(tag, c), "B"] if (tag, c) in pv.index else np.nan for c in crits]
            vd = [pv.loc[(tag, c), "D"] if (tag, c) in pv.index else np.nan for c in crits]
            ax.bar(x - w/2, vb, w, label="B_岭回归", color="#DD8452")
            ax.bar(x + w/2, vd, w, label="D_约束+收缩", color="#4C72B0")
            d_or = pv.loc[(tag, "oracle"), "D"]
            ax.axhline(d_or, color="#4C72B0", ls=":", lw=1.5,
                       label=f"D 的 oracle 上界 = {d_or:.3f}")
            ax.set_xticks(x)
            ax.set_xticklabels(crits, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("诚实技能分 S")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{tag}  各 lambda 选择准则对比\n"
                         f"(oracle 是作弊上界; 其余都是可落地的)")
            ax.grid(alpha=.3, axis="y")
            ax.legend(fontsize=8)
        fig.suptitle("脚本18 图2：无监督 lambda 选择能否复现 oracle ？", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "fig2_lambda_criteria.png"), dpi=150)
        plt.close(fig)
        print("  [*] 图2: fig2_lambda_criteria.png")
    except Exception as e:
        print("  (图2 失败)", e)

    print(f"\n{'='*96}\n完成。输出目录: {os.path.abspath(OUT_DIR)}\n{'='*96}")


if __name__ == "__main__":
    main()