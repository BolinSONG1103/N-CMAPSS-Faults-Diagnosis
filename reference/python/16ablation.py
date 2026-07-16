# -*- coding: utf-8 -*-
"""
================================================================================
脚本 16：2×2 消融 —— 物理约束的【净贡献】到底是多少？
================================================================================

【脚本15 暴露的问题】
  C(纯约束) 的技能分 −0.25, 惨败给 B(岭回归) 的 0.91。
  但同一批数据里, C 的【虚警幅度】只有 0.0011~0.0041, 比 B 的 0.0057~0.0158
  好 4~5 倍; 检出相关性 0.92~0.98, 和 B 打平。

  ⇒ C 能【准确找到是哪个部件坏、几乎不虚报】, 但【算不准坏了多少】。
  ⇒ 原因: C 完全没有【收缩项】。非正+单调只是一个【锥形可行域】——
     它挡住了"自愈", 但在锥的【内部】, 解沿病态方向仍可自由乱跑。
     C 本质上是"被关进锥里的无约束最小二乘"。
     (证据: A 的技能分 −367, C 是 −0.25 —— 锥确实救回了大部分, 但不够。)

  ⇒ 而且: 交接文档的原方案有【三条】约束 λ₁/λ₂/λ₃, 其中 λ₁ 是【稀疏性】,
     本身就是一个收缩型正则化。脚本14/15 只实现了 λ₂(非正)+λ₃(单调),
     把 λ₁ 整个漏掉了。所以前两轮【根本没在测原方案】, 而且是对原方案不公平。

【本脚本的实验设计: 一个干净的 2×2 消融】

                   无收缩项              有收缩项(相同的 L2)
      无约束        A (经典GPA)           B (Tikhonov岭回归, Doel 1994)
      有约束        C (纯锥约束)          D (★本文方法: 约束 + 收缩)

  · B − A  = 收缩的贡献 (在无约束下)
  · C − A  = 约束的贡献 (在无收缩下)
  · ★ D − B = 【在收缩项完全相同的条件下, 物理约束的净贡献】
              这是唯一能干净回答"约束到底有没有【独立】价值"的比较。
              λ_D 和 λ_B 用【同样的方式、同样的网格、同样的dev数据】各自调到最优。

【方法D 的求解: 增广 NNLS, 精确解, 不需要 cvxpy】
    目标:  min ‖H_nθ − Δx‖² + λ‖θ‖²   s.t.  θ ≤ 0 , θ(t+1) ≤ θ(t)
    重参数化: θ = −Cum·d ,  d ≥ 0   (Cum = 分块下三角的"累加矩阵")
              → 两个约束自动满足
    Tikhonov 项 λ‖θ‖² = λ‖Cum·d‖²  可以直接【增广到设计矩阵】里:
              min ‖ [ M ; √λ·Cum ] d − [ y ; 0 ] ‖²   s.t. d ≥ 0
    这仍然是一个标准 NNLS → scipy.optimize.nnls 精确求解。

【另一个修复: "早期技能分"是坏指标】
    早期 cycle 的 θ_true ≈ 0 (相对最初几个cycle), 所以技能分的分母 MSE(全零)
    趋近于 0 → 技能分被除爆 (脚本15 里 B 也是 −10.9)。这个指标在早期【没有定义】。
    ⇒ 早期改用【绝对 RMSE】, 不除以任何趋近零的分母。

--------------------------------------------------------------------------------
【预注册预测 (跑之前写死, 跑完照着对)】

  Q4: 方法D 的技能分 > 方法B, 同时保住 C 的低虚警 (≈0.001~0.004)。
  Q5: D−B 的增益随【零空间重合度】增大 (DS03 的增益 > DS02 的增益)。

  🔴 Q6 (最关键, 也是最可能打脸的):
      若 D 只是【打平】B (技能分增益 < 0.02) 且虚警没有改善 ——
      那么"物理约束"就只是通用正则化的一个更麻烦的写法, 核心卖点作废。
      这次不会再找借口: 收缩项已经补上了, 三个bug都修了, 两个验证集都跑了。
--------------------------------------------------------------------------------
运行: python 16_ablation_2x2.py
"""

import os
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
plt.rcParams["axes.unicode_minus"] = False

# ============================================================================
# 配置
# ============================================================================
DATA_DIR = "D:/N-CMPASS/data_set".strip()
OUT_DIR = "./ablation_out"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

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

WINDOWS = [1, 10, 100, 1000]
MAX_PER_CYCLE = 1000
N_REF_CYCLES = 3                       # 零点 = 最初 3 个 cycle (脚本15 已验证可行)
LAMBDA_GRID = np.logspace(-6, 4, 21)   # B 和 D 共用同一个网格 (公平)


# ============================================================================
# 数据读取 / 特征（和脚本15 完全一致）
# ============================================================================
def get_names(f, key):
    """h5 里列名存成字节串, 必须 decode 成普通字符串"""
    arr = np.array(f[key]).ravel()
    return [v.decode() if isinstance(v, bytes) else str(v) for v in arr]


def load_for_ident(path):
    """辨识 H 用的随机抽样读取（先读小的 A 拿行号，再只读需要的行）"""
    with h5py.File(path, "r") as f:
        A_var, W_var = get_names(f, "A_var"), get_names(f, "W_var")
        Xs_var, T_var = get_names(f, "X_s_var"), get_names(f, "T_var")
        df_A = pd.DataFrame(np.array(f["A_dev"]), columns=A_var)
        ih = np.where(df_A["hs"].values == 1)[0]     # 健康样本行号
        idg = np.where(df_A["hs"].values == 0)[0]    # 退化样本行号
        take = np.sort(np.concatenate([
            rng.choice(ih, min(N_HEALTHY_ID, len(ih)), replace=False),
            rng.choice(idg, min(N_DEGRADED_ID, len(idg)), replace=False)]))
        W, Xs, T = f["W_dev"][take, :], f["X_s_dev"][take, :], f["T_dev"][take, :]
    return pd.concat([df_A.iloc[take].reset_index(drop=True),
                      pd.DataFrame(W, columns=W_var),
                      pd.DataFrame(Xs, columns=Xs_var),
                      pd.DataFrame(T, columns=T_var)], axis=1)


def load_per_cycle(path, split):
    """按 (unit, cycle) 分组抽样。窗口是【cycle 内】的, 所以必须这样读。"""
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
                      pd.DataFrame(W, columns=W_var),
                      pd.DataFrame(Xs, columns=Xs_var),
                      pd.DataFrame(T, columns=T_var)], axis=1)


def add_corrected(df, ref):
    """相似理论修正参数 (Walsh & Fletcher 2004, referred parameters)"""
    df = df.copy()
    df["theta_c"] = df["T2"] / ref["T_ref"]      # 无量纲总温
    df["delta_c"] = df["P2"] / ref["P_ref"]      # 无量纲总压
    sq = np.sqrt(df["theta_c"])
    cols = []
    for c in ["Nf", "Nc"]:                                   # 修正转速 N/√θ
        df[f"{c}_c"] = df[c] / sq; cols.append(f"{c}_c")
    df["Wf_c"] = df["Wf"] / (df["delta_c"] * sq); cols.append("Wf_c")
    for c in ["T24", "T30", "T48", "T50"]:                   # 温比
        df[f"{c}_c"] = df[c] / df["T2"]; cols.append(f"{c}_c")
    for c in ["P15", "P21", "P24", "Ps30", "P40", "P50"]:    # 压比
        df[f"{c}_c"] = df[c] / df["P2"]; cols.append(f"{c}_c")
    return df, cols


# ============================================================================
# 2×2 消融的四种估计器 + 全零哨兵
# ============================================================================
def build_cum(T, n):
    """
    构造"累加矩阵" Cum, 形状 (T*n, T*n)。

    作用: 把【增量 d】映射成【累计退化 θ】。
          θ(t) = −Σ_{i≤t} d_i    ⟺    θ = −Cum · d
    结构: 分块下三角, 每个块是 n×n 单位阵。
          [ I  0  0 ]
          [ I  I  0 ]
          [ I  I  I ]
    """
    Cum = np.zeros((T * n, T * n))
    for t in range(T):
        for i in range(t + 1):
            Cum[t*n:(t+1)*n, i*n:(i+1)*n] = np.eye(n)   # np.eye(n): n×n 单位阵
    return Cum


def build_M(Hn, T):
    """
    构造设计矩阵 M, 形状 (13T, 9T)。
    预测残差:  r̂(t) = Hn·θ(t) = Hn·(−Σ_{i≤t} d_i) = −Σ_{i≤t} (Hn·d_i)
    所以 M[第t个13行块, 第i个9列块] = −Hn  当 i ≤ t, 否则为 0。
    """
    m, n = Hn.shape          # m=13(残差通道), n=9(健康参数)
    M = np.zeros((T * m, T * n))
    for t in range(T):
        for i in range(t + 1):
            M[t*m:(t+1)*m, i*n:(i+1)*n] = -Hn
    return M


# ---- 左上: A —— 无约束、无收缩 ----
def solve_A(Hn, R):
    """经典 GPA: θ̂ = H⁺r (Moore-Penrose 伪逆)。病态会在这里爆炸。"""
    return R @ np.linalg.pinv(Hn).T


# ---- 右上: B —— 无约束、有收缩 ----
def solve_B(Hn, R, lam):
    """Tikhonov 岭回归 (Doel 1994): θ̂ = (HᵀH+λI)⁻¹Hᵀr。逐 cycle 独立求解。"""
    A = Hn.T @ Hn + lam * np.eye(Hn.shape[1])
    B = np.linalg.solve(A, Hn.T)         # solve 比直接求逆数值上更稳
    return R @ B.T


# ---- 左下: C —— 有约束、无收缩 ----
def solve_C(Hn, R):
    """纯锥约束: min‖Hnθ−r‖² s.t. θ≤0, θ(t+1)≤θ(t)。等价于 NNLS。"""
    T, m = R.shape
    n = Hn.shape[1]
    M = build_M(Hn, T)
    d, _ = nnls(M, R.reshape(-1), maxiter=50 * T * n)
    return -np.cumsum(d.reshape(T, n), axis=0)


# ---- 右下: D —— ★本文方法: 有约束 + 有收缩 ----
def solve_D(Hn, R, lam):
    """
    min ‖Hnθ − r‖² + λ‖θ‖²   s.t.  θ ≤ 0 , θ(t+1) ≤ θ(t)

    【怎么把 Tikhonov 项塞进 NNLS】
      重参数化后 θ = −Cum·d, 所以 ‖θ‖² = ‖Cum·d‖² (负号不影响范数)。
      于是目标 = ‖M d − y‖² + λ‖Cum d‖²
              = ‖ [    M    ] d − [ y ] ‖²        ← 把 √λ·Cum 【增广】到 M 下面
                ‖ [ √λ·Cum ]       [ 0 ]  ‖         把 0 增广到 y 下面
      这是一个标准的非负最小二乘 → scipy.optimize.nnls 精确求解。
      (这是岭回归"数据增广"技巧的标准做法, 见任何统计教材)

    ⚠ λ 用【和方法B 完全相同的网格、相同的dev数据】各自调最优 —— 保证公平。
    """
    T, m = R.shape
    n = Hn.shape[1]

    M = build_M(Hn, T)                       # (13T, 9T)
    Cum = build_cum(T, n)                    # (9T, 9T)

    # np.vstack: 沿行方向堆叠 → 增广设计矩阵 (13T+9T, 9T)
    M_aug = np.vstack([M, np.sqrt(lam) * Cum])
    # np.concatenate: 拼接向量 → 增广观测 (13T+9T,)
    y_aug = np.concatenate([R.reshape(-1), np.zeros(T * n)])

    d, _ = nnls(M_aug, y_aug, maxiter=50 * T * n)
    return -np.cumsum(d.reshape(T, n), axis=0)


# ---- 哨兵 ----
def solve_Z(Hn, R):
    """全零估计器。任何方法只要跟它打平, 说明那个指标是坏的(奖励什么都不做)。"""
    return np.zeros((R.shape[0], Hn.shape[1]))


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 88)
    print("脚本16：2×2 消融 —— 隔离出物理约束的【净贡献】(D − B)")
    print("=" * 88)

    # ---------------- [1] 重建 H ----------------
    print("\n[1/3] 重建 H ...")
    data = {}
    for fn in IDENT_FILES:
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"  !! 缺 {fn}"); return
        data[fn] = load_for_ident(p)

    pool = pd.concat([d[d["hs"] == 1] for d in data.values()], ignore_index=True)
    ref = {"T_ref": pool["T2"].median(), "P_ref": pool["P2"].median()}
    for fn in data:
        data[fn], corrected_cols = add_corrected(data[fn], ref)
    pool, _ = add_corrected(pool, ref)
    resid_cols = [f"r_{c}" for c in corrected_cols]

    base = {}
    for c in corrected_cols:
        m = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=SEED)
        m.fit(pool[OP_COLS].values, pool[c].values)
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
    U, s, Vt = np.linalg.svd(Hn)
    v_null = Vt[-1] / np.linalg.norm(Vt[-1])
    print(f"  cond(H_n)={s[0]/s[-1]:.1f}, σ_min={s[-1]:.4f}σ")
    del data, pool

    # ---------------- 窗口构造 ----------------
    def build_windows(df, unit):
        """零点 = 该单元最初 N_REF_CYCLES 个 cycle (脚本15 已验证真值可行)"""
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

    # ---------------- [2~3] 逐验证子集 ----------------
    all_rows = []
    for tag, (fn, true_faults) in VALID_FILES.items():
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            print(f"  !! 缺 {fn}"); continue

        print("\n\n" + "#" * 88)
        print(f"# {tag}   真实退化部件: {true_faults}")
        print("#" * 88)

        ds = {}
        for split in ["dev", "test"]:
            d = load_per_cycle(p, split)
            d, _ = add_corrected(d, ref)
            ds[split] = pd.concat([d, residual(d) / resid_std], axis=1)

        # 零空间重合度 (预注册规律的自变量)
        th_final = ds["dev"][THETA9].min().values / span
        align = abs(float(v_null @ th_final)) / np.linalg.norm(th_final)
        print(f"\n  ★ 零空间重合度 = {align*100:.1f}%")

        idx_tf = [THETA9.index(q) for q in true_faults]
        other = [i for i in range(9) if i not in idx_tf]

        # ---- 调 λ: B 和 D 【各自】在 dev 上调, 用【同一个网格、同一批数据】 ----
        dev_units = sorted(ds["dev"]["unit"].unique())[:3]   # 3台够了, D 的NNLS较慢
        dev_wins = {u: build_windows(ds["dev"], u) for u in dev_units}

        print(f"\n  --- 调 λ (在 {tag}-dev 上, B 和 D 用完全相同的方式) ---")
        lamB, lamD = {}, {}
        for N in WINDOWS:
            errB, errD = [], []
            for lam in LAMBDA_GRID:
                eB, eD = [], []
                for u in dev_units:
                    R, TH = dev_wins[u][N]
                    eB.append(np.mean((solve_B(Hn, R, lam) - TH) ** 2))
                    eD.append(np.mean((solve_D(Hn, R, lam) - TH) ** 2))
                errB.append(np.mean(eB))
                errD.append(np.mean(eD))
            lamB[N] = LAMBDA_GRID[int(np.argmin(errB))]
            lamD[N] = LAMBDA_GRID[int(np.argmin(errD))]
            print(f"    N={N:>5d}   λ_B*={lamB[N]:.2e}   λ_D*={lamD[N]:.2e}")

        # ---- 在 test 上评测 ----
        test_units = sorted(ds["test"]["unit"].unique())
        print(f"\n  --- 在 {tag}-test 上评测 (单元 {[int(u) for u in test_units]}) ---")
        for u in test_units:
            wins = build_windows(ds["test"], u)
            for N in WINDOWS:
                R, TH = wins[N]
                T = len(R)
                n_early = max(1, int(0.3 * T))       # 早期 = 前 30% 的 cycle

                preds = {
                    "Z_全零":       solve_Z(Hn, R),
                    "A_无约束无收缩": solve_A(Hn, R),
                    "B_无约束+收缩": solve_B(Hn, R, lamB[N]),
                    "C_约束无收缩":  solve_C(Hn, R),
                    "D_约束+收缩★":  solve_D(Hn, R, lamD[N]),
                }
                mse_zero = float(np.mean(TH ** 2))   # 技能分的分母

                for name, TH_hat in preds.items():
                    err = TH_hat - TH

                    # 技能分 S = 1 − MSE/MSE(全零)。1=完美, 0=跟什么都不做一样, <0=更糟
                    mse = float(np.mean(err ** 2))
                    skill = 1.0 - mse / max(mse_zero, 1e-12)

                    # ★ 早期改用【绝对RMSE】。
                    #   脚本15 用技能分, 但早期 θ_true≈0 → 分母趋零 → 指标被除爆。
                    #   绝对 RMSE 没有这个问题。
                    rmse_early = float(np.sqrt(np.mean(err[:n_early] ** 2)))

                    # 虚警幅度: 6个【没退化】的部件上 |θ̂| 的均值。真值恒为0, 应≈0。
                    false_alarm = float(np.mean(np.abs(TH_hat[:, other])))

                    # 检出相关性: 3个【真退化】部件上, θ̂ 与真值的相关系数
                    corrs = []
                    for i in idx_tf:
                        if np.std(TH[:, i]) > 1e-9 and np.std(TH_hat[:, i]) > 1e-9:
                            corrs.append(np.corrcoef(TH_hat[:, i], TH[:, i])[0, 1])
                    detect = float(np.mean(corrs)) if corrs else 0.0

                    all_rows.append({
                        "subset": tag, "align_pct": align * 100, "unit": int(u),
                        "N": N, "method": name, "skill": skill,
                        "rmse_early": rmse_early, "false_alarm": false_alarm,
                        "detect_corr": detect, "rmse": np.sqrt(mse)})

    df_r = pd.DataFrame(all_rows)
    df_r.to_csv(os.path.join(OUT_DIR, "ablation_2x2.csv"),
                index=False, encoding="utf-8-sig")

    # ---------------- 报告 ----------------
    order = ["Z_全零", "A_无约束无收缩", "B_无约束+收缩", "C_约束无收缩", "D_约束+收缩★"]
    print("\n\n" + "=" * 88)
    print("2×2 消融结果")
    print("=" * 88)

    for tag in df_r["subset"].unique():
        sub = df_r[df_r["subset"] == tag]
        al = sub["align_pct"].iloc[0]
        print(f"\n\n{'#'*88}\n# {tag}   零空间重合度 = {al:.1f}%\n{'#'*88}")
        for metric, title in [
            ("skill",       "技能分 S  [1=完美, 0=跟全零一样, <0=更糟]  ★主指标"),
            ("rmse_early",  "早期 RMSE (前30%cycle, 绝对值)  ★★早期检测"),
            ("false_alarm", "虚警幅度 (6个未退化部件上 |θ̂| 均值, 应≈0)"),
            ("detect_corr", "检出相关性 (3个真退化部件, 越接近1越好)"),
        ]:
            pv = sub.pivot_table(index="N", columns="method", values=metric, aggfunc="mean")
            pv = pv[[c for c in order if c in pv.columns]]
            print(f"\n  --- {title} ---")
            print(pv.round(4).to_string())

    # ---------------- 裁决 ----------------
    print("\n\n" + "#" * 88)
    print("# 裁决")
    print("#" * 88)

    print("\n  【消融分解】各项的贡献 (技能分, 各N平均):\n")
    print(f"  {'子集':6s} {'重合度':>7s} {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s} "
          f"{'B−A(收缩)':>11s} {'C−A(约束)':>11s} {'★D−B(约束净贡献)':>17s}")
    print("  " + "-" * 96)
    gains = {}
    for tag in df_r["subset"].unique():
        sub = df_r[df_r["subset"] == tag]
        al = sub["align_pct"].iloc[0]
        g = {m: sub[sub["method"] == m]["skill"].mean() for m in order[1:]}
        dmb = g["D_约束+收缩★"] - g["B_无约束+收缩"]
        gains[tag] = (al, dmb)
        print(f"  {tag:6s} {al:>6.1f}% {g['A_无约束无收缩']:>8.2f} "
              f"{g['B_无约束+收缩']:>8.3f} {g['C_约束无收缩']:>8.3f} "
              f"{g['D_约束+收缩★']:>8.3f} "
              f"{g['B_无约束+收缩']-g['A_无约束无收缩']:>11.2f} "
              f"{g['C_约束无收缩']-g['A_无约束无收缩']:>11.2f} "
              f"{dmb:>+17.4f}")

    print("\n  --- 判据 Q4: D 的技能分 > B? 同时保住低虚警? ---")
    for tag in df_r["subset"].unique():
        sub = df_r[df_r["subset"] == tag]
        fa_B = sub[sub["method"] == "B_无约束+收缩"]["false_alarm"].mean()
        fa_D = sub[sub["method"] == "D_约束+收缩★"]["false_alarm"].mean()
        al, dmb = gains[tag]
        ok_skill = dmb > 0.02
        ok_fa = fa_D < fa_B
        print(f"    {tag}: 技能分增益 {dmb:+.4f} {'🟢' if ok_skill else '🔴'}  |  "
              f"虚警 B={fa_B:.4f} → D={fa_D:.4f} "
              f"({'🟢 降低' if ok_fa else '🔴 未降低'})")

    print("\n  --- 判据 Q5: D−B 的增益 随零空间重合度 增大? ---")
    if "DS02" in gains and "DS03" in gains:
        a2, g2 = gains["DS02"]
        a3, g3 = gains["DS03"]
        print(f"    DS02 (重合度 {a2:.1f}%): D−B = {g2:+.4f}")
        print(f"    DS03 (重合度 {a3:.1f}%): D−B = {g3:+.4f}")
        if g3 > g2:
            print("    🟢 Q5 成立: 重合度越高, 物理约束的净贡献越大。")
            print("       ⇒ 这条规律给出了【约束什么时候有用】的边界条件,")
            print("         比'物理约束有用'这种万能药式的宣称强得多。")
        else:
            print("    🔴 Q5 不成立: 增益与重合度无关。规律不成立。")

        print("\n  --- 🔴 判据 Q6 (最关键) ---")
        if max(g2, g3) < 0.02:
            print("    ❌❌❌ D 只是打平 B (两个子集的增益都 < 0.02)。")
            print("       结论: 在【收缩项完全相同】的条件下, 物理约束【没有独立价值】。")
            print("       '物理先验约束' 只是通用正则化的一个更麻烦的写法。")
            print("       ⇒ 核心卖点作废。这个负面结果是可信的, 应如实写进论文。")
            print("       ⇒ 主线改用 ⓪-d 的 H(w)=H₀·diag(g(w)) 归纳偏置")
            print("         (E1成立+E2触发, 有独立实验依据, 不依赖本实验)。")
            print("       ⇒ 但注意: 即便如此, C/D 的【虚警幅度】仍显著低于 B ——")
            print("         '约束降低虚警' 这个更小、更诚实的结论仍然可以成立,")
            print("         而且对工程实践(避免误拆发动机)是有价值的。")
        else:
            print(f"    ✅ D 赢过了 B (最大增益 {max(g2,g3):+.4f})。")
            print("       而且这是在【收缩项完全相同、λ 用同样方式调优】的条件下赢的,")
            print("       所以这个增益【只能】归因于物理约束本身。")
            print("       ⇒ 核心卖点成立。")

    # ---------------- 主图 ----------------
    try:
        subs = list(df_r["subset"].unique())
        fig, axes = plt.subplots(2, len(subs), figsize=(7.5*len(subs), 9.5), squeeze=False)
        style = {"Z_全零": ("gray", "x"), "A_无约束无收缩": ("#C44E52", "o"),
                 "B_无约束+收缩": ("#DD8452", "s"), "C_约束无收缩": ("#55A868", "v"),
                 "D_约束+收缩★": ("#4C72B0", "^")}
        for j, tag in enumerate(subs):
            sub = df_r[df_r["subset"] == tag]
            al = sub["align_pct"].iloc[0]
            for i, (metric, ylab, logy) in enumerate([
                ("skill", "技能分 S", False),
                ("false_alarm", "虚警幅度 (越小越好)", True)]):
                ax = axes[i][j]
                pv = sub.pivot_table(index="N", columns="method", values=metric,
                                     aggfunc="mean")
                for meth, (col, mk) in style.items():
                    if meth in pv.columns:
                        ax.plot(pv.index, pv[meth], marker=mk, color=col,
                                label=meth, lw=2)
                ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(-0.6, 1.05)
                    ax.axhline(0, color="k", ls=":", lw=1)
                ax.set_xlabel("聚合窗口 N")
                ax.set_ylabel(ylab)
                ax.set_title(f"{tag} (零空间重合度 {al:.1f}%) — {ylab}")
                ax.grid(alpha=0.3, which="both")
                ax.legend(fontsize=8)
        fig.suptitle("2×2 消融：物理约束的净贡献 = D − B（收缩项相同）", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "ablation_2x2.png"), dpi=150)
        plt.close(fig)
        print("\n  ⭐ 主图: ablation_2x2.png")
    except Exception as e:
        print("  (画图失败)", e)

    print(f"\n{'='*88}\n✅ 完成: {os.path.abspath(OUT_DIR)}\n{'='*88}")


if __name__ == "__main__":
    main()