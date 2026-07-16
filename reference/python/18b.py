# 18b_cross_lambda.py —— 只读 unsup_out 里的两个 csv，几秒钟跑完
import numpy as np, pandas as pd

raw = pd.read_csv("./unsup_out/raw_scan.csv")
sel = pd.read_csv("./unsup_out/selected_lambda.csv")

# lam 是浮点数，直接 merge 有精度风险 → 建一个稳定的整数键
for d in (raw, sel):
    d["lamkey"] = np.round(np.log10(d["lam"]), 6)

KEY = ["subset", "unit", "N"]

# ===== A. 逐点虚警/检出占优（把 P3 加强成和 P1 同级的证据）=====
print("\n" + "="*80)
print("A. 逐点占优（虚警越小越好 / 检出越大越好）")
print("="*80)
for metric, better in [("false_alarm", "less"), ("detect_corr", "greater")]:
    g = (raw[raw.method != "Z_zero"]
         .pivot_table(index=["subset", "N", "lam"], columns="method",
                      values=metric, aggfunc="mean").dropna())
    win = (g["D"] < g["B"]) if better == "less" else (g["D"] > g["B"])
    print(f"\n  --- {metric} ---   总占优率 = {win.mean():.1%} ({win.sum()}/{len(win)})")
    for (tag, N), gg in g.groupby(level=[0, 1]):
        w = (gg["D"] < gg["B"]) if better == "less" else (gg["D"] > gg["B"])
        print(f"    {tag}  N={N:>5d}   占优 {w.sum():>2d}/{len(gg):>2d} = {w.mean():>6.1%}")

# ===== B. P4(a): oracle 的 lambda 在 B 和 D 上是否一致？=====
print("\n" + "="*80)
print("B. P4(a)  lambda_oracle(B) 与 lambda_oracle(D) 是否重合？（交叉选择的前提）")
print("="*80)
orc = (sel[sel.criterion == "oracle"]
       .pivot_table(index=["subset", "N"], columns="method",
                    values="lamkey", aggfunc="median"))
orc["diff_dec"] = (orc["B"] - orc["D"]).abs()     # log10 尺度上差几个数量级
print(orc.round(3).to_string())
P4a = bool((orc["diff_dec"] <= 0.5).all())
print(f"\n  P4(a)  所有格点 |Δlog10 λ| ≤ 0.5 ?  ->  {'[OK] 成立' if P4a else '[X] 不成立'}")

# ===== C. P4(b)(c): 用 B 的路径选 λ，用 D 解 =====
print("\n" + "="*80)
print("C. 交叉选择：lambda 由【无约束的 B】选出，解由【有约束的 D】给出")
print("="*80)
rawD = raw[raw.method == "D"]
d_orc = (sel[(sel.method == "D") & (sel.criterion == "oracle")]
         .groupby("subset")["skill"].mean())      # D 的逐N-oracle 上界

rows = []
for crit in ["L_curve", "Morozov_tau1.0", "Morozov_tau3.0", "oracle"]:
    # B 在该准则下选的 λ（B 自己的路径，不看真值 —— oracle 除外，仅作参考）
    pick = sel[(sel.method == "B") & (sel.criterion == crit)][KEY + ["lamkey", "skill"]]
    pick = pick.rename(columns={"skill": "skill_B"})
    # 把这个 λ 塞给 D
    m = pick.merge(rawD[KEY + ["lamkey", "skill", "false_alarm", "detect_corr"]],
                   on=KEY + ["lamkey"], how="left")
    assert m["skill"].notna().all(), f"{crit}: merge 有漏，检查 lamkey"
    for tag, gg in m.groupby("subset"):
        rows.append(dict(subset=tag, crit_on_B=crit,
                         D_at_lamB=gg["skill"].mean(),
                         B_itself=gg["skill_B"].mean(),
                         D_oracle=d_orc[tag],
                         loss=d_orc[tag] - gg["skill"].mean(),
                         D_fa=gg["false_alarm"].mean()))
res = pd.DataFrame(rows)
print(res.round(4).to_string(index=False))

lc = res[res.crit_on_B == "L_curve"]
P4b = bool((lc["loss"] < 0.03).all())
P4c = bool((lc["D_at_lamB"] > lc["B_itself"]).all())
print(f"\n  P4(b)  D(λ_Lcurve(B)) 距 D-oracle 损失 < 0.03 ?  -> {'[OK]' if P4b else '[X]'}")
print(f"  P4(c)  D(λ_Lcurve(B)) > B(L_curve) ?             -> {'[OK]' if P4c else '[X]'}")
print(f"\n  ==> 交叉选择方案 {'【可用】，P2 被救活' if (P4a and P4b and P4c) else '【不可用】，见下面的备选'}")