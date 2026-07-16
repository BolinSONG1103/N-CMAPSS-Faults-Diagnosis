# 第3章项目最终交接

> 阶段 A、B、C 与正文图件最终重制已于 2026-07-16 完成。项目以 MATLAB 为主实现；为复现锁定正式结果，HistGradientBoostingRegressor 与精确 NumPy 随机流通过唯一入口 `matlab/src/health_baseline_py.m` 调用，该兼容层为永久依赖。

## 完成状态

T0-T5、阶段 A 机制诊断以及阶段 B/C 最终裁决均已执行。正式产物位于 `matlab/outputs/` 和 `chapter3_results/`。

| 判据 | 正式结果 | 裁决 |
|---|---:|---|
| T0 cond(H_n) | 1562.791820，相对指纹误差 0.0005% | 通过 |
| T0 DS03 技能分 | 0.957563972，历史值差 0.000499 | 通过 |
| P1 D 逐点占优 | 165/168=98.2% | 通过 |
| P2 经典 λ 准则 | D 路径上共同退化 | 未通过 |
| P4 B 路径代理 | DS03 通过，DS02 loss=0.038856 | 部分通过 |
| T3 τ 与 √N | R²=0.9670 | 通过 |
| Q1 前向有效性 | NRMSE 降低 37.90% | 通过 |
| Q2f 整周期机制 | 中位 gap=-0.000655 | 描述性一致 |
| Q2s highTRA | 62.5%，中位 gap=0.003149 | 未通过 |
| Q3 冻结 H 方向 | 扰动比 0.003696，最差余弦 0.999913 | 通过 |
| Q4 MoE 对自由 MLP | 满足 0.005 容差 | 通过 |
| Q6 MoE 对 TRA4 | +0.00110/-0.00071 | 描述性边界 |
| E1 分布内公平性 | 最佳数据驱动 0.7505，D=0.7349 | 通过 |
| E2 组合外推 | D 在 DS02/DS03 逐臂全胜 | 通过 |
| E3′-a 深层臂 | 两臂占优率均为 100% | 通过 |
| E3′-b 线性臂 | 占优率 83.3%-88.9% | 描述性边界 |
| 原 E3 | 跨子集落差不再用于结论 | 已归档 |

## 最终科学叙事

- 全局 H0、量程归一化、Tikhonov 收缩与非正-单调硬约束构成主方法。D 在 98.2% 的完整 λ 路径点上优于 B，并在所有点上降低虚警。
- MoE 的前向 NRMSE 从 0.32780 降至 0.20355；λ=0.001 时诊断占优率为 100%，但 λ≈31.6 时中位增益接近 0，说明主工作点的约束与收缩吸收了大部分工况幅值失配。
- 组合外推中，线性臂仍保留 0.82-0.84 的技能分，但虚警约为 D 的 3.2-3.3 倍；深层臂进一步恶化。结论是线性可加性提供外推，硬约束提供特异性。
- 严重度审计的五种口径均显示 OOD 低于分布内组，不能再使用“OOD 退化更重”解释原 E3。

## 图表与文稿

- 正文 8 图：图3-1由作者手画；`chapter3_results/figures/` 保存图3-2至图3-8的 600 dpi PNG 和矢量 PDF。
- 11 张正式表：`chapter3_results/tables/`，均保留 Z_zero 或 Z_zero_skill=0。
- 图清单：`chapter3_results/figure_manifest_v2.csv`。
- 图件重制记录：`chapter3_results/figures_changelog.md`。
- 完整章节：`chapter3_results/第3章完整稿.md`。
- 结果初稿：`chapter3_results/第3章实验结果初稿.md`。
- 数值汇总：`chapter3_results/数值汇总.md`。
- 一致性核查：`chapter3_results/一致性核查.md`。

## 复现入口

```matlab
cd('C:/Users/25691/Desktop/thesis_part3/matlab/run')
run_build_results
run_verify_stage_bc
```

`run_build_results` 仅从锁定正式 CSV 和阶段结果生成 B/C 裁决、表格与图，不重跑 P1-P4、Q1 或 Q3。重新运行 T1/T2 会覆盖对应正式输出，不属于最终出图所需步骤。

## 目录纪律

- `matlab/src/`：MATLAB 实现与唯一 Python 兼容入口。
- `matlab/run/`：运行入口。
- `matlab/cache/`：正式缓存和 13 个 HistGB 模型。
- `matlab/outputs/`：原始 CSV、裁决 JSON 与阶段产物。
- `chapter3_results/`：最终图、表、文稿与核查报告。
- `reference/`：锁定 Python 参考脚本和固定结果，仅用于追溯。

FAST_MODE 数字、旧图 A-E、旧预注册判据表、IDE 配置和重复结果不属于最终交付。
