# 第3章 MATLAB 收尾阶段 A 交接

> 状态：阶段 A（A1–A3）已完成；阶段 B/C 尚未执行。根据收尾任务书的阶段门控，当前停下等待作者确认。

## A1　Python 依赖隔离与纯 MATLAB 健康基准对照

### 结论

对锁定的预注册正式结果，Python 兼容层应视为**永久依赖**，不能自动切换为 MATLAB LSBoost。原因不是最终 DS03 技能分，而是影响矩阵的数值指纹发生了明显漂移：纯 MATLAB LSBoost 的 `cond(H_n)` 相对锁定 HistGB 增加 31.11%，远超 1% 容差。

主动 MATLAB 代码中的所有直接 Python API 已收敛到唯一入口：

- `matlab/src/health_baseline_py.m`

该入口同时隔离两类历史数值兼容需求：

1. 13 个 scikit-learn `HistGradientBoostingRegressor` 健康基准的拟合、保存、载入与预测；
2. NumPy PCG64 抽样及状态恢复，用于逐位复现锁定结果。

`ncmapss_lib.m` 及其他实验文件只调用该入口，不再直接出现 `py.importlib`、`pyargs` 或 `py.str`。

### 支撑数字

| 后端 | cond(H_n) | DS03 技能分 | 健康模型数 |
|---|---:|---:|---:|
| 锁定 sklearn HistGB | 1562.791820 | 0.957563972 | 13 |
| MATLAB LSBoost 对照 | 2048.976413 | 0.957716428 | 13 |
| Z_zero | — | 0 | 0 |

- `cond(H_n)` 相对漂移：`31.1100%`，不满足 `<1%` 的切换条件。
- DS03 技能分绝对漂移：`0.000152456`，满足 `<0.01`。
- 两条条件没有同时满足，因此 `automatic_switch_allowed=false`。

来源：

- `matlab/outputs/stage_a/a1_baseline/baseline_comparison_summary.csv`
- `matlab/outputs/stage_a/a1_baseline/baseline_drift.csv`
- `matlab/outputs/stage_a/a1_baseline/verdict_a1.json`

## A2　T0 第二条验收的原始数字

### 结论

原 T0 第二条验收继续通过。历史 `raw_scan.csv` 使用网格点 `lambda=31.6227766017`，MATLAB 正式配置记录为 `lambda=31.6`；二者对应任务书中的同一主 lambda。

| 数字 | 数值 |
|---|---:|
| raw_scan 原始聚合值 X | 0.957064895267 |
| MATLAB 重算值 Y | 0.957563972105 |
| 绝对差 | 0.000499076838 |
| 容差 | 0.005 |

`|X-Y|=0.000499076838 <= 0.005`，所以保持通过，不需要排查移植偏差。

来源：

- `reference/fixed_results/unsupervised/raw_scan.csv`
- `matlab/outputs/t0/t0_ds03_raw.csv`
- `matlab/outputs/stage_a/a2_t0/t0_raw_comparison.csv`
- `matlab/outputs/stage_a/a2_t0/verdict_a2.json`

## A3-1　Q2s 小 lambda 诊断

### 正式裁决不变

正式 Q2s 仍为未通过：占优率 `62.5%`，中位 gap `0.003148645`。本节只提供事后机制证据，不替换正式裁决。

### 诊断结果

在 `q=0.75`、相同 `(subset,N)` 配对点上：

| 条件 | MoE 占优率 | MoE-H0 中位技能分 gap |
|---|---:|---:|
| 小收缩，lambda=0.001 | 100% | 0.349427709 |
| 主收缩，lambda=31.6228 | 37.5% | -0.000161689 |

- 小 lambda 相对主 lambda 的配对中位提升：`0.349718710`。
- 5000 次 bootstrap 95% 区间：`[0.318577582, 0.370229581]`，完全大于 0。
- q=0.75 的门控增益偏离 1 的中位绝对幅度为 `0.100585`，因此不能解释为“分段后平均门控仍几乎等于 1”。
- 收紧到 q=0.85/0.90/0.95 后，正式 lambda 网格上的占优率分别为 72.2%/72.2%/70.8%，仍未达到 95%；q=0.95 的中位 gap 为 0.00988，也仍低于 0.01。各分位阈值回退率均为 0。

### 解释

数据支持原来的“收缩兜住 H 幅值失配”解释：关掉 Tikhonov 收缩后，MoE 的幅值修正确实产生大幅正 gap；恢复主 lambda 后，H0 与 MoE 的差异被约束反演强烈压平。因此 Q1 的正问题改善与 Q2s 的反问题增益有限可以同时成立。

来源：

- `matlab/outputs/stage_a/a3_q2/q2_lambda_raw.csv`
- `matlab/outputs/stage_a/a3_q2/q2_small_vs_main_paired.csv`
- `matlab/outputs/stage_a/a3_q2/q2_quantile_summary.csv`
- `matlab/outputs/stage_a/a3_q2/verdict_a3_q2.json`

## A3-2　E3 退化严重度与替代指标

### 原解释被数据否定

原交接中“外推子集退化幅度更大”的解释不成立。按量程归一化真值、活动故障分量计算，OOD 的严重度在五种口径下均低于分布内留出组：

| 严重度口径 | 分布内中位数 | OOD 中位数 | OOD/分布内 | OOD-分布内 95% CI |
|---|---:|---:|---:|---:|
| 末期中位数 | 0.464750 | 0.282284 | 0.607 | [-0.272230, -0.013541] |
| 末期 Q75 | 0.639264 | 0.410085 | 0.641 | [-0.331221, -0.172871] |
| 末期 Q90 | 0.778861 | 0.549708 | 0.706 | [-0.399284, -0.196665] |
| 末期最大值 | 0.854378 | 0.633326 | 0.741 | [-0.440471, -0.192116] |
| 轨迹 RMS | 0.213001 | 0.135694 | 0.637 | [-0.100971, -0.038551] |

因此：

- E3 正式裁决仍为未通过；
- 不能再用“外推退化幅度更大”解释负泛化落差；
- 跨不同故障组成、不同数据子集的技能分落差仍不是因果指标，但当前数据只证明了**严重度解释错误**，不能把失败简单归因于严重度混杂。

### 同子集内替代指标（事后诊断，尚未作为 B1 正式裁决）

DS02/DS03 上逐 `(unit,N)` 比较 D 与数据驱动臂：

| 子集 | 对手 | D 占优率 | D-对手中位技能分差 |
|---|---|---:|---:|
| DS02 | E_Lin | 100% | 0.071907 |
| DS02 | E_MLP | 100% | 0.738357 |
| DS02 | E_MixLinear | 75.0% | 0.027378 |
| DS02 | E_WPMixer | 100% | 4.989407 |
| DS03 | E_Lin | 83.3% | 0.052537 |
| DS03 | E_MLP | 100% | 0.598390 |
| DS03 | E_MixLinear | 87.5% | 0.140340 |
| DS03 | E_WPMixer | 100% | 5.388374 |

合并 OOD 配对后，E_Lin 与 E_MixLinear 的 D 占优率分别为 88.9% 和 83.3%，而 E_MLP 与 E_WPMixer 均为 100%。这支持一个更细的边界结论：深层非线性臂在未见故障组合上明显失稳；线性臂具有较强可外推性，D 的优势并非在每个线性配对点都成立。阶段 B 若采用 95% 的 E3' 阈值，必须据此如实裁决，不能预设“全部通过”。

来源：

- `matlab/outputs/stage_a/a3_e3/severity_metric_comparison.csv`
- `matlab/outputs/stage_a/a3_e3/same_subset_paired_raw.csv`
- `matlab/outputs/stage_a/a3_e3/same_subset_paired_summary.csv`
- `matlab/outputs/stage_a/a3_e3/verdict_a3_e3.json`

## 阶段 A 新增图形

阶段 A 共新增 12 组图，每组同时输出：

- 600 dpi PNG；
- 矢量 PDF；
- SVG。

统一采用色盲友好配色、无上/右边框坐标轴、紧凑多面板布局、面板字母、清晰单位与直接数值标注。图形位于：

- `matlab/outputs/stage_a/a1_baseline/figures/`（2 组）
- `matlab/outputs/stage_a/a2_t0/figures/`（1 组）
- `matlab/outputs/stage_a/a3_q2/figures/`（3 组）
- `matlab/outputs/stage_a/a3_e3/figures/`（6 组）

## 等待作者确认

请作者确认以下两点后再进入阶段 B：

1. 接受 HistGB Python 兼容层作为锁定正式结果的永久依赖，并在论文方法节明示；
2. 接受 E3 的原“外推退化更大”解释被否定，后续改用同子集内配对指标，并如实保留线性臂未达到 95% 的结果。

在确认前，B1/B2 与 C1/C2 均未执行。
