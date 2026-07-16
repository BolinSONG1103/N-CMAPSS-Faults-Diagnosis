# N-CMAPSS 第3章 MATLAB 主实现

本目录包含第3章的 MATLAB 主管线。健康基准 HistGradientBoostingRegressor 与锁定 NumPy PCG64 随机流通过唯一入口 `src/health_baseline_py.m` 调用；其余数据处理、H 辨识、约束反演、MoE、数据驱动对照、裁决、表格和图形均为 MATLAB 实现。

## 诊断闭环 v2

旧版 T0-T5 结果用于追溯连续健康参数反演的开发过程，不再作为最终“故障检测—隔离—阶段—拒识”闭环的正式证据。原因是旧版 `experiment_20` 在声明 DS05/DS07 留出单元之前，物理管线已经使用全部 dev 单元构造健康基准和影响矩阵。v2 已改为每个辨识子集内部的 unit 级 `train/calibration/test_id` 三划分：

- `train`：健康基准、残差尺度、H 和数据驱动对照训练；
- `calibration`：lambda、诊断阈值、持续周期数、退化等级和拒识阈值；
- `test_id`：分布内正式测试；
- DS02/DS03：未见故障组合测试。

研究定义、依据、结论边界和预注册指标见 `../docs/第三章诊断闭环研究协议.md`。

正式运行顺序：

```matlab
cd('D:/PythonProjects/N-CMAPSS-fault-diagnosis/matlab/run')
run_21_diagnostic_closure
run_22_robustness_ablation
run_build_diagnostic_figures
run_verify_diagnostic_closure
run_23_repeated_splits
run_24_family_diagnosis
```

第一步生成并锁定 `outputs/protocol/unit_split_manifest.csv`，随后完成 calibration 选参、ID/OOD 诊断和 leave-one-fault-family-out 拒识；第二步完成约束消融、标准化残差噪声、固定偏置和逐通道缺失实验；第三步只读取落盘 CSV 生成五组闭环结果图；第四步验证 unit 隔离、选参来源、五个拒识折和图件分辨率。

`run_23_repeated_splits` 会对五个 unit 划分重新拟合健康基准和 H，计算量最大，应在单次正式闭环与图表验证通过后运行。它不能用只重复决策阈值的方式替代，因为划分不确定性必须传播到健康基准、H、lambda 和诊断输出。程序默认将 lambda 路径、持续性选择、阶段敏感性和未知故障折汇总到 `outputs/t6_repeated_splits/`，全部写入成功后删除四个可再生 seed 中间目录；调试时可调用 `experiment_23_repeated_splits(true)` 保留中间目录。

`run_24_family_diagnosis` 读取主划分已经锁定的连续反演模型，按运行前冻结的 HPT/Fan/HPC/LPT/LPC 映射建立五部件族多标签诊断；协议见 `../docs/部件族诊断补充协议.md`。该实验只修正离散诊断目标层级，不覆盖九维参数级结果。

建议先运行单元测试：

```matlab
results = runtests('../tests');
assertSuccess(results)
```

## 正式兼容层

任务书将 HistGB 数值输出与 cond(H_n)=1562.8 绑定。正式管线得到 1562.791820；MATLAB LSBoost 对照为 2048.976413，相对漂移 31.11%。虽然 DS03 技能分仅漂移 0.000152，两个后端仍不能视为数值等价，因此 HistGB 兼容层是锁定结果的永久依赖。

## 最终入口

```matlab
cd('C:/Users/25691/Desktop/thesis_part3/matlab/run')
run_build_results
run_verify_stage_bc
```

- `run_build_results`：从锁定 CSV 和现有阶段结果生成 E3′、E4、11 张表与图3-2至图3-8；图3-1为作者手画占位。
- `run_verify_stage_bc`：检查正式裁决、7 组 PNG/PDF、600 dpi、Z_zero、禁用词、U+2212、signed-log 禁令和 FAST_MODE。

## 其他入口

```matlab
run_t0_acceptance
run_19b
run_20
run_tau_check
run_stage_a_baseline
run_stage_a_t0_raw
run_posthoc_q2
run_stage_a_e3
run_verify_stage_a
```

这些入口用于重建早期实验或阶段 A 机制诊断。最终 B/C 构建不会重跑 P1-P4、Q1 或 Q3，也不会移动阈值、删除失败项或使用 FAST_MODE 数字。

## 目录

- `src/`：实现、图形规范与验证器。
- `run/`：运行入口。
- `cache/`：复用模型和正式管线缓存。
- `outputs/`：原始结果、阶段诊断和裁决；烟测目录与重复划分中间目录不进入版本库。
- `../chapter3_results/`：正文 8 图清单、7 组程序图、11 张表和章节文稿。
