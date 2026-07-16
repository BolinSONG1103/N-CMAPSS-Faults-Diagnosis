# N-CMAPSS 硕士论文第3章项目

本项目是 MATLAB 主实现。健康基准回归与锁定 NumPy 随机流通过隔离的 Python 兼容层完成，所有直接 Python API 仅位于 `matlab/src/health_baseline_py.m`；其余数据读取、影响矩阵辨识、约束反演、网络、指标、裁决和出图均由 MATLAB 完成。

## 当前状态

项目正在由“连续健康参数反演”升级为“连续估计—故障检测—多标签隔离—有序退化等级—未知故障拒识”的诊断闭环。新协议已经修复旧版数据驱动对照中的 unit 留出泄漏：健康基准、残差尺度和影响矩阵 H 只使用 train 发动机，lambda 与全部决策阈值只使用 calibration 发动机，test_id、DS02 与 DS03 不参与模型选择。

闭环研究定义与结论边界见 `docs/第三章诊断闭环研究协议.md`；论文重构稿见 `chapter3_results/第3章诊断闭环重构稿.md`；实现入口见 `matlab/README.md`。正式闭环、鲁棒性、图组验证和五次 unit 重复划分已经完成；结果边界与失败项见 `chapter3_results/诊断闭环结果审查.md`。

旧版 T0-T5、阶段 A 与阶段 B/C 产物暂时保留用于追溯连续反演方法的形成过程，但不属于诊断闭环 v2 的正式验证结果。旧版“E1 分布内公平性通过”不再属于正式结论，因为旧管线在声明留出单元前已经使用全部 dev 单元构造物理方法。

锁定 HistGB 管线得到 cond(H_n)=1562.791820。MATLAB LSBoost 对照的条件数为 2048.976413，漂移 31.11%，因此 HistGB 兼容层是复现正式数字的永久依赖，而非临时过渡代码。

## 目录

- `matlab/src/`：共享管线、实验实现、最终出图和验证器。
- `matlab/run/`：可直接运行的 MATLAB 入口。
- `matlab/cache/`：影响矩阵管线和健康基准模型缓存。
- `matlab/outputs/`：T0-T3、阶段 A 和阶段 B/C 的 raw CSV 与 verdict JSON。
- `matlab/outputs/t4_diagnostic_closure/`：主划分的正式诊断闭环证据。
- `matlab/outputs/t5_robustness_ablation/`：约束消融与传感器扰动证据。
- `matlab/outputs/t6_repeated_splits/`：五次划分的汇总证据；各 seed 中间目录不进入版本库。
- `chapter3_results/figures/`：程序生成的图3-2至图3-8，每图含 600 dpi PNG 和矢量 PDF。
- `chapter3_results/tables/`：表1至表11。
- `chapter3_results/第3章完整稿.md`：完整章节正文。
- `reference/`：已定案的 Python 参考脚本及固定结果，仅用于追溯。
- `docs/`：原始任务书与项目学习手册。

## 最终构建与验证

```matlab
cd('D:/PythonProjects/N-CMAPSS-fault-diagnosis/matlab/run')
run_21_diagnostic_closure
run_22_robustness_ablation
run_build_diagnostic_figures
run_verify_diagnostic_closure
run_23_repeated_splits
run_24_family_diagnosis
```

前四个入口形成并验证单次正式闭环；最后一个入口在五个 unit 划分下重新拟合健康基准、H、lambda 和决策层，用于评估结论对机队划分的稳定性。

`run_23_repeated_splits` 默认在所有跨划分表写入成功后删除四个可再生 seed 中间目录，只保留 `t6_repeated_splits` 汇总；如需调试，可直接调用 `experiment_23_repeated_splits(true)` 保留中间目录。

`run_24_family_diagnosis` 是依据 `docs/部件族诊断补充协议.md` 冻结的独立补充实验：保留九维连续估计，将离散隔离目标映射为 HPT、Fan、HPC、LPT、LPC 五个物理部件族。它不覆盖 v2 结果，也不改变阶段和未知故障协议。

需要复核既有连续反演阶段时，仍可使用 `run_t0_acceptance`、`run_19b`、`run_20`、`run_tau_check` 和阶段 A 入口，但其输出不得覆盖闭环 v2 的正式结果。
