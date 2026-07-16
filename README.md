# N-CMAPSS 硕士论文第3章项目

本项目是 MATLAB 主实现。健康基准回归与锁定 NumPy 随机流通过隔离的 Python 兼容层完成，所有直接 Python API 仅位于 `matlab/src/health_baseline_py.m`；其余数据读取、影响矩阵辨识、约束反演、网络、指标、裁决和出图均由 MATLAB 完成。

## 当前状态

T0-T5、阶段 A 诊断和阶段 B/C 收尾均已完成。图件最终版将正文精简为 8 图：图3-1由作者手画，MATLAB 生成图3-2至图3-8共 7 组；同时保留 11 张正式表、第3章完整稿、数值汇总、裁决 JSON 和自动一致性核查。

锁定 HistGB 管线得到 cond(H_n)=1562.791820。MATLAB LSBoost 对照的条件数为 2048.976413，漂移 31.11%，因此 HistGB 兼容层是复现正式数字的永久依赖，而非临时过渡代码。

## 目录

- `matlab/src/`：共享管线、实验实现、最终出图和验证器。
- `matlab/run/`：可直接运行的 MATLAB 入口。
- `matlab/cache/`：影响矩阵管线和健康基准模型缓存。
- `matlab/outputs/`：T0-T3、阶段 A 和阶段 B/C 的 raw CSV 与 verdict JSON。
- `chapter3_results/figures/`：程序生成的图3-2至图3-8，每图含 600 dpi PNG 和矢量 PDF。
- `chapter3_results/tables/`：表1至表11。
- `chapter3_results/第3章完整稿.md`：完整章节正文。
- `reference/`：已定案的 Python 参考脚本及固定结果，仅用于追溯。
- `docs/`：原始任务书与项目学习手册。

## 最终构建与验证

```matlab
cd('C:/Users/25691/Desktop/thesis_part3/matlab/run')
run_build_results
run_verify_stage_bc
```

`run_build_results` 从锁定结果完成 E3′、E4、最终表格和 7 组正文程序图，不重跑 P1-P4、Q1 或 Q3；仅在图3-2轨迹 CSV 不存在时，按任务书允许范围复算一台 DS03 发动机。`run_verify_stage_bc` 检查裁决一致性、图件格式、PNG 分辨率、Z_zero、禁用表述和 FAST_MODE 状态。

需要从头复核既有阶段时，可使用 `run_t0_acceptance`、`run_19b`、`run_20`、`run_tau_check` 以及阶段 A 的四个诊断入口；这些步骤不是最终出图的必要前置条件。
