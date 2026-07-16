# N-CMAPSS 第3章 MATLAB 主实现

本目录包含第3章的 MATLAB 主管线。健康基准 HistGradientBoostingRegressor 与锁定 NumPy PCG64 随机流通过唯一入口 `src/health_baseline_py.m` 调用；其余数据处理、H 辨识、约束反演、MoE、数据驱动对照、裁决、表格和图形均为 MATLAB 实现。

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
- `outputs/`：原始结果、阶段诊断和裁决。
- `../chapter3_results/`：正文 8 图清单、7 组程序图、11 张表和章节文稿。
