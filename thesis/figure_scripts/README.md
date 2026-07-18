# 第3章正式出图脚本

本目录提供 MATLAB 与 R 两套正式排版入口；`figure_pipeline/` 的 Python 版本用于在无 MATLAB/R
环境中一键重建、核对信息量和做回归测试。

## MATLAB：连续估计侧

运行 `MATLAB/build_continuous_figures.m`。该入口不依赖原始数据或 MATLAB cache，直接从锁定 CSV
生成图3-2至图3-9及图3-19，输出到 `thesis/figures/matlab_official/`。当前脚本已在 MATLAB R2025a
实跑通过；生成目录被 `.gitignore` 排除，避免与仓库内 Python 预览图重复提交。

## R/ggplot2：DTAE 诊断侧

在仓库根目录运行：

```r
source("thesis/figure_scripts/R/build_dtae_figures.R")
build_dtae_figures()
```

依赖：`ggplot2`、`readr`、`dplyr`、`tidyr`、`scales`、`patchwork`。输出到
`thesis/figures/r_official/`，包含图3-11至图3-18的 600 dpi PNG 与矢量 PDF。

## 诚实性说明

- 两套脚本只读取锁定 CSV，不重新拟合、不重新选参。
- 多标签混淆矩阵是逐类召回/预测共现率，行和不要求等于 1。
- 图3-19使用旧物理约束统一决策层的锁定鲁棒性结果，不标成 DTAE 鲁棒性。
