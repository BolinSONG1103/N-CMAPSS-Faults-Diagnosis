# 第3章完整图集中间数据审计

审计时间：2026-07-16。审计原则是先确认已有数据，再仅为出图补算指定单元的 theta_hat 或中间预测，不重跑 P/Q/E 实验。

| 编号 | 数据项 | 审计状态 | 已有/补存路径 | 处理说明 |
|---|---|---|---|---|
| 1 | 相似修正前后的通道-工况数据 | 缺失→需补 | 计划：`matlab/outputs/complete_figures/data/a1_similarity_correction.csv` | 使用 DS03 unit 13 前 3 个 cycle，保存 T30/P40 修正前后值及环境比值 |
| 2 | 健康基准预测值与实测值 | 缺失→需补 | 计划：`matlab/outputs/complete_figures/data/a2_baseline_fit.csv` | 使用锁定 HistGB 管线对 DS03 unit 13 前 3 个 cycle 预测 13 个修正通道 |
| 3 | H_n、奇异值与列夹角 | 已存在 | `matlab/cache/pipeline_cache_exact.mat` | H_n 和 cond 已锁定；奇异值、夹角可直接从 H_n 计算 |
| 4 | 代表机 D/B/真值反演轨迹 | 已存在 | `matlab/outputs/stage_bc/figure_data/fig3_2_ds03_unit13_trajectory.csv` | DS03 unit 13、N=1000、lambda=31.6，含九个部件 |
| 5 | 三个 lambda 下的 theta_hat 轨迹 | 缺失→需补 | 计划：`matlab/outputs/complete_figures/data/b3_lambda_trajectory.csv` | 同一 unit 13，补算 lambda=0.001、31.6、10000 |
| 6 | 多台机、逐部件估计误差 | 缺失→需补 | 计划：`matlab/outputs/complete_figures/data/b4_multiunit_component_error.csv` | 补算 DS02/DS03 全部 test unit 的 D 轨迹 RMSE 与末期误差，并保留 Z_zero |
| 7 | 外推 D/E_Lin/E_MLP 轨迹 | 缺失→需补 | 计划：`matlab/outputs/complete_figures/data/c3_c4_extrapolation_trajectory.csv` | 加载锁定 `t2_dd/models.mat`，仅对 DS03 unit 13 复算预测轨迹 |
| 8 | 工况自适应 gap-lambda | 已存在 | `matlab/outputs/t1_moe/pointwise_gap.csv`；`matlab/outputs/stage_a/a3_q2/q2_lambda_raw.csv` | 可直接绘制 full/highTRA 路径 |
| 9 | tau_required-sqrt(N) | 已存在 | `matlab/outputs/t3_tau/tau_required_points.csv`；`verdict_tau.json` | 可直接绘制线性拟合与 R² |

补数据脚本：`matlab/src/prepare_complete_figure_data.m`。完成补存后，本文件将把第 1、2、5、6、7 项更新为“缺失→已补”。
