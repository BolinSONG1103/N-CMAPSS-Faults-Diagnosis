# N-CMAPSS 硕士论文第3章：物理结构约束的气路故障诊断与退化程度评估

本项目围绕 NASA N-CMAPSS 真实飞行工况涡扇发动机退化数据，构建**从异常出现到部件级健康状态的完整诊断闭环**：

> 故障发现（相似修正 + 健康基准残差 + 故障检测）→ 识别分类（五部件族隔离）
> → 趋势与程度（连续退化轨迹 + 有序退化等级）→ 未知拒识（留一故障族开放集）。

方法以数据辨识的传感器—部件影响矩阵表示故障物理指纹，通过非正—单调重参数化把不可逆退化先验写入解空间，以 Tikhonov 收缩处理 $\text{cond}(H_n)\approx1563$ 的病态反问题，并在连续估计之上建立统一决策层。计算主体为 MATLAB；健康基准回归与锁定随机流经隔离的 Python 兼容层（仅 `matlab/src/health_baseline_py.m`）完成。

## 目录结构

| 目录 | 内容 |
|---|---|
| `thesis/` | **论文交付物**：第3章正稿、正文图（图3-2–图3-10）、表格、手绘图规格、结论边界。 |
| `figure_pipeline/` | **正文出图程序**（Python）：从锁定结果 CSV 一键重建全部程序图，统一制图审美。 |
| `matlab/` | MATLAB 主实现：管线、实验入口、单元测试与锁定输出（`outputs/` 为证据 CSV/JSON）。 |
| `reference/` | 已定案的 Python 参考脚本与固定结果（影响矩阵、oracle 等），仅供追溯。 |
| `docs/` | 冻结的研究协议、部件族补充协议、任务书与学习手册。 |
| `legacy/` | 归档：被取代的旧稿与旧图（git 可追溯，不参与正稿与出图）。 |

## 快速开始

重建全部正文图（无需 MATLAB，仅依赖 numpy/pandas/matplotlib/scipy）：

```bash
cd figure_pipeline && python3 make_all.py      # 输出到 thesis/figures/（600 dpi PNG + 矢量 PDF）
```

重跑实验（需 MATLAB 与 N-CMAPSS 原始数据，见 `matlab/README.md`）：

```matlab
cd matlab/run
run_21_diagnostic_closure      % 正式闭环：检测/隔离/等级/拒识
run_22_robustness_ablation     % 约束消融与鲁棒性
run_23_repeated_splits         % 五次 unit 划分稳定性
run_24_family_diagnosis        % 五部件族诊断补充实验
```

## 结论边界（诚实性）

正式结果保留全部预注册失败项：未见故障组合检测虚警偏高（FAR≈0.55）且首次报警早于真实起点；
物理重构不一致度对单参数 HPT 故障拒识失败且具重复性；精细部件隔离对噪声较敏感；标定 $\lambda$ 跨划分不确定性较大。
详见 `thesis/结果审查与结论边界.md` 与 `docs/第三章诊断闭环研究协议.md`。数据辨识矩阵 $H$ 不宣称为第一性原理模型；
数据定义的退化等级不等同于适航或 OEM 告警阈值；合成数据结论不直接外推真实机队。
