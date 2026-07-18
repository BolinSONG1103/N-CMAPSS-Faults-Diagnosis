# N-CMAPSS 硕士论文第3章：物理结构约束的气路故障诊断与退化程度评估

本项目围绕 NASA N-CMAPSS 真实飞行工况涡扇发动机退化数据，构建两条互补主线：

> 主线一：相似修正与健康基准残差 → 影响矩阵 → 非正—单调约束反演 → 连续退化程度与趋势（跨发动机泛化）。
>
> 主线二：气路残差与物理时序特征 → DTAE → 故障检测 → 四部件主隔离（分布内在线诊断）。

方法以数据辨识的传感器—部件影响矩阵表示故障物理指纹，通过非正—单调重参数化把不可逆退化先验写入解空间，以 Tikhonov 收缩处理 $\text{cond}(H_n)\approx1563$ 的病态反问题；DTAE 进一步融合残差时序、连续估计严重度动态与 T48/T50 涡轮区分特征。两条主线使用不同评估协议，仓库与论文均明确标注，不相互冒充。

## 目录结构

| 目录 | 内容 |
|---|---|
| `thesis/` | **论文交付物**：第3章正稿、19 个图号（17 张程序图+2 张手绘规格）、表格与结论边界。 |
| `figure_pipeline/` | **正文出图程序**（Python）：从锁定结果 CSV 一键重建全部程序图，统一制图审美。 |
| `matlab/` | MATLAB 主实现：管线、实验入口、单元测试与锁定输出（`outputs/` 为证据 CSV/JSON）。 |
| `reference/` | 已定案的 Python 参考脚本与固定结果（影响矩阵、oracle 等），仅供追溯。 |
| `docs/` | 冻结的研究协议、部件族补充协议、任务书与学习手册。 |
| `legacy/` | 归档：被取代的旧稿与旧图（git 可追溯，不参与正稿与出图）。 |

## 快速开始

重建 17 张程序图（无需 MATLAB/R，仅依赖 numpy/pandas/matplotlib）：

```bash
cd figure_pipeline && python3 make_all.py
python3 validate_outputs.py                    # 校验17组PNG/PDF和2张手绘规格
```

重跑实验（需 MATLAB 与 N-CMAPSS 原始数据，见 `matlab/README.md`）：

```matlab
cd matlab/run
run_prepare_complete_figure_data % 连续估计正式图数据
run_22_robustness_ablation       % 物理约束管线补充鲁棒性
```

DTAE 指标与图数据由 `improve/dtae_diagnosis.py`、`improve/dtae_export.py` 复现；正式
R/ggplot2 出图脚本见 `thesis/figure_scripts/R/`。

## 结论边界（诚实性）

连续估计采用发动机单元完全隔离协议；DTAE 诊断采用同机循环不重叠的分布内协议，两者不混称。
HPT/LPT 细分受气路指纹近共线限制，因此四部件（涡轮合并）作为主诊断，五部件仅作深度分析。
数据辨识矩阵 $H$ 不宣称为第一性原理模型，N-CMAPSS 合成数据结论不直接外推真实机队。
