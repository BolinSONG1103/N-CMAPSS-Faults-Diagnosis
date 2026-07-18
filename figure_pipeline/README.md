# figure_pipeline —— 第3章正文出图程序

从 `../matlab/outputs/` 的锁定结果 CSV/JSON 生成第3章全部程序图，统一制图审美
（对齐参考博士论文：中文标注、语义化配色、序贯蓝色混淆矩阵、干净留白）。

## 依赖

```
python3 >= 3.9
numpy, pandas, matplotlib, scipy
中文字体：文泉驿正黑 (/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc)
```

## 用法

```bash
python3 make_all.py        # 重建全部图到 ../thesis/figures/
```

单独重建某图，例如：

```bash
python3 -c "import style as S; S.apply(); import fig_influence as F; F.fig_influence()"
```

## 模块

| 文件 | 生成 |
|---|---|
| `style.py` | 全局样式、语义配色、中文字体、混淆矩阵配色、九→五族映射常量。 |
| `dataio.py` | 结果 CSV 读取与九参数→五部件族确定性聚合。 |
| `fig_influence.py` | 图3-2 影响矩阵结构与可辨识性。 |
| `fig_estimation.py` | 图3-3 部件退化轨迹估计；图3-4 约束反演的独立价值。 |
| `fig_detection_isolation.py` | 图3-5 故障检测混淆矩阵；图3-6 五部件族故障隔离。 |
| `fig_timeline.py` | 图3-7 诊断闭环时间线。 |
| `fig_stage_unknown_ablation.py` | 图3-8 有序退化等级；图3-9 未知故障拒识；图3-10 消融与鲁棒性。 |
| `make_all.py` | 一键重建全部图。 |

所有图只读取锁定结果，不重新拟合、不重新选参、不触碰测试真值做选择。
框架/算法结构图（图3-1）由作者手绘，不由本流程生成（规格见 `../thesis/figure_specs/`）。
