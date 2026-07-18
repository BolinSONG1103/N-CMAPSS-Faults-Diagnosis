# figure_pipeline —— 第3章 17 张程序图预览与复核管线

从仓库锁定 CSV 生成图3-2至图3-19中的 17 张程序图；图3-1和图3-10按手绘规格完成。
每张程序图同时导出 600 dpi PNG 和矢量 PDF。Python 版本用于一键重建与信息核对，正式
排版可使用 `thesis/figure_scripts/` 中的 MATLAB/R 脚本。

## 依赖

```
python3 >= 3.9
numpy, pandas, matplotlib
中文字体：文泉驿正黑、微软雅黑、黑体或宋体
```

## 用法

```bash
python3 make_all.py        # 重建全部图到 ../thesis/figures/
python3 validate_outputs.py # 校验19个图号、17组PNG/PDF和2张手绘规格
```

单独重建某图，例如：

```bash
python3 -c "import style as S; S.apply(); import fig_dtae as F; F.fig_part_metrics()"
```

## 模块

| 文件 | 生成 |
|---|---|
| `style.py` | 全局样式、语义配色、中文字体、混淆矩阵配色、九→五族映射常量。 |
| `dataio.py` | 结果 CSV 读取与九参数→五部件族确定性聚合。 |
| `fig_preprocessing.py` | 图3-2 相似修正；图3-3 健康基准。 |
| `fig_identifiability.py` | 图3-4 故障指纹；图3-5 病态性与子空间可辨识性。 |
| `fig_continuous.py` | 图3-6至图3-9 连续退化估计。 |
| `fig_dtae.py` | 图3-11至图3-18 DTAE 检测、隔离与深度分析。 |
| `fig_validation.py` | 图3-19 物理约束管线补充鲁棒性。 |
| `make_all.py` | 一键重建 17 张程序图并清理旧编号产物。 |

所有图只读取锁定结果，不重新拟合、不重新选参。多标签混淆图采用“真值类 × 预测类共现率”，
用于同时表达逐类召回与并发退化，因此行和不要求等于 1，图注中已明确说明。
