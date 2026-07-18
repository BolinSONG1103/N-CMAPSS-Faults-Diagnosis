"""一键重建第3章全部程序生成正文图（600 dpi PNG + 矢量 PDF）。

用法：  python3 make_all.py
数据来源：仓库内锁定的 MATLAB 闭环实验 CSV/JSON（见各 fig_*.py）。
框架/算法结构图（图3-1）由作者手绘，不由本流程生成。
"""
import warnings
warnings.filterwarnings("ignore")

import style as S
import fig_influence
import fig_estimation
import fig_detection_isolation as fdi
import fig_timeline
import fig_stage_unknown_ablation as fsua

S.apply()

BUILDERS = [
    fig_influence.fig_influence,            # 图3-2 影响矩阵结构与可辨识性
    fig_estimation.fig_trajectory,          # 图3-3 部件退化轨迹估计
    fig_estimation.fig_constraint_value,    # 图3-4 约束反演的独立价值
    fdi.fig_detection,                      # 图3-5 故障检测混淆矩阵
    fdi.fig_family_isolation,               # 图3-6 五部件族故障隔离
    fig_timeline.fig_timeline,              # 图3-7 诊断闭环时间线
    fsua.fig_stage,                         # 图3-8 有序退化等级判定
    fsua.fig_unknown,                       # 图3-9 未知故障拒识
    fsua.fig_ablation,                      # 图3-10 消融与鲁棒性
]

if __name__ == "__main__":
    for build in BUILDERS:
        path = build()
        print("[ok]", path.split("/")[-1])
