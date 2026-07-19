"""一键重建第3章 15 张程序图（600 dpi PNG + 矢量 PDF）。

图3-1（总体框架）与图3-10（DTAE结构）由作者手绘；其余图3-2至图3-17
全部由本脚本从仓库锁定 CSV 重建，不重新拟合、不重新选参。
"""
import warnings
warnings.filterwarnings("ignore")

import os

import style as S
import dataio as D
import fig_preprocessing as prep
import fig_identifiability as ident
import fig_continuous as cont
import fig_dtae as dtae
import fig_validation as valid

S.apply()

BUILDERS = [
    prep.fig_similarity_correction,         # 图3-2
    prep.fig_baseline_quality,              # 图3-3
    ident.fig_fingerprint,                  # 图3-4
    ident.fig_condition_identifiability,    # 图3-5
    cont.fig_tracking,                      # 图3-6
    cont.fig_multiunit_error,               # 图3-7
    cont.fig_lambda_sensitivity,            # 图3-8
    cont.fig_constraint_value,              # 图3-9
    dtae.fig_detection,                     # 图3-11
    dtae.fig_part_confusion,                # 图3-12
    dtae.fig_latent,                        # 图3-13
    dtae.fig_timeline,                      # 图3-15
    dtae.fig_family_confusion,              # 图3-15
    dtae.fig_turbine_detail,                # 图3-16
    valid.fig_robustness,                   # 图3-17
]

if __name__ == "__main__":
    figdir = os.path.join(D.ROOT, "thesis", "figures")
    # 只清理程序生成图，保留 manifest；避免旧编号和旧结论混入正稿。
    for name in os.listdir(figdir):
        if name.startswith("图3-") and name.lower().endswith((".png", ".pdf")):
            os.remove(os.path.join(figdir, name))
    for build in BUILDERS:
        path = build()
        print("[ok]", path.split("/")[-1])
    print(f"[done] 共生成 {len(BUILDERS)} 张程序图（每张 PNG+PDF）。")
