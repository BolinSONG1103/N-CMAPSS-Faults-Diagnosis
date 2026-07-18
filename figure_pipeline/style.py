"""第3章图件统一样式模块。

设计目标：对齐参考博士论文（王昆）第二三章的制图审美——中文标注、语义化配色、
干净留白、序贯蓝色混淆矩阵、细边框、清晰图例。所有正文图统一从本模块取样式，
保证整章视觉一致。

配色语义（跨全章保持一致）：
    Fan=绿, HPC=蓝, HPT=红, LPT=橙, LPC=紫；
    真值=深灰, ID测试=蓝, 未见组合(OOD)=青绿。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- 中文字体
_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    _CJK = fm.FontProperties(fname=_FONT_PATH).get_name()
else:  # pragma: no cover
    _CJK = "sans-serif"

# ---------------------------------------------------------------- 语义配色
FAMILY_COLORS = {
    "HPT": "#D62728",   # 红
    "Fan": "#2CA02C",   # 绿
    "HPC": "#1F77B4",   # 蓝
    "LPT": "#FF7F0E",   # 橙
    "LPC": "#9467BD",   # 紫
}
FAMILY_ORDER = ["HPT", "Fan", "HPC", "LPT", "LPC"]

REGIME_COLORS = {
    "test_id": "#1F77B4",   # 分布内测试 —— 蓝
    "ood_combo": "#2CA02C",  # 未见组合 —— 绿
}
REGIME_LABEL = {"test_id": "分布内测试", "ood_combo": "未见组合"}

TRUTH_COLOR = "#3A3A3A"       # 真值曲线/条 —— 深灰
D_COLOR = "#D62728"           # 约束反演 D —— 红
B_COLOR = "#1F77B4"           # 无约束基线 B —— 蓝
ACCENT = "#FF7F0E"            # 强调橙
GRID = "#D9D9D9"

# 序贯蓝色混淆矩阵配色（浅→深，末端不至纯黑，便于叠加深色文字）
CONF_CMAP = LinearSegmentedColormap.from_list(
    "thesis_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
)

# 九维健康参数 → 五部件族 冻结映射（来自数据故障构造，不依结果调整）
PARAM_NAMES = ["HPT_eff_mod", "fan_eff_mod", "fan_flow_mod", "HPC_eff_mod",
               "HPC_flow_mod", "LPT_eff_mod", "LPT_flow_mod", "LPC_eff_mod",
               "LPC_flow_mod"]
PARAM_CN = ["HPT效率", "Fan效率", "Fan流量", "HPC效率", "HPC流量",
            "LPT效率", "LPT流量", "LPC效率", "LPC流量"]
FAMILY_OF_PARAM = {1: "HPT", 2: "Fan", 3: "Fan", 4: "HPC", 5: "HPC",
                   6: "LPT", 7: "LPT", 8: "LPC", 9: "LPC"}
STAGE_CN = ["健康", "早期", "中期", "严重"]


def apply():
    """设置全局绘图参数（论文级）。"""
    plt.rcParams.update({
        "font.family": _CJK,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "lines.linewidth": 1.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#BBBBBB",
        "legend.fancybox": False,
        "mathtext.fontset": "cm",
    })


def panel_tag(ax, tag, dx=-0.02, dy=1.04, fontsize=14):
    """在子图左上角标注 (a)(b)(c)，与标题分离，避免碰撞。"""
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="right")


def save(fig, path_noext):
    """同时导出 600 dpi PNG 与矢量 PDF。"""
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    fig.savefig(path_noext + ".png", dpi=600)
    fig.savefig(path_noext + ".pdf")
    plt.close(fig)
    return path_noext + ".png"
