"""Draws the Feishu and paper chart of the map: cost of moving each group from Muon to Adam. Numbers come from Table tab:map in paper/main.tex.

python3 debug/fig_map.py figures/map_vo_qk.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COND = ["Width 64\n500 steps", "Width 64\n2,000 steps", "Width 64\n8,000 steps", "Width 128\n2,000 steps", "Width 256\n2,000 steps"]
VO, VOE = [.0366, .0135, .0025, .0202, .0403], [.0036, .0026, .0013, .0018, .0019]
QK, QKE = [-.0022, .0002, .0033, -.0056, .0012], [.0032, .0019, .0016, .0014, .0014]
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 13})
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200, facecolor="white"); ax.set_facecolor("white")
x, w = np.arange(5), .3
ax.bar(x - w / 2 - .01, VO, w, yerr=VOE, color=BLUE, label="Value and output", error_kw=dict(ecolor=INK, lw=1.2, capsize=3))
ax.bar(x + w / 2 + .01, QK, w, yerr=QKE, color=ORANGE, label="Query and key", error_kw=dict(ecolor=INK, lw=1.2, capsize=3))
ax.axhline(0, color=INK, lw=1); ax.set_xticks(x, COND, color=INK); ax.tick_params(length=0); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}".replace("0.", ".", 1) if v >= 0 else f"-{-v:.2f}".replace("0.", ".", 1))
ax.set_ylabel("Loss gets worse by", color=MUTED); ax.tick_params(axis="y", colors=MUTED); ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
for s in ax.spines.values(): s.set_visible(False)
for i in (0, 4): ax.annotate(f"{VO[i]:.4f}".replace("0.", ".", 1), (x[i] - w / 2 - .01, VO[i] + VOE[i]), xytext=(0, 5), textcoords="offset points", ha="center", color=INK, fontsize=12)
ax.set_title("Value and output usually need Muon more than query and key do", loc="left", color=INK, fontsize=15, pad=34, x=-.06)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(-.07, 1.11), ncol=2, handlelength=1, handleheight=1, labelcolor=INK)
fig.text(.012, .012, "Mean over 5 worlds not used for tuning. Error bars show one standard error. Below zero means Adam was better.", color=MUTED, fontsize=11)
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(sys.argv[1], facecolor="white")
