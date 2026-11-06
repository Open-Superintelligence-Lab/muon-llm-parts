"""Figure 1: cost of moving one group from tuned Muon to Adam, by width, at 2,000 steps (Table 2, held-out worlds 5-9; width 512 from Kaggle jobs map-f and map-g, worlds 5-7, drawn open and dashed). python3 paper/fig/width.py writes paper/fig/width.pdf."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = [64, 128, 256]
ROWS = [("Value and output", "#2a78d6", "o", [-.0135, -.0202, -.0403], [.0026, .0018, .0019], -.0476, .0001),
        ("Gate, up and down", "#eb6834", "s", [-.0022, -.0053, -.0111], [.0022, .0015, .0014], -.0165, .0018),
        ("Query and key", "#1baf7a", "^", [-.0002, .0056, -.0012], [.0019, .0014, .0014], -.0091, .0006)]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.axhline(0, color="#888", lw=.8)
for name, c, m, y, e, y5, e5 in ROWS:
    ax.errorbar(W, y, yerr=e, color=c, marker=m, ms=5, lw=1.6, capsize=2, label=name)
    ax.plot([256, 512], [y[-1], y5], color=c, lw=1.2, ls="--"); ax.errorbar([512], [y5], yerr=[e5], color=c, marker=m, ms=5, mfc="white", capsize=2, lw=0)
ax.set_xscale("log", base=2); ax.set_xticks(W + [512]); ax.set_xticklabels([str(w) for w in W + [512]]); ax.set_xlim(52, 620)
ax.set_xlabel("Width (2,000 steps)"); ax.set_ylabel("Gap after moving to Adam\n(below 0: Muon better)")
ax.grid(axis="y", color="#eee", lw=.6); ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "width.pdf")); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "width.png"), dpi=200)
