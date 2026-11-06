"""Figure 2: gap from Muon when one group's step sizes are its singular values to the power p (Table 5). python3 paper/fig/shape.py writes paper/fig/shape.pdf."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = [1, 0, -.5, -1]
D = {"250 steps": [("Value and output", "#2a78d6", "o", [-.0229, 0, -.0036, -.0343], [.0020, 0, .0020, .0018]),
                   ("Gate, up and down", "#eb6834", "s", [-.0065, 0, -.0015, -.0073], [.0013, 0, .0013, .0013]),
                   ("Query and key", "#1baf7a", "^", [.0029, 0, -.0218, -.0469], [.0013, 0, .0014, .0023])],
     "1,000 steps": [("Value and output", "#2a78d6", "o", [-.0192, 0, -.0027, -.0315], [.0028, 0, .0009, .0017]),
                     ("Gate, up and down", "#eb6834", "s", [-.0046, 0, .0001, -.0043], [.0015, 0, .0014, .0013]),
                     ("Query and key", "#1baf7a", "^", [.0001, 0, -.0079, -.0371], [.0015, 0, .0017, .0052])]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
for ax, (title, rows) in zip(axes, D.items()):
    ax.axhline(0, color="#888", lw=.8)
    for name, c, m, y, e in rows: ax.errorbar(P, y, yerr=e, color=c, marker=m, ms=5, lw=1.6, capsize=2, label=name)
    ax.set_xticks(P); ax.set_xticklabels(["1\ngradient", "0\nMuon", "-.5", "-1\ninverted"]); ax.invert_xaxis()
    ax.set_title(title, fontsize=9); ax.set_xlabel("Power p of singular values"); ax.grid(axis="y", color="#eee", lw=.6)
axes[0].set_ylabel("Gap from Muon\n(below 0: Muon better)"); axes[1].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); here = os.path.dirname(os.path.abspath(__file__)); fig.savefig(os.path.join(here, "shape.pdf")); fig.savefig(os.path.join(here, "shape.png"), dpi=200)
