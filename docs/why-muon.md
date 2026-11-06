# How Muon helps each part of an LLM, and why other optimizers do not beat it

Notes for co-authors, 2026-11-06. Every number comes from `RESULTS.md`, where each experiment lists its script and data file. The draft paper is `paper/main.tex`.

## The short version

1. **Value and output are the parts that need Muon.** Moving them to Adam costs the most, and the cost doubles each time the width doubles. Query and key barely need Muon. The three feed-forward matrices can each move alone at no cost, but not all together.
2. **Groups need different things from Muon's step.** Value and output need the weak directions of the step to keep a full-size step. Query and key need the strong directions to keep theirs.
3. **The weak directions carry information.** Scrambling where they point hurts value and output about as much as deleting them.
4. **Nothing we tried beats tuned Muon per group.** The groups fail on opposite sides of Muon's step shape, and near Muon the loss barely changes. The best changes gain about .002 at 1,000 steps and nothing at 4,000.
5. **For ICML this is not enough yet.** The value and output result is already published. The query and key side and the "why nothing beats it" account are new, but we still need a tested mechanism for query and key.

## Words used here

- **Step.** The change Muon makes to a weight matrix in one training step.
- **Directions.** Any matrix step can be split into independent directions (its singular directions). Each has a size (its singular value).
- **Strong and weak directions.** The gradient pushes hard along a few directions (strong) and softly along many others (weak). In this note, the strongest quarter are strong and the rest are weak.
- **What Muon does.** It keeps the gradient's directions and gives every direction the same size. Adam and plain gradient steps keep the gradient's uneven sizes, so strong directions dominate.
- **Power p.** We set each direction's size to (its gradient size)^p and rescale to Muon's total size. p=1 is the gradient's own shape, p=0 is Muon, and p=-1 inverts the sizes so weak directions get the biggest steps.
- **Gap.** Loss with the change minus loss of tuned Muon, on held-out worlds. Negative means Muon was better. The number in brackets is the standard error. "5/5" means the change won in 5 of 5 worlds.

## How we test

- **Model.** A small LLM with RoPE, RMSNorm, QK-norm and a gated feed-forward layer. Most step tests use width 48 with 2 layers. Size tests on Kaggle GPUs go from width 64 (0.71M weights) to width 256 (6.8M weights).
- **Worlds.** A world is its own training text, validation text and data order. Worlds 0-4 choose settings, worlds 5-9 report them, and worlds 10-14 rescore anything picked as best of many. Five more worlds are sealed and have not been read.
- **Fair baseline.** Muon's learning rate and momentum are tuned first. Every change gets its own learning rate chosen on tune worlds before it is scored.
- **Win line.** A gap above .005 with the same sign on all 5 held-out worlds.

## Finding 1: value and output need Muon, and more so as width grows

Cost of moving each group from tuned Muon to Adam at 2,000 steps, worlds 5-9:

| Moved to Adam | Width 64 | Width 128 | Width 256 |
|---|---|---|---|
| Value and output | -.0135 (.0026) | -.0202 (.0018) | -.0403 (.0019) |
| Gate, up and down | -.0022 (.0022) | -.0053 (.0015) | -.0111 (.0014) |
| Query and key | -.0002 (.0019) | +.0056 (.0014) | -.0012 (.0014) |

- Value and output cost about twice as much at each doubling of width.
- Query and key stay near zero. At width 128, Adam on query alone even beats Muon by .0063 on 5/5 worlds.
- Each feed-forward matrix moved alone costs nothing at any size.
- Longer training shrinks every gap. At width 64, moving all seven matrices costs .0433 at 500 steps, .0119 at 2,000 and .0041 at 8,000.

This ranking agrees with [Wang et al. (ICLR 2026)](https://arxiv.org/abs/2509.26030), who added Muon to one group at a time at 160M.

**Caveat found last night.** The width 128 baseline uses one shared Muon momentum (.8) for all groups. Value and output prefer .95: exact Muon at .95 on value and output alone gains +.0059 there. So the width 128 baseline undersells Muon on value and output, and their cost of moving to Adam is a lower bound. Kaggle job `papers-c` walks this properly.

## Finding 2: each group needs a different step shape

Gap from Muon when one group's direction sizes follow power p, width 48:

| Group | Steps | p=1 (gradient's shape) | Random sizes | p=-.5 | p=-1 (inverted) |
|---|---|---|---|---|---|
| Value and output | 250 | -.0229 | -.0018 | -.0036 | -.0343 |
| | 1,000 | -.0192 | +.0004 | -.0027 | -.0315 |
| | 4,000 | -.0128 | +.0005 | +.0010 | -.0238 |
| Query and key | 250 | +.0029 | -.0031 | -.0218 | -.0469 |
| | 1,000 | +.0001 | -.0011 | -.0079 | -.0371 |
| | 4,000 | -.0034 | +.0012 | -.0026 | -.0273 |
| Gate, up and down | 250 | -.0065 | -.0024 | -.0015 | -.0073 |
| | 4,000 | -.0098 | -.0014 | +.0002 | -.0029 |

What this says:

- **Value and output** fail when moved toward the gradient's shape, which starves weak directions. They are fine with weak directions a bit larger than Muon gives (p=-.5).
- **Query and key** fail when strong directions shrink (p=-.5, p=-1). In short training they are fine with the gradient's shape. That tolerance fades: +.0029 at 250 steps, +.0001 at 1,000, -.0034 at 4,000.
- **Random sizes** (Muon's directions, random sizes) stay within .003 of Muon in every group at every length.

Figure 2 of the draft plots this.

## Finding 3: the weak directions carry information

Width 48, gap from Muon:

| Group | Steps | Weak directions removed | Weak directions scrambled |
|---|---|---|---|
| Value and output | 250 | -.0112 | -.0069 |
| | 1,000 | -.0152 | -.0122 |
| Gate, up and down | 1,000 | -.0103 | -.0113 |
| Query and key | 1,000 | -.0019 | -.0032 |
| All seven | 1,000 | -.0282 | -.0275 |

"Scrambled" keeps each weak direction's size but pairs it with the wrong partner direction, so the step is the same size but points the wrong way. For value, output and feed-forward, scrambling hurts about as much as removing. So the weak directions matter because of where they point, not just because they make the step bigger. Removing them from all seven costs .0282, twice the whole Muon-vs-Adam gap of .0140, so a Muon step without weak directions is worse than Adam.

## Why other optimizers do not beat tuned Muon

**The two groups fail on opposite sides of Muon.** Value and output break on the gradient side (p=1). Query and key break on the inverted side (p=-1). Muon's equal sizes (p=0) are safe for both.

**Near Muon, the loss is flat.** Random sizes on Muon's directions change the loss by at most .003. Once the directions are right and no group is starved, exact sizes barely matter.

**So every variant that only reshapes the sizes has little room.** What we tried, each with its own tuned learning rate (and usually momentum):

| What | Best result |
|---|---|
| Published variants: Kaon, Freon (powers), Dion3 at fractions .25, .5, 1 | All lose on all hidden matrices together (-.008 to -.093). Best on one group: Freon c=1/4 on query and key, +.0032 at 250 steps |
| 1,000 random update rules (100 per part at 250 and 1,000 steps, 300 per part at 1,000) | Best v,o rule +.0025 on worlds 5-9, +.0015 on unused worlds 10-14. FFN best +.0015 → +.0009 |
| Hand-set "tilted" step on value and output (bigger weak, smaller strong, like [BulkBoost](https://arxiv.org/abs/2610.07497)) | Beats a re-tuned Muon by .0019 at 1,000 steps on 10/10 worlds, by nothing at 4,000 |
| Own step shape per group (exponent and lr chosen per group) | +.0052 at 250 steps, +.0028 at 1,000 |
| Per-head Muon on query and key (like [Group Muon](https://arxiv.org/abs/2605.08933)) | +.0032 on worlds 5-9 → +.0020 on unused worlds → +.0011 at 4,000 steps |

Two patterns repeat:

1. **Best-of-many picks shrink on fresh worlds.** Gains of +.002 to +.004 picked on worlds 5-9 fall to about +.001 on worlds 10-14.
2. **Gains fade with training length.** The tilted step, per-head Muon and query and key's tolerance of the gradient's shape all shrink between 1,000 and 4,000 steps.

Much of what first looked like a better step was a better learning rate or momentum. Re-tuning Muon itself took most of it back. For example, at 4,000 steps Muon at momentum .97 and a third of its learning rate gains +.0036, more than the tilted step's +.0031. Momentum preference is itself per group. At 4,000 steps, raising Muon's momentum to .95 helps value and output by +.0022 but query and key by only +.0004, and .97 gives query and key +.0001.

## Explanations we tested that failed

Each had a prediction written before the run.

- **QK-norm.** Prediction: QK-norm is why query and key don't need Muon. Without QK-norm, moving them to Adam still gains +.0023, so this is not the reason.
- **One-step loss drop.** Prediction: Muon's step lowers the loss on the next unseen batch more than the gradient's shape does. It is the reverse: the gradient's shape aligns three to five times better with the next batch's gradient. Muon's benefit is not a better single step.
- **Weight health (stable rank).** Prediction: weak directions keep value and output from collapsing to few directions. The gradient's shape does collapse them (value 13.7 → 8.1), but it collapses query and key even more with no loss cost. So collapse alone does not explain it.
- **Gradient concentration.** Value and output gradients are the most concentrated (.965 of energy in the top quarter), but feed-forward is less concentrated yet loses more than query and key. Concentration does not order the costs.

## What is already published, and what is ours

| Claim | Status |
|---|---|
| Value, output and feed-forward carry Muon's benefit; query and key little | Published by [Wang et al.](https://arxiv.org/abs/2509.26030) (adding Muon to Adam at 160M). They explain it as associative memory: Muon learns all stored facts, including rare ones, at an equal rate. We confirm it from the other direction (removing Muon from a tuned Muon baseline) |
| Value and output cost grows with width; all gaps shrink with length | Ours, at 0.7M to 6.8M weights |
| Query and key prefer the opposite side of Muon, and that fades with length | Ours. No paper we found splits step shape by group |
| Weak directions matter by where they point (scramble = remove) | Ours |
| Why reshaping sizes cannot beat tuned Muon (two failure sides, flat middle) | Ours. Related papers reshape the spectrum for all matrices at once: [Muon^p](https://arxiv.org/abs/2606.13867), [BulkBoost](https://arxiv.org/abs/2610.07497), [Freon and Kaon](https://arxiv.org/abs/2605.11181), [HTMuon](https://arxiv.org/abs/2603.10067) |
| Other theories of why Muon works | [Curvature](https://arxiv.org/abs/2606.04662), [river-valley](https://arxiv.org/abs/2606.21514). Not yet tested against our data |

## Is this enough for ICML?

Not yet. On the reviewer form:

- **Soundness:** good. Held-out worlds, re-tuned baselines, picks rescored on unused worlds, predictions written before runs.
- **Originality:** medium. The main value and output map repeats Wang et al. The query and key side and the flat-middle account are new.
- **Significance:** limited. Our models go up to 6.8M weights, against 160M in the closest work, and every per-group gain fades with length.
- **Main gap:** we say why value and output need Muon, using Wang et al.'s associative-memory account. We do not yet know why query and key don't. Our best guess: deciding where to attend needs only a few strong directions, while storing facts needs many. That guess has not been tested.

Estimate as it stands: weak reject to borderline. With the query and key mechanism tested and holding, borderline accept is realistic.

## Next tests, each with a written prediction

1. **How many directions each group really uses.** Cut each group's trained weights to their top-r directions and measure the loss. Prediction: query and key lose little even at small r. Value, output and feed-forward lose a lot. This takes seconds per model.
2. **Rare vs common tokens.** Split the loss by token frequency when value and output, or query and key, move to Adam. Prediction: the value and output cost sits on rare tokens, and query and key show no rare-token cost.
3. **Why query and key's tolerance fades.** Repeat test 1 at 250 and 4,000 steps. Prediction: query and key use more directions later in training.
4. **Why the value and output cost doubles with width.** Repeat test 1 at widths 64, 128 and 256. Prediction: the number of directions value and output use grows with width.

## Where things are

- [Feishu copy](https://vcn2lh6dzewa.feishu.cn/wiki/HJ1CwkImpiEpd8kle52cLR6CnAg) of this note.
- `paper/main.tex`: draft, with Table `tab:map` (size map), `tab:shape` (step shapes), `tab:weak` (weak directions) and `tab:tilt` (tilted step vs Muon).
- `RESULTS.md`: every experiment, prediction, script and data file.
- `results/`, `kaggle/out/`: raw data. `kaggle/*/job.py` is the exact code each GPU job ran.
- `debug/`: the experiment scripts (`map_parts.py` map, `spec_parts.py` step shapes, `band_grid.py` tilted step, `c_lr_grid.py` exponent and lr grids, `random_opt.py` random search).
