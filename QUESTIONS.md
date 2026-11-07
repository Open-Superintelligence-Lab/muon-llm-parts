# Questions to pick

Pick one question, tell Vuk which one, and post your results within one week. Write how you will run it before you run it: which change, which width and length, and what result would mean yes or no. Every question has a useful answer whichever way it turns out.

**Where we are (2026-11-07).** Value and output need Muon the most, and more so as the model gets wider. Query and key need Muon only from width 512. Each feed-forward matrix can leave Muon alone at no cost, but not all three together. Nothing we tried beats tuned Muon on any part. [docs/why-muon.md](docs/why-muon.md) explains all of this; [RESULTS.md](RESULTS.md) has every run.

| # | Question | Start from |
|---|---|---|
| 1 | Can any optimizer beat tuned Muon on query and key at width 512, the first size where they need Muon? | `kaggle/make.py`, `debug/map_parts.py`, `search.py custom --modules qk` |
| 2 | Why do query and key need Muon at width 512 but not below? Head size alone was ruled out. Change width, depth and training length one at a time and see which one makes the cost appear. | `debug/map_parts.py --few --few-groups qk`, RESULTS.md entry "Is it head size, not width" |
| 3 | For value and output, the weak directions of Muon's step carry information. Can a step that keeps them, but does not give every direction the same size, beat Muon? | `debug/spec_parts.py`, `debug/band_grid.py`, `optimizers/x_band.py` |
| 4 | What is the cheapest rule for all three feed-forward matrices (gate, up, down) that matches Muon's loss? Report loss and time per step. | `search.py custom --modules gu,down`, `debug/fine_parts.py` |
| 5 | Which update rule should the embedding and the output head use? Compare at least 2 rules with the current one (Adam). | specs `muon`, `muon-all`, `sym-rownorm` in `arms.py` |
| 6 | Which update rule should the norm weights use? Compare at least 2 rules with the current one (Adam). | `norm=` in the specs in `arms.py` |
| 7 | Does giving each attention head its own step beat tuned Muon? | `optimizers/x_head_apart.py`, `debug/perhead_long.py` |
| 8 | Does Muon copy earlier text better than Adam on normal text? On text with planted repeated spans, at the same loss and the same attention, Muon's heads write the copied word about 1.6 times harder. Check it on text with no planted repeats. | `debug/induction.py`, `debug/induction_heads.py`, `debug/induction_eval.py` |
| 9 | Do the best per-part designs still help at 4,000 and 8,000 steps? Every gap so far shrinks with longer training. | `gate.py`, the per-part rows in RESULTS.md |
| 10 | Does the per-part map hold above width 512? Measure the cost of moving each part to Adam at a larger width on a GPU. | `kaggle/make.py`, `debug/map_parts.py` |
| 11 | Does value and output's need for Muon grow with width, or with fewer training tokens per weight? Wider runs also saw fewer tokens per weight, so the two are mixed. Hold one fixed and change the other. | `debug/map_parts.py --few`, partial data in `results/tpw/` |
| 12 | Which papers already use different optimizers for different parts or layers of an LLM? Find 10, one line each on what they found. Start with Wang et al. (ICLR 2026). | the "What is already published" table in `docs/why-muon.md` |

## What each answer tells us

1. **A rule wins:** the first part where a custom optimizer beats Muon, and the paper's main result. **None wins:** Muon is right for query and key at scale, and the paper says why.
2. The one that makes the cost appear is the condition under which query and key start to need Muon, which no paper reports.
3. **Wins:** Muon's equal sizes are not best for value and output. **Loses:** equal sizes are the right choice there, which explains why Muon helps value and output most.
4. **A cheaper rule matches:** feed-forward can drop part of Muon's cost. **Nothing cheaper matches:** feed-forward needs full Muon as a group.
5. **Another rule wins:** a better default for these two parts. **Adam stays best:** Muon's own recipe is right here.
6. Same as 5, for the norm weights.
7. **Wins:** heads need different steps, which supports the attention-head angle of the paper. **Loses:** one step for all heads is enough.
8. **Holds:** a mechanism for Muon's gain on real text, located in what the heads write. **Goes away:** the effect belongs to planted repeats only.
9. **Gains stay:** the designs are worth a bigger run. **Gains go:** they only speed up early training.
10. **Same ranking:** the map holds at larger size and goes in the paper. **Ranking changes:** the paper must report where it changes.
11. **Width:** wider models need Muon more for value and output. **Tokens per weight:** the cost is about undertraining, and it fades with more data.
12. The list sets what the paper can claim as new.

## Before you start

- Read [docs/why-muon.md](docs/why-muon.md) first, then the README's "What stops cheating" table.
- `pip install -r requirements.txt`, `python3 setup_data.py`, then `python3 env.py --spec muon --tier 5s --world 7` runs one small model.
- Tune Muon's learning rate and momentum for every part before you compare with it. Choose settings on worlds 0-4 and report on worlds 5-9.
- A win is a gap above .005 with the same sign on all 5 report worlds.
- Write your prediction in RESULTS.md before the run, as the earlier entries do.
