# How Muon influences each part of an LLM

We are using this repo to write a paper for ICML 2027. The question is how Muon influences each part of an LLM, and whether custom optimizers per part help, maybe focusing on attention heads.

This repo is private and shared with co-authors. **The 12 questions to pick are on the [paper page](https://vcn2lh6dzewa.feishu.cn/wiki/NnnowGBw0iibM4kVlvvcQxpLngb), and the start point for each one is on the [Setup page](https://vcn2lh6dzewa.feishu.cn/wiki/CGBuweZ1Ai6wLNkJ1facGbBMn6f).**

The text is split into data sets that never share a token: tuning sets 0-4 pick settings, test sets 5-9 report scores, recheck sets 10-14 confirm a best-of-many pick, and 5 locked test sets are opened only by `gate.py`. The code calls a data set a world (`--world`).

- `docs/why-muon.md`: start here. What we found, why Muon helps each part, why other optimizers don't beat it, and what's next.
- `paper/main.tex`: the draft.
- `RESULTS.md`: every experiment, its numbers and where its data is.
- `results/`, `kaggle/out/`: the data behind those numbers. `kaggle/map-*/job.py` is the exact code each GPU job ran.
- `debug/map_parts.py`, `kaggle/make.py`: map where Muon matters on GPUs.
- `env.py`, `arms.py`, `rules.py`, `evals.py`: train an LLM with one optimizer per part.
- `baselines.py`, `baselines.jsonl`: train the 11 baselines once on an open data set. These scores come from before momentum tuning and must be rerun.
- `search.py`: tune lrs and pick a rule per part, on tune rows of tuning set 0 only. Its rows hold no report score.
- `gate.py`: score a design on locked test sets. A person runs it.
- `sandbox.py`, `optimizers/`: where invented optimizers live and what they may touch.
- `check.py`, `debug/`: code checks, cheat tests, leak test and gate test.
- `attnenv.py`: test query and key rules on six attention tasks.

## Inventing an optimizer

Write `optimizers/x_name.py` with `def step(w, g, st, lr, k)`, as in `optimizers/x_muon_copy.py`. Tune it on chosen parts:

```bash
python3 search.py custom x_name --modules qk,vo --tier 5s
```

Then a person submits it:

```bash
python3 gate.py x_name --tier 5s
```

An AI agent gets `search.py` and `optimizers/` only. It gets no shell, no `env.py`, no `gate.py` and no `~/.muon-llm-parts`. On a machine where an agent has a full shell, it could read that folder, so the seal needs a tool-only agent or a container.

## What stops cheating

| Cheat | What stops it |
| --- | --- |
| Reading test rows, files or the network | An optimizer gets only its weight, gradient, state, lr and knobs. File checks and a runtime hook refuse the rest. |
| Taking a second gradient or more tokens | No access to the model or data, and the gate compares tokens and gradients. |
| Memorising numbers | At most 6,000 bytes and 200 numbers per file. |
| Sharing state between weights or runs | No globals, shared defaults or other weights, and state is capped at 3x the weight. |
| Tuning on the test | Tuning sees tune rows of tuning set 0 only, and `debug/poison.py` checks this. |
| Winning on momentum that the opponent never tuned | Every arm tunes lrs and momentum per module, and the gate refuses rows without momentum tuning. |
| Reaching the training code through `torch.inspect`, `torch.sys` or `torch.os` | Those attributes are refused before running, and frame and live-object calls are blocked while running. |
| Trying designs until one wins | Every submission is logged before scoring, and each one raises the bar for later ones. |
| Winning by tuning longer | The opponent must have at least 2/3 of the design's settings. |
| Using more time or memory | Time per step at most 1.25x and state at most 2x the opponent's. |
| Editing the test or the optimizer afterwards | Code hashes sit outside the repo, and an optimizer's hash is part of its tune row. |

`debug/cheats.py` tries these and `debug/gate_test.py` checks that the gate finds a known effect and stays quiet on noise.

Earlier results and the full run history are in [llm-module-optimizers](https://github.com/vukrosic/llm-module-optimizers).

```bash
pip install -r requirements.txt
python3 setup_data.py
python3 env.py --spec muon --tier 5s --world 7
```
