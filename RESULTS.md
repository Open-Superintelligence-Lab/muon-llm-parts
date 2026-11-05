# Results log

Every experiment in this repo, what ran and where its data lives. Positive gaps mean the change beat the baseline, and negative gaps mean Muon was better. Brackets hold the standard error over the five held-out worlds. All scores come from tune rows of open worlds; the sealed worlds and their numbers are kept outside this repo and are not used here.

## Where the data is

| Data | Files |
| --- | --- |
| Tuning runs on world 0 | `tune_results.jsonl`, `choices.jsonl` |
| Screening runs at the smallest size | `results/small/tiny_check.jsonl`, `tiny_screen.jsonl`, `tiny2.jsonl` |
| Design comparisons on open worlds | `results/small/open_pick.jsonl`, `open_qk.jsonl`, `open_joint.jsonl` |
| Five-world tuned rows, width 48 and width 32 | `results/small/robust.jsonl`, `long.jsonl` |
| Per-matrix map at small size | `results/small/fine.jsonl`, `fine5.jsonl`, `fine_long.jsonl`, `whole.jsonl` |
| Per-matrix map on Kaggle GPUs | `kaggle/out/map-*/map.jsonl` (every run), `map_out.jsonl` (gaps), `log.txt` |
| The exact code each Kaggle job ran | `kaggle/map-*/job.py`, made by `kaggle/make.py` |

## 2026-11-04: Variants of the update, width 48, 250 steps

Compared with a tuned per-head row on open worlds, using `debug/open_pick.py`, `debug/open_worlds.py` and `debug/pair.py`. No design reached a gap of .005.

| Design | Gap |
| --- | --- |
| Exact orthogonalization by SVD | +.0021 (.0016) |
| Raising singular values to the power .25 | -.0049 (.0016) |
| Raising singular values to the power .5 | -.0174 (.0023) |
| Removing the weak directions from the update | -.0170 (.0009) |
| Longer momentum on weak directions | -.0006 to -.0083 |

## 2026-11-04: Per-matrix map at small size

One matrix or group moves to Adam from a five-world tuned baseline with per-head query and key, using `debug/pair.py` with `FINE=1`. Tuned on worlds 0 to 4 and scored on worlds 5 to 9.

| Moved to Adam | Width 48, 2 layers, 250 steps | Width 32, 2 layers, 1,000 steps |
| --- | --- | --- |
| Value and output | -.0209 (.0015) | -.0136 (.0022) |
| Output | -.0076 (.0017) | -.0040 (.0020) |
| Value | -.0076 (.0027) | -.0021 (.0015) |
| Query | -.0019 (.0013) | -.0006 (.0013) |
| Key | -.0011 (.0006) | -.0002 (.0018) |
| Gate | .0000 (.0008) | +.0014 (.0007) |
| Up | +.0003 (.0012) | +.0013 (.0016) |
| Down | -.0001 (.0015) | +.0004 (.0013) |
| Query and key | -.0034 (.0014) | -.0028 (.0024) |
| Gate, up and down | -.0041 (.0010) | +.0012 (.0015) |
| Query, key, gate, up and down | -.0065 (.0018) | -.0029 (.0009) |
| All seven | -.0271 (.0020) | -.0176 (.0016) |

With a whole-matrix Muon baseline at width 48 (`results/small/whole.jsonl`), query and key on Adam at lr .01 gave -.0008 (.0016), at lr .025 gave +.0026 (.0023), and per-head Muon gave +.0069 (.0015), winning 5 of 5. These Adam lrs were compared on held-out worlds, so they were not chosen on tune worlds.

## 2026-11-05: Per-matrix map on Kaggle GPUs

All tiers use V=4000, 64 rows of 128 tokens per step and a 100M-token file per world, with whole-matrix Muon as the baseline. Tuned on worlds 0 and 1 with `debug/map_parts.py` and scored on worlds 5 to 9. Each cell gives the gap, the error, and how many of 5 worlds the change won.

| Moved to Adam | Width 64, 4 layers, 500 steps | Width 64, 4 layers, 2,000 steps | Width 128, 6 layers, 2,000 steps |
| --- | --- | --- | --- |
| Value and output | -.0366 (.0036) 0 | -.0135 (.0026) 0 | -.0202 (.0018) 0 |
| Value | -.0105 (.0031) 0 | -.0076 (.0033) 0 | -.0069 (.0005) 0 |
| Output | -.0136 (.0027) 0 | -.0058 (.0017) 0 | -.0103 (.0018) 0 |
| Query and key | +.0022 (.0032) 3 | -.0002 (.0019) 2 | +.0056 (.0014) 4 |
| Query | +.0010 (.0019) 4 | -.0026 (.0028) 2 | +.0063 (.0012) 5 |
| Key | +.0009 (.0025) 3 | -.0045 (.0019) 1 | +.0024 (.0012) 4 |
| Gate, up and down | -.0064 (.0028) 1 | -.0022 (.0022) 1 | -.0053 (.0015) 0 |
| Gate | -.0017 (.0033) 3 | -.0010 (.0019) 2 | +.0016 (.0021) 4 |
| Up | -.0016 (.0028) 3 | -.0009 (.0011) 2 | +.0017 (.0014) 3 |
| Down | -.0018 (.0033) 2 | -.0014 (.0015) 2 | -.0007 (.0006) 2 |
| Query, key, gate, up and down | -.0061 (.0036) 2 | -.0027 (.0011) 1 | -.0084 (.0013) 0 |
| All seven | -.0433 (.0035) 0 | -.0119 (.0017) 0 | -.0511 (.0029) 0 |
| Per-head query and key (not Adam) | +.0045 (.0023) 4 | .0000 (.0021) 2 | +.0052 (.0008) 5 |

| Setting | Width 64, 500 steps | Width 64, 2,000 steps | Width 128, 2,000 steps |
| --- | --- | --- | --- |
| Baseline held-out loss | 4.6453 | 4.4258 | 4.2629 |
| Adam lr, value and output | .01 | .01 | .004 |
| Adam lr, query and key | .025 | .0625 | .01 |
| Adam lr, feed-forward | .01 | .01 | .01 |
| Per-head lr factor | 2 | 2 | 1 |

Limits of this design: settings are tuned on 2 worlds, single matrices and joint moves inherit their group's Adam lr, and the "all seven" row was not tuned jointly.

Width 256 (`map-c`) and width 64 at 8,000 steps (`map-d`) were still running when this was written.

## Related results

Wang et al., arXiv:2509.26030, add Muon to Adam one group at a time at 160M weights and 10,000 steps with Muon's lr fixed. In their gated table, all Muon scores 3.5125, all Adam 3.8837, query and key only 3.8518, value and output only 3.6874, and value, output and feed-forward with query and key on Adam 3.5312, so Adam on query and key costs them .019.
