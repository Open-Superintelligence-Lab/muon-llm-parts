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

## 2026-11-05: Adding Muon to Adam with fixed lrs, width 48, 250 steps

Wang et al.'s design on our model, with `debug/map_parts.py 5s --add`, data in `results/add/w48.jsonl`. Adam everywhere uses one lr chosen on worlds 0 and 1 (.01). Muon lrs stay at the whole-matrix baseline's. All Muon scores 4.3606 and all Adam scores 4.3871 on worlds 5 to 9.

| Muon on | Against all Muon | Against all Adam |
| --- | --- | --- |
| Query and key only | -.0244 (.0027) | +.0021 |
| Value and output only | -.0066 (.0020) | +.0199 |
| Gate, up and down only | -.0188 (.0029) | +.0077 |
| Value, output, gate, up and down, with Adam's shared lr .01 on query and key | -.0025 (.0009) | +.0239 |
| The same, with Adam's lr on query and key chosen on its own (.025) | +.0023 (.0012) | +.0288 |

Query and key on Adam cost .0025 at Adam's shared lr and gain .0023 at their own lr, so at this size the sign of the query and key cost depends on that one lr. The whole-matrix map at this size is in `results/add/w48_out.jsonl`: value and output -.0198 (.0031), query and key +.0023 (.0012), gate, up and down -.0065 (.0027), per-head +.0022 (.0014).

## 2026-11-05: Adding Muon to Adam with fixed lrs, width 32, 1,000 steps

The same design at the `long` tier, data in `results/add/w32.jsonl`. Adam's shared lr and its own lr for query and key are both .01 here, so fixing that lr changes nothing. All Muon scores 4.2602 and all Adam scores 4.2749.

| Muon on | Against all Muon | Against all Adam |
| --- | --- | --- |
| Query and key only | -.0141 (.0018) | +.0006 |
| Value and output only | -.0032 (.0023) | +.0115 |
| Gate, up and down only | -.0163 (.0029) | -.0016 |
| Value, output, gate, up and down, with query and key on Adam | +.0002 (.0015) | +.0149 |
| The same, with the other Muon lrs retuned | -.0007 (.0020) | +.0140 |

The whole-matrix map at this size is in `results/add/w32_out.jsonl`: value and output -.0137 (.0040), query and key +.0002 (.0015), gate, up and down -.0024 (.0005), per-head +.0001 (.0010), all seven -.0147 (.0019).

## 2026-11-05: Adding and removing Muon agree at small size

Each group's gain from adding Muon to all Adam nearly equals its cost when removed from all Muon, and the groups nearly sum to the total, so the groups act independently at these sizes.

| Group | Width 48, added | Width 48, removed | Width 32, added | Width 32, removed |
| --- | --- | --- | --- | --- |
| Value and output | +.0199 | .0198 | +.0115 | .0137 |
| Gate, up and down | +.0077 | .0065 | -.0016 | .0024 |
| Query and key | +.0021 | -.0023 | +.0006 | -.0002 |
| Sum of added gains, against the all Muon gain | .0297 vs .0265 | | .0105 vs .0147 | |

Wang et al.'s gated table does not behave this way: each feed-forward matrix added alone gains about .09 to .10, while removing one from the full set costs .017 to .052. Their query and key cost of .0187 is 5% of their all Adam to all Muon gap of .3712, and at width 48 the cost at Adam's shared lr is .0025 of .0265, which is 9%, so the two results differ less than the raw numbers suggest.

## 2026-11-05: When the gain appears during training

From the saved curves, averaged over worlds 5 to 9. Curve values use only the first 64 validation rows at each mark, while final gaps use 3,200 rows, so one world's curve gap can differ by .01 from its final gap. Use final gaps for claims.

| Moved to Adam, width 128 | Step 250 | Step 500 | Step 1,000 | Step 2,000 |
| --- | --- | --- | --- | --- |
| All seven | -.1864 | -.1226 | -.0819 | -.0555 |
| Value and output | -.0495 | -.0309 | -.0254 | -.0221 |
| Gate, up and down | -.0548 | -.0249 | -.0157 | -.0088 |
| Query and key | +.0091 | +.0088 | +.0028 | -.0012 |

At width 64, value and output go from -.0473 at step 250 to -.0043 at step 2,000. Much of Muon's gain at these lengths is early speed, and the 8,000-step run tests whether it keeps shrinking.

## 2026-11-05: Muon on value and output alone, width 48, 1,000 steps

`debug/map_parts.py 5sx4 --add`, data in `results/add/w48x1000.jsonl`. All Muon scores 4.1323 and all Adam 4.1463 on worlds 5 to 9, with Adam's shared lr at .01.

| Muon on | Against all Muon | Against all Adam |
| --- | --- | --- |
| Value and output only | -.0001 (.0016) | +.0139 |
| Query and key only | -.0166 (.0026) | -.0026 |
| Gate, up and down only | -.0144 (.0030) | -.0005 |
| Value, output, gate, up and down | -.0001 (.0006) | +.0139 |
| The same, with other Muon lrs and Adam on query and key retuned | +.0011 (.0014) | +.0150 |

Removing from all Muon at this size: value and output -.0136 (.0031), gate, up and down -.0026 (.0022), query and key +.0011 (.0014), all seven -.0145 (.0016), per-head +.0032 (.0013). Muon on value and output alone matches all Muon here. At width 48 with 250 steps it falls short by .0066 (.0020), at width 32 with 1,000 steps by .0032 (.0023), and in Wang et al.'s gated table by .175 of .371.

## 2026-11-05: Muon on value and output alone, width 48, 4,000 steps

`debug/map_parts.py 5sx16 --add-only`, data in `results/add/w48x4000.jsonl` and `w48x4000_out.jsonl`. All Muon 3.9826, all Adam 3.9965, so the whole gap is .0139. Gaps are from all Muon on worlds 5 to 9.

| Muon on | Gap from all Muon | Gain over all Adam |
| --- | --- | --- |
| Query and key | -.0134 (.0007) | +.0005 |
| Value and output | -.0041 (.0014) | +.0098 |
| Gate, up and down | -.0106 (.0008) | +.0033 |
| Value, output, gate, up, down | -.0025 (.0006) | +.0114 |
| The same, Adam lr on query and key .01, other Muon lrs x.63 | -.0007 (.0004) | +.0132 |

Muon on value and output alone matched all Muon at 1,000 steps (-.0001) but not at 4,000 (-.0041), so that match was not stable with length. The three group gains add to .0136 against a whole gap of .0139, so the groups still act close to additively. Query and key give +.0005, which is nothing.

## 2026-11-05: Why Muon helps: its weak directions, width 48, 250 steps

Prediction written before running: removing weak directions from Muon's step hurts value and output far more than query and key. `debug/weak_parts.py`, data in `results/weak/w48.jsonl`. `optimizers/x_muon_shape.py` splits the step into its strongest quarter of directions and the weak rest. "Removed" drops the weak rest. "Scrambled" keeps its size and pairs each weak left direction with the wrong right direction, so it loses the gradient's information. The lr factor is chosen on worlds 0 and 1, and gaps are on worlds 5 to 9 from whole-matrix Muon.

| Group | Weak removed | Weak scrambled | Moved to Adam |
| --- | --- | --- | --- |
| Value and output | -.0112 (.0011) | -.0069 (.0016) | -.0198 (.0031) |
| Gate, up and down | -.0055 (.0016) | -.0074 (.0009) | -.0065 (.0027) |
| Query and key | +.0011 (.0008) | -.0006 (.0018) | +.0023 (.0012) |
| All seven | -.0141 (.0005) | -.0184 (.0022) | -.0199 (.0028) |

Exact orthogonalization by SVD on all seven gives -.0011 (.0014), so the shape code alone costs nothing. The prediction holds for query and key, which lose nothing, but feed-forward matrices also lose about as much as they do on Adam. Scrambling hurts about as much as removing, so the weak directions help through the information they carry and not only through their size. At all seven, removing weak directions accounts for .0141 of the .0199 that Adam costs.

### The same test at 1,000 steps

`debug/weak_parts.py 5sx4`, data in `results/weak/w48x1000.jsonl`. Base 4.1323, all Adam 4.1463, so moving everything to Adam costs .0140.

| Group | Weak removed | Weak scrambled | Moved to Adam |
| --- | --- | --- | --- |
| Value and output | -.0152 (.0017) | -.0122 (.0016) | -.0136 |
| Gate, up and down | -.0103 (.0008) | -.0113 (.0012) | -.0026 |
| Query and key | -.0019 (.0013) | -.0032 (.0008) | +.0011 |
| All seven | -.0282 (.0011) | -.0275 (.0024) | -.0145 |

Exact orthogonalization on all seven gives -.0005 (.0011). Removing the weak directions costs about twice what moving to Adam costs, so a Muon step without them is worse than Adam, which keeps them. The feed-forward row found lr x4 alone while the all-seven row uses one shared factor, x1, so the all-seven cost is an upper bound until each group gets its own factor. Query and key still lose the least.

## Related results

Wang et al., arXiv:2509.26030, add Muon to Adam one group at a time at 160M weights and 10,000 steps with Muon's lr fixed. In their gated table, all Muon scores 3.5125, all Adam 3.8837, query and key only 3.8518, value and output only 3.6874, and value, output and feed-forward with query and key on Adam 3.5312, so Adam on query and key costs them .019.
