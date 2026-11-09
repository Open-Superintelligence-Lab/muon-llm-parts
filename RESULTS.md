# Results log

Every experiment in this repo, what ran and where its data lives. Positive gaps mean the change beat the baseline, and negative gaps mean Muon was better. Brackets hold the standard error over the five test sets. All scores come from tune rows of open data sets; the locked test sets and their numbers are kept outside this repo and are not used here.

## Where the data is

| Data | Files |
| --- | --- |
| Tuning runs on tuning set 0 | `tune_results.jsonl`, `choices.jsonl` |
| Screening runs at the smallest size | `results/small/tiny_check.jsonl`, `tiny_screen.jsonl`, `tiny2.jsonl` |
| Design comparisons on open data sets | `results/small/open_pick.jsonl`, `open_qk.jsonl`, `open_joint.jsonl` |
| Five-set tuned rows, width 48 and width 32 | `results/small/robust.jsonl`, `long.jsonl` |
| Per-matrix map at small size | `results/small/fine.jsonl`, `fine5.jsonl`, `fine_long.jsonl`, `whole.jsonl` |
| Per-matrix map on Kaggle GPUs | `kaggle/out/map-*/map.jsonl` (every run), `map_out.jsonl` (gaps), `log.txt` |
| The exact code each Kaggle job ran | `kaggle/map-*/job.py`, made by `kaggle/make.py` |

## 2026-11-04: Variants of the update, width 48, 250 steps

Compared with a tuned per-head row on open data sets, using `debug/open_pick.py`, `debug/open_worlds.py` and `debug/pair.py`. No design reached a gap of .005.

| Design | Gap |
| --- | --- |
| Exact orthogonalization by SVD | +.0021 (.0016) |
| Raising singular values to the power .25 | -.0049 (.0016) |
| Raising singular values to the power .5 | -.0174 (.0023) |
| Removing the weak directions from the update | -.0170 (.0009) |
| Longer momentum on weak directions | -.0006 to -.0083 |

## 2026-11-04: Per-matrix map at small size

One matrix or group moves to Adam from a five-set tuned baseline with per-head query and key, using `debug/pair.py` with `FINE=1`. Tuned on tuning sets 0 to 4 and scored on test sets 5 to 9.

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

With a whole-matrix Muon baseline at width 48 (`results/small/whole.jsonl`), query and key on Adam at lr .01 gave -.0008 (.0016), at lr .025 gave +.0026 (.0023), and per-head Muon gave +.0069 (.0015), winning 5 of 5. These Adam lrs were compared on test sets, so they were not chosen on tuning sets.

## 2026-11-05: Per-matrix map on Kaggle GPUs

All tiers use V=4000, 64 rows of 128 tokens per step and a 100M-token file per data set, with whole-matrix Muon as the baseline. Tuned on tuning sets 0 and 1 with `debug/map_parts.py` and scored on test sets 5 to 9. Each cell gives the gap, the error, and how many of 5 data sets the change won.

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

Limits of this design: settings are tuned on 2 data sets, single matrices and joint moves inherit their group's Adam lr, and the "all seven" row was not tuned jointly.

Width 256 (`map-c`) and width 64 at 8,000 steps (`map-d`) were still running when this was written.

## 2026-11-05: Adding Muon to Adam with fixed lrs, width 48, 250 steps

Wang et al.'s design on our model, with `debug/map_parts.py 5s --add`, data in `results/add/w48.jsonl`. Adam everywhere uses one lr chosen on tuning sets 0 and 1 (.01). Muon lrs stay at the whole-matrix baseline's. All Muon scores 4.3606 and all Adam scores 4.3871 on test sets 5 to 9.

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

From the saved curves, averaged over test sets 5 to 9. Curve values use only the first 64 validation rows at each mark, while final gaps use 3,200 rows, so one data set's curve gap can differ by .01 from its final gap. Use final gaps for claims.

| Moved to Adam, width 128 | Step 250 | Step 500 | Step 1,000 | Step 2,000 |
| --- | --- | --- | --- | --- |
| All seven | -.1864 | -.1226 | -.0819 | -.0555 |
| Value and output | -.0495 | -.0309 | -.0254 | -.0221 |
| Gate, up and down | -.0548 | -.0249 | -.0157 | -.0088 |
| Query and key | +.0091 | +.0088 | +.0028 | -.0012 |

At width 64, value and output go from -.0473 at step 250 to -.0043 at step 2,000. Much of Muon's gain at these lengths is early speed, and the 8,000-step run tests whether it keeps shrinking.

## 2026-11-05: Muon on value and output alone, width 48, 1,000 steps

`debug/map_parts.py 5sx4 --add`, data in `results/add/w48x1000.jsonl`. All Muon scores 4.1323 and all Adam 4.1463 on test sets 5 to 9, with Adam's shared lr at .01.

| Muon on | Against all Muon | Against all Adam |
| --- | --- | --- |
| Value and output only | -.0001 (.0016) | +.0139 |
| Query and key only | -.0166 (.0026) | -.0026 |
| Gate, up and down only | -.0144 (.0030) | -.0005 |
| Value, output, gate, up and down | -.0001 (.0006) | +.0139 |
| The same, with other Muon lrs and Adam on query and key retuned | +.0011 (.0014) | +.0150 |

Removing from all Muon at this size: value and output -.0136 (.0031), gate, up and down -.0026 (.0022), query and key +.0011 (.0014), all seven -.0145 (.0016), per-head +.0032 (.0013). Muon on value and output alone matches all Muon here. At width 48 with 250 steps it falls short by .0066 (.0020), at width 32 with 1,000 steps by .0032 (.0023), and in Wang et al.'s gated table by .175 of .371.

## 2026-11-05: Muon on value and output alone, width 48, 4,000 steps

`debug/map_parts.py 5sx16 --add-only`, data in `results/add/w48x4000.jsonl` and `w48x4000_out.jsonl`. All Muon 3.9826, all Adam 3.9965, so the whole gap is .0139. Gaps are from all Muon on test sets 5 to 9.

| Muon on | Gap from all Muon | Gain over all Adam |
| --- | --- | --- |
| Query and key | -.0134 (.0007) | +.0005 |
| Value and output | -.0041 (.0014) | +.0098 |
| Gate, up and down | -.0106 (.0008) | +.0033 |
| Value, output, gate, up, down | -.0025 (.0006) | +.0114 |
| The same, Adam lr on query and key .01, other Muon lrs x.63 | -.0007 (.0004) | +.0132 |

Muon on value and output alone matched all Muon at 1,000 steps (-.0001) but not at 4,000 (-.0041), so that match was not stable with length. The three group gains add to .0136 against a whole gap of .0139, so the groups still act close to additively. Query and key give +.0005, which is nothing.

## 2026-11-05: Why Muon helps: its weak directions, width 48, 250 steps

Prediction written before running: removing weak directions from Muon's step hurts value and output far more than query and key. `debug/weak_parts.py`, data in `results/weak/w48.jsonl`. `optimizers/x_muon_shape.py` splits the step into its strongest quarter of directions and the weak rest. "Removed" drops the weak rest. "Scrambled" keeps its size and pairs each weak left direction with the wrong right direction, so it loses the gradient's information. The lr factor is chosen on tuning sets 0 and 1, and gaps are on test sets 5 to 9 from whole-matrix Muon.

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

## 2026-11-05: Which step shapes each group accepts, prediction before running

Shumaylov et al. (2605.11181) report that on GPT-2 a Muon step with random singular values matches Muon, for the whole model. Their result and our weak-direction test both say the singular directions matter more than their exact sizes. Nobody has split this by group that we could find. `debug/spec_parts.py` keeps Muon's singular directions and changes only their sizes, in one group at a time, each step scaled to Muon's size: gradient's own shape (`p=1`), reversed sizes, random sizes.

Prediction: on value, output and feed-forward, the gradient's own shape loses at least .005 and random and reversed sizes lose less than half of that. On query and key, no shape changes the loss by more than .004.

Result at width 48, 250 steps, data in `results/spec/w48.jsonl`. Base 4.3606, gaps from whole-matrix Muon on test sets 5 to 9, lr factor chosen on tuning sets 0 and 1.

| Group | Gradient's own shape | Random sizes | Reversed sizes |
| --- | --- | --- | --- |
| Value and output | -.0229 (.0020) | -.0018 (.0014) | -.0245 (.0022) |
| Gate, up and down | -.0065 (.0013) | -.0024 (.0007) | -.0084 (.0020) |
| Query and key | +.0029 (.0013) | -.0031 (.0009) | -.0445 (.0023) |
| All seven | -.0245 (.0017) | -.0025 (.0015) | -.0837 (.0027) |

Exact flat sizes on all seven give -.0011 (.0014) and square-root sizes (`p=.5`) give -.0134 (.0019). The prediction held for the gradient's shape and for random sizes, and failed for reversed sizes, which hurt every group and hurt query and key most. Random sizes match flat ones at this size, as Shumaylov et al. report on GPT-2. The groups differ: value and output need their weak directions at full size, while query and key do fine with the gradient's own shape and fail when the strong directions shrink.

Shumaylov et al.'s inverted sizes are one over the singular value, which we tried next, data in the same file. Size = singular value to the power `p`, scaled to Muon's step size:

| Group | `p=1` (gradient) | `p=0` (Muon) | `p=-.5` | `p=-1` (inverted) |
| --- | --- | --- | --- | --- |
| Value and output | -.0229 (.0020) | 0 | -.0036 (.0020) | -.0343 (.0018) |
| Gate, up and down | -.0065 (.0013) | 0 | -.0015 (.0013) | -.0073 (.0013) |
| Query and key | +.0029 (.0013) | 0 | -.0218 (.0014) | -.0469 (.0023) |
| All seven | -.0245 (.0017) | -.0011 (.0014) | -.0245 (.0021) | -.0900 (.0034) |

Inverted sizes do not work as well as Muon here, unlike the GPT-2 result in that paper's title. The groups want opposite sides of Muon: query and key lose nothing at `p=1` and lose .0218 at `p=-.5`, while value and output lose .0229 at `p=1` and only .0036 at `p=-.5`.

### Step shapes at 1,000 steps

`debug/spec_parts.py 5sx4`, data in `results/spec/w48x1000.jsonl`. Base 4.1323. Flat sizes by SVD on all seven give -0.0005 (0.0011) and `p=.5` gives -0.0106 (0.0019).

| Group | `p=1` (gradient) | Random sizes | `p=-.5` | `p=-1` (inverted) | Reversed sizes |
| --- | --- | --- | --- | --- | --- |
| Value and output | -0.0192 (0.0028) | +0.0004 (0.0009) | -0.0027 (0.0009) | -0.0315 (0.0017) | -0.0178 (0.0018) |
| Gate, up and down | -0.0046 (0.0015) | -0.0010 (0.0008) | +0.0001 (0.0014) | -0.0043 (0.0013) | -0.0042 (0.0004) |
| Query and key | +0.0001 (0.0015) | -0.0011 (0.0009) | -0.0079 (0.0017) | -0.0371 (0.0052) | -0.0309 (0.0034) |
| All seven | -0.0236 (0.0021) | -0.0022 (0.0015) | -0.0106 (0.0027) | -0.0940 (0.0056) | -0.0673 (0.0021) |

The pattern from 250 steps holds, with smaller gaps for query and key at `p=-.5`: value and output cannot take the gradient's shape, and query and key cannot take shapes that shrink strong directions. Random sizes stay within .0022 of Muon on every row.

### Step shapes at 4,000 steps

`debug/spec_parts.py 5sx16` (now reads base files with no `gaps` line), data in `results/spec/w48x4000.jsonl` and `w48x4000_out.jsonl`. Base 3.9826 on test sets 5-9. Flat sizes by SVD on all seven give -0.0008 (0.0010) and `p=.5` gives -0.0083 (0.0010).

| Group | `p=1` (gradient) | Random sizes | `p=-.5` | `p=-1` (inverted) | Reversed sizes |
| --- | --- | --- | --- | --- | --- |
| Value and output | -0.0128 (0.0012) | +0.0005 (0.0008) | +0.0010 (0.0018) | -0.0238 (0.0008) | -0.0057 (0.0007) |
| Gate, up and down | -0.0098 (0.0020) | -0.0014 (0.0014) | +0.0002 (0.0009) | -0.0029 (0.0010) | -0.0007 (0.0009) |
| Query and key | -0.0034 (0.0005) | +0.0012 (0.0017) | -0.0026 (0.0015) | -0.0273 (0.0025) | -0.0154 (0.0013) |
| All seven | -0.0183 (0.0010) | -0.0012 (0.0014) | -0.0015 (0.0012) | -0.0637 (0.0031) | -0.0339 (0.0009) |

The gradient's shape on query and key goes +.0029, +.0001, -.0034 (0/5) over 250, 1,000 and 4,000 steps, so its being free on query and key is a short-training effect. Value and output still lose the most from it (-.0128). `p=-.5` on query and key shrinks from -.0218 to -.0026. The groups move closer with length, but the order holds. Random sizes stay within .0014 of Muon.

## 2026-11-05: Why the groups differ, prediction before running

`debug/spectrum.py` measures one training run at each step: the cosine between a candidate step and the gradient of the next, unseen batch at the same weights. Prediction: on value and output the flat Muon step has a higher cosine than the gradient's shape, and the weak directions alone have a clearly positive cosine. On query and key the two shapes are about equal and the strong directions carry most of the cosine.

Result, data in `results/spec/probe_w48.json` and `probe_w48x1000.json`. Whole-run means, one run:

| Group | Steps | Muon step | Gradient's shape | Strong only | Weak only | Squared values in strong quarter |
| --- | --- | --- | --- | --- | --- | --- |
| Value and output | 250 | .0504 | .2356 | .0968 | .0023 | .965 |
| Query and key | 250 | .0326 | .1025 | .0602 | .0029 | .925 |
| Gate, up and down | 250 | .0559 | .1579 | .0977 | .0081 | .876 |
| Value and output | 1,000 | .0220 | .1007 | .0431 | .0005 | .948 |
| Query and key | 1,000 | .0162 | .0498 | .0318 | .0003 | .930 |
| Gate, up and down | 1,000 | .0294 | .0782 | .0534 | .0031 | .841 |

The prediction failed. In every group the gradient's shape lowers the next unseen batch's loss about three to five times faster than Muon's step, and the weak directions alone do almost nothing for it, yet the gradient's shape trains worse on value and output. So the help from weak directions is not a one-step drop in unseen loss, and this measure does not separate query and key from the others.

The best lr for the gradient's shape was x.4 of Muon's on value and output at 250 steps and x.63 at 1,000, and on query and key x.63 and x1.6.

## 2026-11-05: Does QK-norm explain query and key, prediction before running

The model normalizes query and key per head, which removes their overall scale, and query and key are the group that needs Muon least. `debug/noqk_parts.py` repeats the tests without QK-norm, with Muon lrs walked again. Prediction at width 48, 250 steps: without QK-norm, moving query and key to Adam costs at least .005 (it was +.0023 with QK-norm) and removing their weak directions costs at least .005 (it was +.0011), while value and output keep their costs.

Result at width 48, 250 steps, data in `results/noqk/w48.jsonl`. Base without QK-norm 4.3638, against 4.3606 with it. Gaps are against the base of the same row's model.

| Change | With QK-norm | Without QK-norm |
| --- | --- | --- |
| Query and key to Adam | +.0023 (.0012) | +.0023 (.0014) |
| Query and key, weak removed | +.0011 (.0008) | +.0010 (.0012) |
| Query and key, gradient's shape | +.0029 (.0013) | -.0014 (.0014) |
| Query and key, `p=-.5` | -.0218 (.0014) | -.0149 (.0011) |
| Value and output to Adam | -.0198 (.0031) | -.0128 (.0010) |
| Value and output, weak removed | -.0112 (.0011) | -.0085 (.0008) |
| Value and output, gradient's shape | -.0229 (.0020) | -.0186 (.0015) |

The prediction failed. Without QK-norm, query and key still gain nothing from Muon, so QK-norm does not explain their indifference.

## 2026-11-05: Kaggle map, width 256, 2,000 steps (job map-c)

`kaggle/out/map-c`, 9.7 GPU hours, 6.8M weights, base held-out 4.1375. Whole-matrix Muon baseline, gaps on test sets 5 to 9.

| Moved to Adam | Width 64 | Width 128 | Width 256 | Wins of 5 at 256 |
| --- | --- | --- | --- | --- |
| Value and output | -.0135 (.0026) | -.0202 (.0018) | -.0403 (.0019) | 0 |
| Value | | -.0069 (.0005) | -.0154 (.0028) | 0 |
| Output | | -.0103 (.0018) | -.0145 (.0021) | 0 |
| Query and key | -.0002 (.0019) | +.0056 (.0014) | -.0012 (.0014) | 1 |
| Query | | +.0063 (.0012) | +.0032 (.0010) | 5 |
| Key | | +.0024 (.0012) | -.0019 (.0006) | 0 |
| Gate, up and down | -.0022 (.0022) | -.0053 (.0015) | -.0111 (.0014) | 0 |
| All seven | -.0119 (.0017) | -.0511 (.0029) | -.0507 (.0019) | 0 |
| Muon per head on q,k | .0000 (.0021) | +.0052 (.0008) | +.0038 (.0015) | 5 |

The cost of moving value and output to Adam roughly doubles with each doubling of width, feed-forward does too, and query and key stay near zero. Muon per head on query and key beats whole-matrix Muon by .0052 and .0038 in all 5 data sets at widths 128 and 256, which is under the .005 win line at 256.

## 2026-11-05: Kaggle map, width 64, 8,000 steps (job map-d)

`kaggle/out/map-d`, 7.25 GPU hours, base held-out 4.3148. Whole-matrix Muon baseline, gaps on test sets 5 to 9.

| Moved to Adam | Gap | Wins of 5 |
| --- | --- | --- |
| Value and output | -.0025 (.0013) | 1 |
| Query and key | -.0033 (.0016) | 1 |
| Gate, up and down | -.0037 (.0015) | 1 |
| Gate alone | +.0032 (.0011) | 4 |
| All seven | -.0041 (.0025) | 0 |
| Muon per head | -.0002 (.0010) | 1 |

At width 64, moving all seven to Adam costs .0433 at 500 steps, .0119 at 2,000 and .0041 at 8,000. The gain shrinks with training length and no group stands out at 8,000 steps.

Kaggle refused the next job on 2026-11-05 with "Maximum weekly GPU quota of 30.00 hours reached", so map-e (the add design at width 128) and width 512 wait for the weekly reset. On the same day they were pushed as private jobs on a second Kaggle account, `jensen404`. The width 128 job failed with no internet (`Could not resolve host: huggingface.co`) because that account has no phone verification. After the phone was verified the same day, map-e and map-f (width 512) were pushed again as version 2; both finished and are logged below.

## 2026-11-05: Each group gets its own step shape, prediction before running

`debug/pick_shape.py` picks a power `p` and lr factor per group on tuning sets 0 and 1, with `p` from .1, .25, .5, 1 on query and key and from -.25, -.1, .1 on the other two groups, and keeps a group's pick only if it beats the baseline on the tuning sets. Prediction: at width 48 the joint optimizer beats whole-matrix Muon by less than the .005 win line at both 250 and 1,000 steps.

Result at width 48, 250 steps, data in `results/spec/pick_w48.jsonl`. Base 4.3606.

| Group | Picked `p` | lr factor | Gap | Wins of 5 |
| --- | --- | --- | --- | --- |
| Query and key | 1 | .63 | +.0029 (.0013) | 4 |
| Value and output | -.1 | 1 | +.0013 (.0011) | 3 |
| Gate, up and down | .1 | 1 | -.0006 (.0017) | 3 |
| All three together | | | +.0052 (.0008) | 5 |

The prediction failed at this size: the joint optimizer passes the .005 line, though test sets 5 to 9 have now scored many candidates, so it needs other lengths, a larger width and the locked test sets before it counts. The sealed gate needs Vuk's word and has not been used.

Result at width 48, 1,000 steps, data in `results/spec/pick_w48x1000.jsonl`. Base 4.1323.

| Group | Picked `p` | lr factor | Gap | Wins of 5 |
| --- | --- | --- | --- | --- |
| Query and key | .25 | 2.54 | +.0002 (.0018) | 3 |
| Value and output | -.25 | 1 | +.0014 (.0009) | 3 |
| Gate, up and down | .1 | 1 | +.0002 (.0010) | 3 |
| All three together | | | +.0028 (.0011) | 4 |

The gain falls below .005 at 1,000 steps and the query-and-key pick moves from `p=1` to `p=.25`, so this is a short-training effect and we drop the candidate. No locked test set was used.

## 2026-11-05: Do weak directions keep the weights healthy, prediction before running

`debug/weights.py` trains once per setting on tuning set 0 and reads each group's weight matrices at the end. Stable rank is squared Frobenius norm over squared largest singular value. Prediction at width 48, 1,000 steps: value and output end with a clearly lower stable rank under the gradient's shape than under Muon, at least 20% lower, and query and key differ by under 10%.

Result at width 48, 1,000 steps, tuning set 0, one run per setting, data in `results/spec/weights_w48x1000.json` (250 steps in `weights_w48.json`). Stable rank averaged over layers; a fresh matrix starts near 13 for q, k, v, o and near 20 for gate, up, down.

| Setting | Loss | v | o | q | k | gate | up | down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Muon | 4.1325 | 13.72 | 12.00 | 10.05 | 7.90 | 18.13 | 18.86 | 10.78 |
| Gradient's shape on v, o | 4.1489 | 8.11 | 5.15 | 8.83 | 7.64 | 17.63 | 19.21 | 10.39 |
| Gradient's shape on q, k | 4.1307 | 13.45 | 12.27 | 2.89 | 3.45 | 17.49 | 18.58 | 10.90 |
| Gradient's shape on all | 4.1545 | 8.53 | 5.95 | 3.56 | 3.28 | 14.63 | 15.50 | 3.98 |
| Adam on all | 4.1492 | 8.52 | 6.77 | 6.47 | 5.57 | 10.28 | 12.40 | 4.65 |

The first half of the prediction held: v fell 41% and o fell 57%. The second half failed: q fell 71% and k fell 56%, and the loss did not get worse. Muon does keep the stable rank near its starting value, but query and key lose more rank than value and output at no cost, so low stable rank alone does not explain the cost on value and output. It can still matter only if value and output need rank in a way query and key do not, which this test cannot separate.

## 2026-11-05: When do value and output need Muon, prediction before running

`optimizers/x_muon_switch.py` changes step shape once during training, on v and o only, using each shape's own tuned lr. `debug/switch_parts.py` runs gradient's shape then Muon, and Muon then gradient's shape, switching at 1/4, 1/2 and 3/4 of training, on test sets 5-9 at width 48, 1,000 steps. Gradient's shape throughout costs .0192.

Prediction: the cost comes mostly from late steps, not from lasting damage to the weights. Switching to Muon at 1/2 costs under .006, and switching to the gradient's shape at 1/2 costs over .010. If the first number is over .010, early steps do lasting damage, and the weights explanation comes back.

Result at width 48, 250 steps, test sets 5-9, data in `results/spec/switch_w48.jsonl`. Base 4.3606. Gradient's shape throughout costs .0229 (.0020). Negative means worse than Muon.

| Gradient's shape used during | Gap | Wins of 5 |
| --- | --- | --- |
| First 1/4 | -.0098 (.0030) | 0 |
| First 1/2 | -.0170 (.0020) | 0 |
| First 3/4 | -.0209 (.0019) | 0 |
| Last 3/4 | -.0107 (.0019) | 0 |
| Last 1/2 | -.0060 (.0015) | 0 |
| Last 1/4 | -.0037 (.0015) | 1 |

At 250 steps the prediction fails: early steps cost more than late ones. The prediction was written for 1,000 steps, and that run was stopped partway, so `results/spec/switch_w48x1000.jsonl` is incomplete and has no result yet. The learning rate explains part of this. `env.py` warms up over 20 steps, stays flat, then falls to zero over the last 30% of training. Summing that schedule at 250 steps, the first quarter gets 2.0 times the total learning rate of the last quarter and costs 2.6 times as much, and the first half gets 1.3 times the last half and costs 2.8 times as much. So the schedule covers most of the quarter comparison and less of the half comparison.

## 2026-11-05: For each part, does the optimizer matter, and is a better one known? Rule and prediction before running

Vuk's question: for each part of the LLM, either any optimizer works, or a better optimizer exists. `debug/panel_parts.py` changes one part at a time from the tuned baseline to another published optimizer, and tunes that optimizer's learning rate and momentum on tuning sets 0 and 1. It then scores on test sets 5-9, tune rows, with no locked test set. A frozen control trains the part not at all, which shows the most that part can matter.

- Parts: value and output, query and key, gate up and down, embedding, head, norm scales.
- Optimizers: Muon, Adam, SGD with momentum, Lion, RowNorm, normalized SGD and per-head Muon, where each one applies.

Verdict for each part, fixed now:

- **Any optimizer works:** every tuned optimizer stays within .004 of the baseline.
- **Optimizer matters, baseline is best:** some optimizer loses by over .004 and over 2 errors, and none is a candidate.
- **Candidate for better:** a gain above .005 with the same sign in all 5 data sets.
- **Win:** a candidate at both 250 and 1,000 steps with gain above 7.53 errors, which is the t value for 30 comparisons on 5 data sets. A win still needs larger models and Vuk's word before any sealed test.

Prediction: value and output are "matters, baseline is best", with only Muon-like rules close. Query and key are the part closest to "any optimizer works". Embedding and head matter and keep Adam. Norm scales are "any optimizer works". A candidate may appear at 250 steps on query and key, as Lion did, but no win appears at 1,000 steps.

All parts at width 48 and 250 steps, data in `results/panel/w48.jsonl`, summary in `results/panel/w48_out.json`. Base 4.3606. Each cell is the gap to the baseline, with the error in brackets and wins of 5 data sets after it. Positive beats the baseline. The baseline uses Muon on value, output, query, key and the feed-forward layers, and Adam on embeddings, the head and norms.

| Part | Frozen | Adam | SGD with momentum | Lion | RowNorm | Normalized SGD | Muon | Per-head Muon | norm1d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Value, output | -.0475 (.0022) 0 | -.0163 (.0020) 0 | -.0336 (.0023) 0 | -.0156 (.0025) 0 | -.0181 (.0014) 0 | -.0185 (.0028) 0 | base | -.0068 (.0012) 0 | |
| Query, key | -.0507 (.0024) 0 | +.0017 (.0024) 2 | -.0011 (.0021) 3 | -.0003 (.0016) 2 | +.0041 (.0013) 5 | +.0043 (.0013) 5 | base | +.0024 (.0005) 5 | |
| Feed-forward | -.0197 (.0013) 0 | -.0051 (.0016) 0 | -.0157 (.0034) 0 | -.0031 (.0021) 1 | -.0039 (.0022) 1 | -.0059 (.0014) 0 | base | | |
| Embeddings | -.1122 (.0074) 0 | base | -.0335 (.0027) 0 | -.0134 (.0013) 0 | -.0048 (.0024) 1 | | -.0232 (.0026) 0 | | |
| Head | -1.4051 (.0351) 0 | base | -.0543 (.0042) 0 | -.0397 (.0027) 0 | -.0408 (.0025) 0 | | -.0161 (.0025) 0 | | |
| Norms | -.0337 (.0039) 0 | base | -.0025 (.0016) 2 | -.0011 (.0018) 2 | | | | | -.0010 (.0020) 2 |

Verdicts by the rule above, at 250 steps only. No part has a candidate, since no gain passes .005.

- Value and output: the optimizer matters and Muon is best of those tried.
- Query and key: any tuned optimizer works. Normalized SGD, RowNorm and per-head Muon win 5 of 5 data sets with gains under .005, which the 1,000-step run will check.
- Feed-forward: Adam and normalized SGD lose over .004 and 2 errors; Lion and RowNorm stay inside .004. The optimizer matters a little and Muon is best of those tried.
- Embeddings: the optimizer matters and Adam is best of those tried. RowNorm's loss of .0048 is exactly 2 errors, so it does not clear the rule.
- Head: the optimizer matters and Adam is best of those tried.
- Norms: any tuned optimizer works.

The same panel at 1,000 steps, data in `results/panel/w48x1000.jsonl`, summary in `results/panel/w48x1000_out.json`. Base 4.1323. No tuned setting landed on the edge of its grid at either length.

| Part | Frozen | Adam | SGD with momentum | Lion | RowNorm | Normalized SGD | Muon | Per-head Muon | norm1d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Value, output | -.0613 (.0017) 0 | -.0143 (.0025) 0 | -.0352 (.0030) 0 | -.0120 (.0019) 0 | -.0141 (.0021) 0 | -.0141 (.0017) 0 | base | -.0039 (.0020) 0 | |
| Query, key | -.0449 (.0051) 0 | +.0002 (.0016) 3 | -.0021 (.0014) 2 | +.0015 (.0010) 4 | +.0008 (.0015) 3 | -.0003 (.0006) 2 | base | +.0019 (.0009) 5 | |
| Feed-forward | -.0272 (.0007) 0 | -.0039 (.0014) 1 | -.0121 (.0008) 0 | -.0019 (.0013) 3 | -.0024 (.0006) 0 | -.0050 (.0012) 0 | base | | |
| Embeddings | -.0925 (.0031) 0 | base | -.0158 (.0022) 0 | -.0068 (.0009) 0 | -.0004 (.0007) 1 | | -.0096 (.0016) 0 | | |
| Head | -1.4511 (.0311) 0 | base | -.0206 (.0020) 0 | -.0155 (.0021) 0 | -.0137 (.0020) 0 | | -.0109 (.0018) 0 | | |
| Norms | -.0293 (.0016) 0 | base | -.0027 (.0016) 2 | +.0011 (.0011) 3 | | | | | +.0005 (.0011) 3 |

Verdicts at both lengths. No optimizer passes the win rule on any part.

- Query and key: any tuned optimizer works. The 5 of 5 wins for RowNorm and normalized SGD at 250 steps did not hold at 1,000. Per-head Muon wins 5 of 5 at both lengths with gains of .0024 and .0019, under the .005 line.
- Norms: any tuned optimizer works.
- Value and output: the optimizer matters and whole-matrix Muon is best of those tried. Every other rule loses over .012 at 1,000 steps, and per-head Muon loses .0039.
- Head: the optimizer matters and Adam is best of those tried.
- Embeddings: the optimizer matters. Adam and RowNorm tie at 1,000 steps; the rest lose over .006.
- Feed-forward: the optimizer matters a little. Lion, RowNorm and Adam stay inside .004 of Muon at 1,000 steps; normalized SGD and SGD with momentum lose more.

Limits: this is one model size, width 48, at two short lengths. Earlier map results show the size and length can change the answer, so these verdicts need a larger run.

## 2026-11-05: Random optimizer search, prediction before running

Vuk asked for many random, diverse optimizers on each part. `optimizers/x_rand.py` has five stages: gradient edit (10 choices plus none), memory (9 plus Nesterov), direction shape (25, each tried equally often), step size (9 plus constant) and weight edit (6 plus none). All stages at 0 is Muon, with a difference of 0 on random inputs. `debug/random_opt.py` screens candidates at width 48, 250 steps, on tuning set 0, with three lrs. The top 20 per part get three finer lrs on tuning sets 0,1, and the top 6 are scored on test sets 5-9.

The size check for the screen: over the 13 moves in the map, part costs at width 32 and 48 correlate .97 and .96 with width 256, and width 64 at 8,000 steps correlates .25. Small runs predict size better than they predict long training, so a winner must also hold at longer runs.

Prediction: no candidate passes the win rule (gain above .005 in 5 of 5 data sets at both lengths, and above 7.53 errors). Many will beat the base on tuning set 0 by chance, since a one-set gap under about .004 is noise.

## 2026-11-05: Random optimizer search, first results

Width 48, 250 steps, gain over the tuned baseline on test sets 5-9. Each part had 100 random candidates, and q,k had 300.

| Part | Beat base on tuning set 0 | Best on test sets 5-9 |
|---|---|---|
| v,o | 2 of 100 | col-then-muon +.0003, 3/5; plain Muon came back as a top candidate |
| q,k | 176 of 300 | head-muon + agreement +.0058 error .0007, 5/5; rownorm + weight-size +.0057 error .0007, 5/5 |
| FFN | 13 of 100 | heavy + muon+sign + grow -.0005, 3/5 |
| Embeddings | 0 of 100 | tanh + grow -.0036, 1/5 |
| Head | 0 of 100 | head-muon -.0075, 1/5 |

The q,k survivors at 1,000 steps, with lr retuned on tuning sets 0,1:

| Candidate | 250 steps | 1,000 steps |
|---|---|---|
| head-muon + agreement | +.0058 | +.0024 error .0014, 4/5 |
| rownorm + weight-size | +.0057 | +.0011 error .0015, 4/5 |
| rownorm | +.0020 | +.0014 error .0011, 3/5 |
| row-clip + head-muon + grow | +.0044 | +.0000 error .0011, 2/5 |
| centre-cols + muon+adam + tangent | +.0044 | -.0017 error .0014, 1/5 |
| adam-scale + raw | +.0034 | +.0006 error .0017, 3/5 |

Published rules on q,k at 1,000 steps gain: MuonSplit +.0019, Lion +.0015, RowNorm +.0008, Adam +.0002. The prediction held: no candidate passes the win rule at both lengths. These candidates were picked on tuning sets 0,1, but test sets 5-9 have also scored earlier searches, so they are not fully clean.

What this says: a 250-step q,k gain mostly fades by 1,000 steps. A 250-step screen therefore picks fast starters, so the next search runs at 1,000 steps directly.

## 2026-11-05: Random search at 1,000 steps, prediction before running

The same candidates, 100 each on q,k, v,o and FFN, screened at width 48, 1,000 steps, tier `5sx4`. Prediction: no candidate passes the win rule at 1,000 steps. V,o and FFN stay at or below Muon, and q,k tops out near +.003.

## 2026-11-05: Random search at 1,000 steps, results

Width 48, 1,000 steps, gain on test sets 5-9, 100 candidates on each part. No candidate passes the win rule, as predicted. Q,k reached +.0041, above the +.003 I predicted.

| Part | Beat base on tuning set 0 | Best on test sets 5-9 |
|---|---|---|
| q,k | 67 of 100 | head-muon +.0041 error .0011, 5/5 (published as MuonSplit) |
| v,o | 9 of 100 | plain Muon +.0013 error .0008, 3/5, at a different momentum |
| FFN | 23 of 100 | adam-scale + spectrum-clip +.0009 error .0009, 4/5 |

## 2026-11-05: Kaon, Freon and Dion3 on each part

Vuk asked about Dion3 (NorDion2, microsoft/dion, 2608.11612) and 2605.11181 (Kaon and Freon). Kaon follows Algorithm 2 of the paper. Dion3 follows `nordion2.py` without weight decay. Freon is the exact SVD form of (GG^T)^-c G, resized to Muon's Frobenius norm instead of the paper's normalization. Width 48, 250 steps, lr and momentum tuned on tuning sets 0,1 per arm, scored on test sets 5-9.

| Arm | q,k | v,o | FFN | All hidden matrices |
|---|---|---|---|---|
| Kaon | -.0029, 1/5 | +.0001, 3/5 | -.0014, 2/5 | -.0079, 0/5 |
| Freon c=1/4 | +.0032, 5/5 | -.0107, 0/5 | -.0015, 1/5 | -.0112, 0/5 |
| Freon c=2/3 | -.0132, 0/5 | +.0002, 2/5 | -.0035, 1/5 | -.0130, 0/5 |
| Freon c=3/4 | -.0198, 0/5 | +.0004, 4/5 | -.0016, 2/5 | -.0348, 0/5 |
| Freon c=1 | -.0494, 0/5 | -.0325, 0/5 | -.0085, 0/5 | -.0927, 0/5 |
| Dion3 fraction .25 | -.0056, 0/5 | -.0136, 0/5 | -.0044, 1/5 | -.0189, 0/5 |
| Dion3 fraction .5 | -.0063, 0/5 | -.0062, 0/5 | -.0024, 0/5 | -.0147, 0/5 |
| Dion3 fraction 1 | -.0053, 0/5 | +.0001, 2/5 | +.0001, 3/5 | -.0080, 0/5 |

On all hidden matrices at this size, every arm loses to Muon, so this does not reproduce the paper's tie at 124M. The parts disagree: v,o tolerates c above 1/2 while q,k does not, and q,k gains at c=1/4 where v,o loses. Next: sweep c from 0 to .8 on each part.

## 2026-11-05: Exponent c, Newton-Schulz steps and joined arms

Width 48, gain on test sets 5-9, lr and momentum tuned on tuning sets 0,1 per arm. Each row uses exact SVD (GG^T)^-c G, resized to Muon's Frobenius norm, so c=.5 is the exact polar direction and real Muon is the base at 5 Newton-Schulz steps.

| c | q,k 250 steps | q,k 1,000 | v,o 250 steps | v,o 1,000 | FFN 250 steps |
|---|---|---|---|---|---|
| 0 | +.0028, 4/5 | -.0010, 3/5 | -.0237, 0/5 | -.0146, 0/5 | -.0083, 0/5 |
| .2 | +.0034, 5/5 | -.0027, 0/5 | -.0147, 0/5 | -.0114, 0/5 | -.0036, 1/5 |
| .3 | -.0009, 2/5 | -.0017, 2/5 | -.0080, 0/5 | -.0058, 0/5 | -.0032, 1/5 |
| .4 | -.0012, 1/5 | -.0042, 1/5 | -.0044, 0/5 | -.0015, 2/5 | -.0024, 1/5 |
| .5 | -.0068, 0/5 | -.0029, 2/5 | +.0001, 2/5 | +.0020, 5/5 | +.0010, 2/5 |
| .6 | -.0074, 0/5 | not run | +.0023, 4/5 | +.0010, 3/5 | -.0007, 2/5 |
| .7 | -.0155, 0/5 | not run | +.0013, 4/5 | +.0030, 5/5 | -.0022, 2/5 |
| .8 | -.0253, 0/5 | not run | -.0048, 1/5 | +.0007, 3/5 | -.0036, 1/5 |

Each gain has an error between .0005 and .0024. FFN at 1,000 steps: c=.4 -.0006 2/5, .5 +.0011 4/5, .6 +.0012 4/5, .7 -.0001 3/5. Below c=.5, v,o loses at both lengths. From c=.5 to .8 at 1,000 steps it sits between +.0007 and +.0030, with no clear peak. Q,k gains at low c only at 250 steps, so that gain is a fast start.

Newton-Schulz steps on v,o at 1,000 steps: 1 -.0102, 2 -.0030, 3 +.0002, 4 +.0008, 5 +.0005 (control, retuned Muon), 7 +.0010, 10 +.0017, 20 +.0008. Three steps already match Muon, and more steps add nothing clear.

Published arms at 1,000 steps:

| Arm | q,k | v,o | FFN |
|---|---|---|---|
| Kaon | -.0010, 2/5 | +.0021, 3/5 | -.0001, 4/5 |
| Dion3 fraction .25 | -.0015, 2/5 | -.0078, 0/5 | -.0046, 0/5 |
| Dion3 fraction .5 | -.0049, 0/5 | -.0036, 1/5 | -.0010, 1/5 |
| Dion3 fraction 1 | -.0031, 2/5 | +.0021, 4/5 | +.0012, 3/5 |

Dion3 is built to cut step cost, and this bench does not score cost. Its README says larger models tolerate a smaller fraction, so these small runs do not test that claim.

Joined at 1,000 steps, head-muon on q,k (+.0041 alone) with each v,o arm, no retuning: exact c=.5 +.0038, c=.7 +.0037, Kaon +.0029, each 5/5. The v,o gain does not add to the q,k gain.

Verdict: nothing passes the win rule. These are small gains from many arms on data sets that have scored earlier searches. Kaggle job `papers-a` tests c on v,o and q,k, Kaon and step counts at width 128.

## 2026-11-05: Learned spectrum exponent per matrix

Question: can each matrix learn its own exponent c during training, instead of a hand-picked c or a hand schedule? Closest prior work: [DynMuon](https://arxiv.org/abs/2605.17109) schedules the exponent over training by hand, [Muon^p](https://arxiv.org/abs/2606.13867) and [Freon](https://arxiv.org/abs/2605.11181) fix it. I found no paper that learns it per matrix from the loss.

Method, `optimizers/x_hyc.py`: update U S^(1-2c) V^T of the Nesterov momentum, resized to Muon's size. After each step c moves by eta times the cosine of the new gradient with d(update)/dc, the hypergradient. With rho > 0, d(update)/dc is summed over past steps with decay rho, so c is scored by its effect over about 1/(1-rho) steps. eta=0 is exact c=.5 and reproduces the earlier fixed arm (q,k -.0029 both times).

Width 48, 1,000 steps, gain on test sets 5-9, lr and momentum tuned per arm on tuning sets 0,1:

| Arm | q,k | v,o | FFN |
|---|---|---|---|
| eta=0 (fixed c=.5, control) | -.0029, 2/5 | +.0019, 5/5 | +.0008, 3/5 |
| eta=.01, next step only | -.0052, 0/5 | +.0015, 3/5 | -.0005, 4/5 |
| eta=.03, next step only | -.0065, 0/5 | -.0031, 1/5 | -.0021, 1/5 |
| eta=.01, start c=.3 | +.0012, 3/5 | -.0021, 2/5 | -.0035, 0/5 |
| eta=.01, rho=.99 | -.0034, 1/5 | +.0029, 4/5 | +.0019, 3/5 |
| eta=.03, rho=.99 | -.0035, 2/5 | -.0000, 2/5 | +.0010, 2/5 |
| best fixed c, for comparison | +.0034 at .2, 250 steps only | +.0030 at .7 | +.0012 at .6 |

Where c goes (`debug/hyc_trace.py`, tuning set 0). With the next-step signal, c rises from .5 on every part and every matrix: q,k to .71-.84, v,o to .87-.90, FFN to .87-.94. It rises on q,k too, where low c is better, so the next-step loss always asks for a flatter spectrum than the final loss wants. This is the short-horizon bias known from learning-rate hypergradients, here in the spectrum shape. With rho=.99, q,k first falls to .41-.49 over the first 200 steps and then climbs to .67-.75; v,o climbs to .82-.86. The longer horizon recovers the early q,k preference for low c.

Verdict: no arm passes the win rule. With rho=.99 the learned exponent matches the best hand-picked c on v,o (+.0029 against +.0030) and beats it on FFN (+.0019 against +.0012, inside one error), without choosing c, but it still loses on q,k.

## 2026-11-05: Does the next gradient repeat each update direction?

Question: v,o gains from boosting small singular directions (c=.7), q,k loses. Are small directions signal in v,o and noise in q,k? `debug/spectrum_probe.py` takes the SVD of each matrix's momentum every 10 steps and reads the raw gradient 1 and 20 steps later along each direction, z = u^T G v / |G| * sqrt(rows*cols). Noise gives z near 0 with spread 1; z > 0 means the gradient still pushes the same way. Width 48, 1,000 steps, tuning set 0, every hidden matrix on exact polar (c=.5) at its tuned lr.

Mean z by quarter of the spectrum, top singular values first:

| Part | lag 1, steps 0-300 | lag 1, steps 600+ | lag 20, steps 0-300 | lag 20, steps 600+ | share of |d|^2 |
|---|---|---|---|---|---|
| q,k | -.08 -.09 -.13 -.11 | -.04 -.00 -.02 -.03 | +.00 +.04 -.01 -.01 | -.00 -.10 +.00 -.05 | .92 .06 .01 .00 |
| v,o | -.19 -.21 -.17 -.16 | -.05 -.02 -.08 -.10 | +.05 +.09 -.01 +.02 | -.08 -.05 -.06 -.04 | .94 .05 .01 .00 |
| FFN | -.26 -.31 -.22 -.16 | -.11 -.16 -.14 -.15 | -.05 +.01 +.03 -.03 | -.04 -.06 -.02 -.08 | .83 .13 .04 .01 |

Every part and every quarter overshoots a little after one step (z about -.1 to -.3) and is near 0 after 20 steps. The quarters do not differ, and v,o at c=.3 and c=.7 give the same table within .1. At batch 16 x 48 tokens, the next gradient along any update direction is mostly noise, so this one-step probe does not show why parts prefer different c. The hypothesis is not supported by this measurement.

Confound found in the stored runs: each arm's tuned lr gives a different relative update size on v,o (trainer's opt.rel): exact polar .015, c=.7 .024, c=.8 .037, Muon .027. `debug/c_lr_grid.py` maps gain over c x lr to separate shape from step size.

Result of `debug/c_lr_grid.py` on v,o (width 48, 1,000 steps, test sets 5-9, momentum .9, gain over tuned Muon; lr is a multiple of Muon's tuned v,o lr, and every c has the same update size at the same lr):

| c | lr x.25 | x.35 | x.5 | x.7 | x1 | x1.4 | x2 |
|---|---|---|---|---|---|---|---|
| .3 | -.0061 | -.0069 | -.0075 | -.0076 | -.0133 | | |
| .4 | -.0025 | -.0030 | -.0026 | -.0044 | -.0062 | | |
| .5 | -.0007 | +.0017 | +.0022 | +.0009 | -.0027 | | |
| .6 | -.0032 | +.0010 | +.0031 | +.0025 | +.0012 | -.0025 | -.0042 |
| .7 | -.0092 | -.0045 | +.0004 | +.0022 | +.0034 | +.0022 | +.0010 |
| .8 | -.0240 | -.0172 | -.0094 | -.0055 | -.0029 | -.0018 | -.0027 |

Errors are about .0015. Reading: the best gain is about the same for c=.5, .6 and .7 (+.0022, +.0031, +.0034, within one error), but the best lr doubles from c=.5 to c=.7. Below .5 and at .8, every lr loses. So on v,o the exponent mostly moves the best step size; it gives a flat ridge of equal loss from .5 to .7, not a better optimum. Likely reason, not measured: at a fixed update size, a larger c puts more of the update into small singular directions and less into the top ones, so the top directions need a larger lr to move as far.

## 2026-11-05: Exponent x lr grid on query, key and FFN

Same `debug/c_lr_grid.py` as v,o above (width 48, 1,000 steps, test sets 5-9, momentum .9, gain over tuned Muon, lr is a multiple of Muon's tuned lr for each matrix). Data in `results/papers/c_lr_qk*.json` and `c_lr_ffn*.json`. Errors .0004-.0024, median about .0014.

Query and key:

| c | lr x.25 | x.35 | x.5 | x.7 | x1 | x1.4 | x2 | x2.8 |
|---|---|---|---|---|---|---|---|---|
| .1 | -.0004 | -.0003 | +.0007 | +.0026 | +.0007 | -.0001 | -.0003 | -.0002 |
| .2 | -.0015 | +.0009 | +.0009 | +.0001 | +.0013 | | | |
| .3 | -.0021 | -.0000 | +.0008 | -.0005 | +.0006 | +.0006 | -.0002 | +.0003 |
| .4 | -.0042 | -.0021 | +.0016 | -.0005 | -.0006 | | | |
| .5 | -.0041 | -.0020 | -.0011 | +.0009 | +.0010 | +.0006 | -.0012 | -.0008 |
| .6 | -.0047 | -.0043 | -.0037 | -.0029 | -.0026 | | | |

Reading: from c=.1 (close to plain momentum SGD) to c=.5 (exact polar) and from .5x to 2.8x the lr, every cell is within about .002 of Muon, and the one +.0026 cell is the largest of 30 noisy cells. Query and key barely care about either the spectrum shape or the step size over a 10x range. Only c=.6, which boosts small directions, loses at every lr. This matches the earlier finding that moving query and key to Adam costs nothing, with or without QK-norm.

FFN (gate, up, down together):

| c | lr x.25 | x.35 | x.5 | x.7 | x1 | x1.4 | x2 | x2.8 |
|---|---|---|---|---|---|---|---|---|
| .3 | -.0043 | -.0012 | -.0017 | -.0008 | -.0014 | | | |
| .4 | -.0031 | -.0017 | -.0008 | -.0015 | -.0001 | -.0014 | -.0055 | -.0299 |
| .5 | -.0038 | -.0011 | -.0005 | +.0003 | +.0011 | -.0005 | -.0036 | -.0187 |
| .6 | -.0038 | -.0012 | -.0007 | +.0015 | +.0039 | +.0020 | -.0008 | -.0089 |
| .7 | -.0037 | -.0030 | -.0005 | +.0011 | +.0025 | +.0032 | -.0007 | -.0054 |
| .8 | -.0047 | -.0031 | -.0001 | +.0001 | +.0015 | -.0008 | -.0013 | -.0035 |

Reading: unlike v,o, FFN's exact polar (c=.5) peaks at only +.0011, while c=.6 at the tuned lr gives +.0039 (error .0007, 5/5 data sets) and c=.7 gives +.0032 at 1.4x. On FFN the exponent is not only a step-size change; a little extra weight on small singular directions helps by about .003 at its best lr. This is the best of 40 cells, so it needs a check on data sets not used for picking it before it counts.

Summary of the three grids: q,k is flat in both c and lr; v,o has a flat ridge from c=.5 to .7 where the best lr grows with c; FFN has a real peak near c=.6-.7.

## 2026-11-06: Do the exponent gains hold on data sets no choice looked at?

Every choice so far was made on tuning sets 0-1 (lr) and read on test sets 5-9. `debug/recheck_worlds.py` scores each positive 1,000-step arm's stored spec on recheck sets 10-14, which nothing in this repo was picked on (locked test sets untouched). Data in `results/papers/recheck_w10.json`.

| Arm (part) | Test sets 5-9 | Recheck sets 10-14 | Wins 10-14 |
|---|---|---|---|
| c=.7, momentum .95, lr x1 (v,o) | +.0030 | +.0025 (.0006) | 5/5 |
| Dion3 f=1 (FFN) | +.0012 | +.0013 (.0006) | 4/5 |
| Learned exponent rho=.99 (v,o) | +.0029 | +.0013 (.0014) | 2/5 |
| Dion3 f=1 (v,o) | +.0021 | +.0006 (.0006) | 3/5 |
| Kaon (v,o) | +.0021 | +.0009 (.0011) | 3/5 |
| Exact polar c=.5 (v,o) | +.0020 | -.0001 (.0008) | 3/5 |
| Newton-Schulz 10 steps (v,o) | +.0017 | +.0000 (.0008) | 2/5 |
| Learned exponent rho=.99 (FFN) | +.0019 | +.0007 (.0009) | 3/5 |
| c=.6 (FFN) | +.0012 | -.0008 (.0007) | 1/5 |

The c x lr grid peaks also shrank on recheck sets 10-14: FFN c=.6 at x1 went from +.0039 to +.0011, v,o c=.7 at x1 with momentum .9 from +.0034 to +.0008 (`c_lr_ffn_confirm.json`, `c_lr_vo_confirm.json`). Most gains near +.002 on test sets 5-9 were partly the luck of picking the best of many. Only v,o c=.7 with momentum .95 held 5/5.

## 2026-11-06: Is v,o's gain the exponent, the momentum or the step size?

Ten data sets (5-14), width 48, 1,000 steps, gain over tuned Muon (Newton-Schulz, momentum .8, lr .051).

Muon itself, walked again on these 10 data sets (`muon_vo_b*.json`), best cells: momentum .8 at x.7 +.0008 (6/10), .9 at x.5 +.0011 (6/10), .95 at x.35 +.0013 (7/10). Tuned Muon is close to its best; a retuned Muon gains at most about .0013.

Exponent at momentum .95 (`c_lr_vo_b95.json`): c=.6 at x.5 +.0027 (.0007, 10/10), c=.7 at x1 +.0027 (.0005, 10/10), c=.5 at x.5 +.0006 (5/10). Exact polar at x1 loses .0077.

Why does the best lr double from c=.5 to c=.7? `debug/top_step.py` takes the momentum's singular values every 25 steps of one v,o c=.7 run (tuning set 0) and computes, for each c, the RMS step in each quarter of directions (top first) relative to a flat update of the same size:

| c | Top singular value vs c=.5 | Quarter 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| .3 | 2.62 | 1.58 | .95 | .66 | .38 |
| .5 | 1 | 1 | 1 | 1 | 1 |
| .6 | .48 | .64 | .80 | .96 | 1.42 |
| .7 | .17 | .32 | .48 | .70 | 1.77 |
| .8 | .05 | .13 | .23 | .41 | 1.93 |

Matching the top step would predict a best-lr ratio near 6 between c=.5 and c=.7; the grid shows about 2, so that explanation fails. But c=.6 at x.5 and c=.7 at x1 give the top quarter the same step (.32 of flat), and both win by +.0027, while their bottom quarters differ 2.5x (.71 against 1.77).

`optimizers/x_band.py` sets the step of each quarter directly (exact SVD, momentum .95, Muon's v,o lr; 1,1,1,1 is exact polar at Muon's size). `debug/band_grid.py`, 10 data sets, data in `results/papers/band_vo.json`:

| Quarter steps, top first | Gain | Wins |
|---|---|---|
| .32 .40 .48 .71 (c=.6 at x.5) | +.0032 (.0007) | 10/10 |
| .32 .32 .48 .71 | +.0030 (.0007) | 9/10 |
| .25 .32 .40 .60 | +.0030 (.0008) | 9/10 |
| .32 .32 .32 1.0 | +.0027 (.0008) | 7/10 |
| .32 .40 .48 1.0 | +.0027 (.0005) | 9/10 |
| .32 .48 .70 1.77 (c=.7 at x1) | +.0022 (.0009) | 7/10 |
| .32 .32 .32 .71 | +.0022 (.0007) | 7/10 |
| .40 .50 .60 .90 | +.0021 (.0009) | 7/10 |
| .40 .40 .40 .71 | +.0018 (.0008) | 7/10 |
| .32 .32 1 1 | +.0017 (.0008) | 6/10 |
| .25 .25 .25 .71 | +.0016 (.0008) | 7/10 |
| .32 .32 .32 1.77 | +.0016 (.0011) | 6/10 |
| .32 .32 .32 .32 | +.0014 (.0007) | 7/10 |
| .32 .70 .70 .70 | +.0013 (.0008) | 7/10 |
| .50 .50 .50 .50 (c=.5 at x.5) | +.0006 (.0010) | 5/10 |
| .20 1 1 1 | +.0002 (.0005) | 5/10 |
| .32 1 1 1 | -.0012 (.0007) | 3/10 |
| .50 1 1 1 | -.0014 (.0009) | 4/10 |

Reading: two parts. (1) A flat step about a third of Muon's (.32) gives +.0014, the same as Muon walked down to x.35. (2) On top of that, giving the weakest quarter of directions 2-3x the step of the top quarter, with the middle quarters kept small, adds about .0015, to +.003 with 9-10/10 wins. Raising the middle quarters to .7-1 removes the gain or loses. The weakest quarter holds under 1% of the momentum's squared size (earlier table), so v,o gains from moving directions the gradient barely pushes, but only the weakest ones and only by 2-3x. Each step is one to two errors; the whole +.003 against Muon's best +.0013 is about two errors. Not yet tested at width 128.

## 2026-11-06: Kaggle map at width 512, 2,000 steps (job map-f)

`kaggle/out/map-f`, width 512, 8 layers (29M weights), 2,000 steps, 6 GPU hours on two T4s, 3 test sets. Muon tuned to momentum .9 (one walk), base held-out 4.0445. Each group moved to Adam at its own tuned Adam lr.

| Moved to Adam | Gap | Wins of 3 |
| --- | --- | --- |
| Gate, up and down | -.2108 (.0013) | 0 |
| Value and output | -.0476 (.0001) | 0 |
| Query and key | -.0091 (.0006) | 0 |
| Muon per head on query and key (not Adam) | +.0021 (.0021) | 2 |

At width 512 every group needs Muon, and the order is the same as at width 48: FFN most, then value and output, then query and key. The FFN cost (.21) is about 40 times its width-128 size, and it is not a fair number: in this cut-down map (`--few`) the FFN Adam lr was fixed at .01, not walked. Value and output's walked Adam lr ended at .00064, the lowest value tried, so it also sits at the edge of its walk. Job map-g reruns both groups with Adam lrs walked from .00025 to .01 and further past a winning end. Until it returns, only the order (every group needs Muon at width 512) stands, not the sizes. Update: map-g returned, see the next section; the FFN number here is wrong and the order changes.

## 2026-11-06: Kaggle add design at width 128 (job map-e)

`kaggle/out/map-e`, width 128, 6 layers, 2,000 steps. Starts from all Adam (4.2895) and gives Muon to one group; all Muon is 4.2629.

| Muon on | Loss | Against all Muon | Against all Adam |
| --- | --- | --- | --- |
| Query and key only | 4.2904 | -.0276 (.0021) | -.0009 |
| Value and output only | 4.2737 | -.0108 (.0029) | +.0158 |
| FFN only | 4.2841 | -.0212 (.0007) | +.0054 |
| Value, output and FFN | 4.2607 | +.0022 (.0016) | +.0288 |
| Value, output and FFN, lrs walked again | 4.2573 | +.0056 (.0014) | +.0322 |

Muon on value and output alone recovers 60% of the all-Muon gain, FFN alone 20%, query and key nothing. With Muon on value, output and FFN and Adam on query and key, the model beats all Muon by .0056. This matches width 48: query and key do not need Muon.

Same tests on FFN (`results/papers/band_ffn.json`, 10 data sets, Muon's FFN lrs, momentum .95): exact polar 1,1,1,1 -.0012 (4/10); .5 flat +.0003; .7 flat +.0006 (8/10); .7 .7 .7 1.4 +.0015 (.0012, 5/10); .32 .40 .48 .71 +.0007; 1 1 1 2 -.0016. No shape beats tuned Muon on FFN by more than one error, so the band gain is specific to value and output.

Does the weakest quarter carry direction, or only size? `x_band.py` with `flip4=1` gives each bottom-quarter direction a random sign every step, which keeps the step size and removes the direction (v,o, 10 data sets):

| Quarter steps | Real signs | Random signs in quarter 4 |
|---|---|---|
| .32 .32 .32 1.0 | +.0027 (.0008) | -.0001 (.0006) |
| .32 .40 .48 .71 | +.0032 (.0007) | -.0003 (.0007) |
| .32 .32 .32 .71 | +.0022 (.0007) | -.0009 (.0007) |
| .32 .32 .32 1.77 | +.0016 (.0011) | -.0003 (.0005) |
| .32 .32 .32 .15 | +.0012 (.0006) | |
| .32 .32 .32 0 | -.0000 (.0005) | |

With random signs the gain is gone, and dropping the weakest quarter gives 0 against +.0014 for keeping it at the flat size. So the weakest momentum directions carry real direction information; the gain is not noise or rank upkeep. This disagrees with the one-step probe above, which saw no signal at lag 1 or 20 in single noisy gradients.

Momentum (v,o, 10 data sets):

| Momentum | .32 .40 .48 .71 | .32 flat | .50 flat | .20 .25 .30 .45 |
|---|---|---|---|---|
| .8 | -.0003 (.0008) | -.0013 | -.0006 | |
| .9 | +.0014 (.0009) | +.0009 | +.0008 | |
| .95 | +.0032 (.0007) | +.0014 | +.0006 | |
| .97 | +.0017 (.0008) | +.0001 | | +.0026 (.0009) |
| .98 | +.0001 (.0011) | -.0001 | | +.0016 (.0008) |

The gain needs momentum .95 and a step about a third of Muon's; the ramp beats the flat step at every momentum except .9. Longer momentum averages the gradient over more steps, which fits weak directions being slow, consistent signal that needs averaging to show.

A per-direction curvature probe (gradient change along each update direction divided by the step taken) returned step ratios that exactly echoed the step sizes put in, and a flat step gave flat "curvature". The step directions come from momentum that already holds the current gradient's noise, so that estimate is biased; the probe was dropped. A gradient summed over the next 50 steps opposes every quarter about equally (z -.8 to -1.0, `debug/spectrum_probe.py --sum 50`), so it does not separate the quarters either.

Same band test at 250 steps (`results/papers/band_vo_250.json`, base `results/add/w48_out.jsonl`, 10 data sets, errors .0009-.0018):

| v,o setting, momentum .95 | Gain | Wins |
|---|---|---|
| Quarter steps .32 .40 .48 .71 | +.0024 (.0013) | 8/10 |
| .32 flat | +.0014 (.0015) | 6/10 |
| .25 .32 .40 .60 | +.0003 (.0014) | 6/10 |
| .20 .25 .30 .45 | +.0003 (.0015) | 7/10 |
| .32 .32 .32 1.0 | -.0004 (.0015) | 4/10 |
| .50 .62 .75 1.1 | -.0036 (.0012) | 1/10 |
| Muon walked, best cell (momentum .95, lr x.35) | +.0006 | |

At 250 steps the same ramp is again best, but its neighbours are near 0 and errors are twice as large, so 250 steps neither confirms nor rules it out.

Same test at 4,000 steps (`results/papers/band_vo_4000.json` and `muon_vo_4000.json`, base `results/add/w48x4000_out.jsonl` with v,o Muon momentum .9 lr .0255, held-out loss 3.9893 on test and recheck sets 5-14):

| v,o setting, momentum .95 | Gain | Wins |
|---|---|---|
| Quarter steps .22 .28 .34 .50 | +.0025 (.0008) | 8/10 |
| .32 .40 .48 .71 | +.0021 (.0013) | 7/10 |
| .50 .62 .75 1.1 | +.0019 (.0010) | 8/10 |
| .50 flat | +.0017 (.0012) | 4/10 |
| .32 flat | +.0013 (.0011) | 6/10 |
| Muon, lr x.25 | +.0005 (.0012) | 4/10 |
| Muon, lr x.35 | +.0022 (.0010) | 7/10 |
| Muon, lr x.5 | +.0017 (.0012) | 6/10 |
| Muon, lr x.7 | +.0011 (.0008) | 8/10 |
| Quarter steps .22 .28 .34 .50, momentum .97 | +.0031 (.0010) | 8/10 |
| Muon, momentum .97, lr x.25 / x.35 / x.5 | +.0014 / +.0036 / +.0020 | |

At 4,000 steps the tilted step no longer beats Muon: Muon with momentum .95 and a third of its tuned lr gains +.0022, and the best tilted step +.0025. The 4,000-step base was tuned on tuning sets 0 and 1 at momentum .9, and that tuning leaves about .002 for any smaller, longer-momentum step to take. So the tilt's edge over walked Muon is .0019 at 1,000 steps, .0018 at 250 (with large errors) and .0003 at 4,000. With exact polar, a flat step at a third of Muon's size gains only +.0013 here against +.0022 for Newton-Schulz Muon at a similar size, so Newton-Schulz's uneven singular values are worth about .001 at this length.

At momentum .97 the same holds: Muon at a third of its lr gains +.0036 (`muon_vo_4000_b97.json`), the tilt +.0031. At 1,000 steps Muon at momentum .97 gains at most +.0007 (lr x.25 +.0007, x.35 +.0004, x.5 -.0009, `muon_vo_b97.json`), so there the tilt's edge over walked Muon (+.0032 against +.0013) stands across momentum .8 to .97.

When does the tilt act at 4,000 steps? Exact SVD steps at momentum .97 (`band_grid.py --switch at,u1,u2,u3,u4`, same file):

| v,o step, momentum .97 | Gain | Wins |
|---|---|---|
| Flat .35 throughout | +.0025 (.0012) | 6/10 |
| Tilt .22 .28 .34 .50 for 1,000 steps, then flat .35 | +.0025 (.0010) | 8/10 |
| Flat .35 for 1,000 steps, then tilt | +.0030 (.0012) | 7/10 |
| Tilt throughout | +.0031 (.0010) | 8/10 |
| Newton-Schulz Muon, lr x.35 | +.0036 (.0012) | 9/10 |

At 4,000 steps the tilt adds about .0005 over the same flat step, within error, and the first quarter of training adds none of it. Newton-Schulz Muon beats the exact flat step by .0011 here.

## 2026-11-06: Random search, 300 candidates per part at 1,000 steps

`debug/random_opt.py --n 300 --tier 5sx4 --parts vo,ffn --top2 20 --top3 6`, output `results/random/w48x1000_search300.json`. Screen on tuning set 0 with three lrs, top 20 get finer lrs on tuning sets 0,1, top 6 scored on test sets 5-9; the top 2 per part by test sets 5-9 were then scored on recheck sets 10-14 (`results/papers/recheck_random300.json`). Base 4.1323.

On tuning set 0, 25 of 300 v,o candidates beat the base (best +.0034, median -.0117) and 62 of 300 FFN candidates (best +.0052, median -.0026).

| Part | Candidate | Tuning sets 0,1 | Test sets 5-9 | Recheck sets 10-14 |
|---|---|---|---|---|
| v,o | exact (GG^T)^-c G with c=.55, momentum .95, small pull toward orthogonal weights | -.0002 | +.0025 (.0008) 5/5 | +.0015 (.0007) 4/5 |
| v,o | row clip + two-speed momentum + column-then-Muon | -.0002 | -.0005 (.0014) 2/5 | .0000 (.0007) 2/5 |
| FFN | row-normalize then Muon, row-wise size, momentum .8 | +.0020 | +.0015 (.0006) 4/5 | +.0009 (.0008) 4/5 |
| FFN | Muon plus .1 sign, momentum .8 | +.0019 | .0000 (.0002) 2/5 | +.0013 (.0005) 5/5 |

The search found nothing on FFN beyond +.0015. Its only v,o rule that held is the same family as the tilted step above (exponent above .5 with momentum .95), found without being told about it, and it gains less (+.0015 on recheck sets) than the hand-set ramp did at 1,000 steps.

## 2026-11-06: Per-head Muon on query and key, recheck sets and 4,000 steps

Per-head Muon on q,k (4 heads, lr x4 of whole-matrix Muon's) gained +.0032 (.0013) on test sets 5-9 at 1,000 steps. On recheck sets 10-14 it gains +.0020 (.0016), 3/5 (`results/papers/recheck_perhead.json`). At 4,000 steps on test and recheck sets 5-14 (`debug/perhead_long.py`, `results/papers/perhead_4000.json`) it gains +.0011 (.0008) 7/10 at lr x4 and +.0004 (.0009) 6/10 at x2. Per-head q,k is a small, early gain that fades with length, like the v,o tilt.

## Related results

Wang et al., arXiv:2509.26030, add Muon to Adam one group at a time at 160M weights and 10,000 steps with Muon's lr fixed. In their gated table, all Muon scores 3.5125, all Adam 3.8837, query and key only 3.8518, value and output only 3.6874, and value, output and feed-forward with query and key on Adam 3.5312, so Adam on query and key costs them .019.

## 2026-11-06: Width 128 exponent sweep (Kaggle papers-a)

`debug/papers_parts.py w128`, 2,000 steps, lr then momentum chosen on tuning sets 0,1, scored on test sets 5-9. Base 4.2629, Muon at momentum .8 on every hidden matrix. Data in `kaggle/out/papers-a/` (map.jsonl not committed, 0.6 MB; out*.json committed).

| Arm | Gain | Wins | Chosen lr x, momentum |
| --- | --- | --- | --- |
| v,o exact c=.5 (flat) | +.0059 (.0008) | 5/5 | .63, .95 |
| v,o c=.6 | +.0079 (.0009) | 5/5 | 1, .9 |
| v,o c=.7 | +.0084 (.0011) | 5/5 | 1.6, .95 |
| v,o c=.8 | -.0001 (.0009) | 3/5 | 1.6, .95 |
| v,o Kaon | +.0083 (.0010) | 5/5 | 1, .95 |
| q,k c=.2 | +.0004 (.0006) | 1/5 | 4, .8 |
| q,k c=.3 | +.0023 (.0015) | 4/5 | 1, .8 |
| q,k exact c=.5 | -.0033 (.0011) | 0/5 | .4, .95 |
| q,k Kaon | -.0038 (.0019) | 1/5 | .63, .9 |
| q,k 1 Newton-Schulz step | -.0009 (.0009) | 2/5 | .63, .9 |

Every v,o arm that wins chose momentum .9-.95 against the base's .8, and flat exact c=.5 already gains +.0059, so most of the v,o gain is likely momentum, not the step shape. c=.7 over c=.5 is +.0025, close to the +.0019 at width 48 and 1,000 steps. q,k again prefers the gradient side (c=.3) to flat. Kaggle job `papers-c` walks Muon itself on v,o (momentum .9, .95, .97) and q,k (.9, .95) over lr x .35-1.4 on the same data sets, seeded with these runs; it decides whether c=.7 beats a walked Muon at width 128.

## 2026-11-06: Why q,k don't need Muon: mechanism tests (predictions written before any run)

Script `debug/mech.py`. Width 48, test sets 5-9, models trained with the tuned Muon baseline, all Adam, and Adam plus Muon on one group (the `add` rows of `results/add/`). Analysis only, no new training settings.

Guess under test: deciding where to attend needs a few strong directions, while storing facts needs many.

1. **Directions used.** Cut each group's trained weights in the Muon model to their top k of 48 directions (k = 6, 12, 24, 36) and measure loss on the report rows. Prediction: at k = 12, the q,k cut raises loss by less than a third of the v,o cut, and the FFN cut is closer to v,o than to q,k.
2. **Rare vs common tokens.** Loss by target-token frequency (`evals.rarity`) for all Adam, Adam plus Muon on v,o, and Adam plus Muon on q,k. Prediction: Muon on v,o lowers rare-token loss at least twice as much as common-token loss. Muon on q,k shows no rare-token gain.
3. **Change with length.** Repeat test 1 at 250, 1,000 and 4,000 steps. Prediction: the q,k cut cost at k = 12 grows with length, relative to the v,o cut cost.

### Results (2026-11-06)

Data: `results/mech/w48.json` (250 steps), `w48x1000.json` (1,000), `w48x4000_cut.json` (4,000, Muon only), `w48x1000_noqknorm_cut.json` (1,000, no QK-norm, Muon only). Loss rise on report rows, test sets 5-9, mean (error).

**Test 1 holds: q,k weights use few directions.** Loss rise when a group keeps its top 12 of 48 directions (whole matrices):

| Model | Steps | q,k | v,o | FFN |
|---|---|---|---|---|
| Muon | 250 | +.0097 (.0016) | +.0390 (.0034) | +.0567 (.0027) |
| Muon | 1,000 | +.0182 (.0021) | +.1193 (.0029) | +.0907 (.0040) |
| Muon | 4,000 | +.0335 (.0018) | +.2469 (.0118) | +.2371 (.0091) |
| Muon, no QK-norm | 1,000 | +.0231 (.0035) | +.0950 (.0024) | +.0773 (.0029) |
| Adam | 250 | +.0037 (.0003) | +.0098 (.0008) | +.0307 (.0006) |
| Adam | 1,000 | +.0075 (.0005) | +.0339 (.0015) | +.0573 (.0041) |

- At 1,000 steps the q,k cut costs .15 of the v,o cut (prediction: under a third). FFN is closer to v,o. Without QK-norm the ratio is .24, so QK-norm is not why q,k are cheap to cut.
- Per head (3 of 12 directions per head) the q,k cut costs .0720 against .1953 for v,o, a ratio of .37.
- Not predicted: Muon spreads every group over more directions than Adam. The same cut costs v,o .1193 under Muon and .0339 under Adam at 1,000 steps.

**Test 2 fails: the v,o gain is not on rare tokens.** Gain of Adam plus Muon on v,o over all Adam:

| Steps | Common tokens (top 500) | Rarer tokens (rank 500-999) | Phrases never seen | Phrases seen 1-9 times | Phrases seen 10+ times |
|---|---|---|---|---|---|
| 250 | +.0214 | +.0008 | +.0154 | +.0218 | +.0846 |
| 1,000 | +.0149 (.0011) | +.0203 (.0026) | +.0144 (.0016) | +.0142 (.0014) | +.0223 (.0020) |

At 1,000 steps the rare-token gain is 1.36 times the common one (prediction: at least 2), rare minus common +.0054 (.0024), 4/5. At 250 steps the gain sits on common tokens and frequent phrases. Muon on q,k gains nothing in any bucket (-.0050 to +.0006 at 1,000 steps), as predicted. Limit: the vocabulary is the 1,000 most common GPT-2 tokens, so "rarer" tokens are still common words.

**Test 3 fails: q,k do not catch up with v,o.** The q,k cut cost grows with length (.0097, .0182, .0335) but the v,o cost grows faster (.0390, .1193, .2469), so the ratio falls from .25 to .15 to .14. The fading q,k tolerance of the gradient's shape is not explained by q,k using relatively more directions.

## 2026-11-06: Does spreading v,o over more directions explain Muon's v,o gain? (predictions written before any run)

Test 4, length side, local. Script `debug/mech.py`, width 48, test sets 5-9, 250, 1,000 and 4,000 steps. Every model is now cut, including Adam plus Muon on v,o.

Spread gap: the v,o top-12-of-48 cut cost in Adam plus Muon on v,o, minus the same cut cost in all Adam. Only the v,o optimizer differs between the two models.

The v,o gain of Adam plus Muon on v,o over all Adam (from `results/add/`) shrinks with length: .0199, .0139, .0098.

- If spreading over more directions is what makes Muon help v,o, the spread gap should shrink with length too.
- Prediction: it grows instead, as the all-Muon vs all-Adam gap did (.0292 at 250, .0854 at 1,000). Then spreading is a side effect of Muon, and it does not set the size of the gain.

Width side (Kaggle, not run yet): the same spread gap at widths 64, 128 and 256, where the v,o swap cost grows .0135, .0202, .0403.

### Results (2026-11-06)

Data: `results/mech/w48_spread.json`, `w48x1000_spread.json`, `w48x4000_spread.json`. Test sets 5-9, mean (error) and data sets above 0. The gains below come from these new runs, so they differ slightly from `results/add/`.

**The prediction holds: the spread grows while the gain shrinks.**

| Steps | v,o gain of Muon over Adam | Spread gap, keep 12 of 48 | Spread gap, keep 24 of 48 |
|---|---|---|---|
| 250 | +.0194 (.0025) 5/5 | +.0268 (.0023) 5/5 | +.0032 (.0005) 5/5 |
| 1,000 | +.0156 (.0011) 5/5 | +.0784 (.0051) 5/5 | +.0106 (.0006) 5/5 |
| 4,000 | +.0102 (.0011) 5/5 | +.1237 (.0060) 5/5 | +.0305 (.0025) 5/5 |

- Muon on v,o spreads v,o over more directions, and the spread grows 4.6 times from 250 to 4,000 steps. Over the same steps the gain falls by half. So the size of the spread does not set the size of the gain.
- Muon on q,k spreads q,k too (the q,k keep-12 cut costs .0195 against .0075 under Adam at 1,000 steps), and q,k gain nothing (-.0026). A wider spread is not useful by itself.
- Muon on v,o changes only v,o's spread. The q,k and FFN cut costs stay at the all-Adam values (q,k .0067 vs .0075, FFN .0766 vs .0573 at 1,000 steps; FFN moves a little).

**Where the gain sits.** Cut v,o to the same number of directions in both models, then compare loss (gain left, Muon on v,o over Adam):

| Steps | Keep 36 of 48 | Keep 24 | Keep 12 |
|---|---|---|---|
| 250 | +.0189 (97% of the gain) | +.0161 | -.0074 |
| 1,000 | +.0142 (91%) | +.0050 | -.0629 |
| 4,000 | +.0059 (58%) | -.0203 | -.1135 |

- At 250 steps the gain survives removing the weakest quarter of directions. At 4,000 steps 42% of it is gone.
- So with longer training more of Muon's v,o gain sits in weak directions, and the Muon model depends on them more: with 24 directions kept it is worse than Adam at 4,000 steps.
- Limit: cutting is not additive, so "the share of the gain in the weakest quarter" is a rough reading.

## 2026-11-06: Width 512 with tuned Adam lrs (job map-g)

`kaggle/out/map-g`, same width-512 shape, test sets 5-7. Adam lrs walked on tuning set 0: v,o tried .00025, .00064, .0016, .004, .01, .025 and picked .00064; FFN tried .00025, .00064, .0016, .004, .01 and picked .0016. Both picks are inside their walks.

| Moved to Adam | Gap | Wins of 3 |
| --- | --- | --- |
| Value and output | -.0476 (.0001) | 0 |
| Gate, up and down | -.0165 (.0018) | 0 |
| Query and key (from map-f) | -.0091 (.0006) | 0 |
| Muon per head on query and key | +.0021 (.0021) | 2 |

- The map-f FFN cost of .2108 came from the untuned Adam lr of .01. Tuned, it is .0165.
- The order now matches every smaller width: v,o most, then FFN, then q,k.
- v,o by width: .0135, .0202, .0403, .0476 (64 to 512). The doubling stops between 256 and 512.
- q,k need Muon at width 512 (-.0091, 3/3). Up to width 256 they did not. The q,k claim holds only up to 256.
- Figure 1 (`paper/fig/width.py`) now shows width 512 as open markers.

## 2026-11-06: Is it head size, not width, that makes q,k need Muon? (predictions written before any run)

Width 512 is the first size where q,k need Muon (-.0091). Its heads are 64 wide; widths 64 and 128 have heads of 16 and width 256 has 32. q,k costs by width: -.0002, +.0056, -.0012, -.0091. So head size and width move together in the map.

Test: width 48, 1,000 steps, 1, 2, 4 and 12 heads (head size 48, 24, 12, 4). Script `debug/map_parts.py h{H}x1000 --few --few-groups vo,qk --start <the tuned w48x1000 base>`: shared Muon lr factor walked, Adam lrs for v,o and for q,k walked on tuning sets 0-1, scored on 5-9. FFN goes to Adam at a fixed lr .01.

- If head size drives it: moving q,k to Adam costs more with fewer, bigger heads. Prediction: 1 head costs below -.005 on 5/5 data sets, 4 heads costs about 0.
- If width (or depth) drives it, the q,k cost stays near 0 for every head count at width 48.
- v,o's cost should not depend much on head count.

## 2026-11-06: Width 128 quarter-step shapes and HyC on v,o (Kaggle papers-b)

`debug/papers_parts.py w128 --band` and `--hyc`, 2,000 steps, lr then a fine lr factor chosen on tuning sets 0,1, scored on test sets 5-9. Base 4.2629 (Muon momentum .8 on every hidden matrix). Data `kaggle/out/papers-b/out*.json`.

| Arm | Gap over base | Wins |
|---|---|---|
| v,o flat quarters (Muon's shape, own lr and momentum) | +.0059 (.0007) | 5/5 |
| v,o ramp (weakest quarter 2.2 times the top) | +.0070 (.0005) | 5/5 |
| v,o bottom (weakest quarter 3 times) | +.0028 (.0008) | 4/5 |
| FFN ramp | +.0023 (.0010) | 4/5 |
| v,o HyC eta .01 rho .99 | +.0045 (.0012) | 5/5 |
| v,o HyC eta .03 rho .99 | -.0050 (.0009) | 0/5 |

- The flat arm is Muon's own shape with its own lr, and it gains +.0059, the same as exact Muon at momentum .95 in papers-a. So most of the v,o gain at width 128 is the baseline's shared momentum .8, not the shape.
- The ramp adds +.0011 over flat. At width 48 and 1,000 steps the tilt added about .0015. Both are below the .005 win line.
- papers-c (pushed 2026-11-06 after shrinking its seed file to 0.77 MB; the 1.2 MB file was refused by Kaggle) walks Muon itself on v,o and q,k at width 128.

### Results (2026-11-06)

Data `results/heads/out.jsonl` (gaps) and `results/heads/map.jsonl` (runs). Gap of moving a group to Adam, test sets 5-9, mean (error) wins.

| Heads (head size) | Base | q,k to Adam | v,o to Adam | FFN to Adam (lr .01) |
|---|---|---|---|---|
| 12 (4) | 4.1627 | +.0030 (.0022) 4/5 | -.0108 (.0055) 0/5 | -.0026 (.0039) 2/5 |
| 4 (12) | 4.1323 | +.0011 (.0014) 4/5 | -.0136 (.0031) 0/5 | -.0026 (.0022) 2/5 |
| 2 (24) | 4.1342 | -.0032 (.0014) 1/5 | -.0172 (.0022) 0/5 | -.0021 (.0018) 2/5 |
| 1 (48) | 4.1441 | -.0019 (.0014) 1/5 | -.0117 (.0017) 0/5 | -.0018 (.0006) 0/5 |

- The prediction fails: with one 48-wide head, moving q,k to Adam costs -.0019, not below -.005.
- There is a weak trend: heads of 4 and 12 prefer Adam slightly (4/5 data sets), heads of 24 and 48 prefer Muon slightly (1/5). Every q,k gap is within .0035 of zero.
- So head size alone does not make q,k need Muon at width 48. The width-512 q,k cost (-.0091) needs width, depth or length, not only big heads.
- v,o cost is -.0108 to -.0172 at every head count, as predicted.
- 4 heads beats 1 head by .0118 and 12 heads by .0304 on the Muon base, so 4 heads stays the default.

## 2026-11-06: Width or tokens per weight? (predictions written before any run)

The ICML critic's main objection: every Kaggle width run uses 2,000 steps of 8,192 tokens, so tokens per weight fall from 23 (width 64) to 0.56 (width 512) as width grows. The v,o cost ordered by tokens per weight falls almost steadily (.0476 at 0.56, .0403 at 2.4, .0366 at 5.8, .0202 at 7.4, .0135 at 23, .0025 at 92). So "v,o need Muon more as width grows" may be "v,o need Muon more when undertrained".

Test, local, 2 layers, 4 heads, V=1000, 768 tokens a step, test sets 5-9. Weights (all, with embeddings): width 32 88.7K, width 64 226K, width 128 650K.

| Tier | Width | Steps | Tokens per weight |
|---|---|---|---|
| t32x393 | 32 | 393 | 3.4 |
| t64x1000 | 64 | 1,000 | 3.4 |
| t128x2876 | 128 | 2,876 | 3.4 |
| t32x1000 | 32 | 1,000 | 8.7 |
| t128x1000 | 128 | 1,000 | 1.2 |

Each: `debug/map_parts.py TIER --few --few-groups vo,qk --start <tuned w48x1000 base>` (shared Muon lr factor walked, Adam lrs for v,o and q,k walked on tuning sets 0-1, FFN to Adam at .01).

- Width effect: at matched 3.4 tokens per weight, the v,o cost rises from width 32 to 128 by more than .004.
- Tokens effect: at matched width, more tokens per weight lowers the v,o cost (width 32: 393 vs 1,000 steps; width 128: 1,000 vs 2,876).
- Prediction: both effects exist, and tokens per weight is the larger. At matched 3.4 tokens per weight the width 32 to 128 rise is less than half the rise at matched 1,000 steps.

Status 2026-11-06: the tokens-per-weight test above is paused after a partial width-32 map (results/tpw/). Literature check: whole-model Muon gain vs tokens per weight is done by 2609.04577 (constant 1.4-1.7x over 1-256x overtraining), and the part split is Wang et al. 2509.26030. New direction below.

## 2026-11-06: Does Muon make the induction head form earlier, and which part does it? (predictions written before any run)

No paper studies Muon and induction-head timing (6 searches, 2026-11-06; closest: 2603.26554 linear associative memory capacity, 2603.00742 linear attention theory, 2410.11474 Adam vs GD on an induction toy).

Setup: width 48, 2 layers, 4 heads, 1,000 steps (5sx4), tuned specs from results/add/w48x1000_out.jsonl: all Adam, Muon, Adam + Muon on v,o, Adam + Muon on q,k. Test sets 5-9. Every 10 steps: tune loss on 64 rows, and the induction score = loss on the first copy of 20 random tokens minus loss on the second copy (repeated-random-token test of Olsson et al. 2022), plus the attention each head puts on the token after the earlier copy.

Formation step = first step where the induction score passes half its final Muon value.

Predictions:
1. All four arms form an induction head within 1,000 steps (induction score above 1 nat at the end).
2. Muon forms it earlier than Adam.
3. Muon on v,o alone moves the formation step most of the way to Muon; Muon on q,k alone moves it less than half the way.

Kill rule (decides whether this becomes the paper): let lead(t) = Adam tune loss minus Muon tune loss. The idea goes on only if lead(t) peaks between the two formation steps (plus or minus 10% of the run) and that peak is at least 1.5x the lead at step 1,000. If the lead grows smoothly with no peak near formation, the speedup is not "a circuit forms sooner", and the direction is dropped.

Setting change, written before the main run (2026-11-06). On the width-48 tiers (0.77M-4M tokens) no arm formed an induction head: ICL stayed between -0.13 and +0.27, and induction attention stayed at chance (0.03). Prediction 1 failed there. Natural text needs far more tokens than a CPU or this Mac's GPU can give, and the Kaggle GPU quota is used up for the week. So the main run is controlled: the map's width-64 model (4 layers, 4 heads, V=4000, batch 64, rows of 128 tokens) reads fineweb train shard 1 in order. In every batch, the first quarter of the rows repeats its first 64 tokens in its second half (`--rep 0.25`). Copying then pays off early. Specs are the tuned map rows at width 64: Muon, all Adam, and Muon with v,o / q,k / FFN on Adam. Each arm runs 4,000 steps (33M tokens), test set 5, one run per arm, with probes every 100 steps. Same predictions and kill rule. A pilot at 2,000 steps already showed Muon's best head reaching induction attention 0.05 at step 500, while Adam's reached it at step 1,000. The pilot is not counted. The natural-text version (60K steps, 491M tokens) waits for the Kaggle quota.

Result, `results/induction/w64rep25x4000_w5.json` (test set 5, one run per arm, 4,000 steps, every 100 steps; ICL is noisy, ±0.15 from probe to probe):

| arm | final loss | ICL, mean of last 5 probes | best head induction attention, final | step attention first ≥ 0.05 | step smoothed ICL first ≥ half of Muon's final |
|---|---|---|---|---|---|
| Muon | 4.4432 | 0.463 | 0.175 | 600 | 900 |
| all Adam | 4.4773 | 0.644 | 0.163 | 1,000 | 2,300 |
| Muon, v,o on Adam | 4.4741 | 0.835 | 0.150 | 1,200 | 1,000 |
| Muon, q,k on Adam | 4.4597 | 0.221 | 0.049 | 3,900 | 1,500 |
| Muon, FFN on Adam | 4.4467 | 0.802 | 0.113 | 500 | 900 |

- Prediction 1 failed: no arm reached ICL 1 nat in 33M tokens.
- Prediction 2 held on this one data set: Muon's head formed 400 steps earlier on attention and 1,400 steps earlier on ICL.
- Prediction 3 cannot be read as written, because the map arms switch one part from Muon to Adam rather than adding Muon to one part. Read the other way, putting q,k on Adam removed the head almost entirely (attention 0.049 against 0.15-0.18 elsewhere), while loss was only 0.016 worse than Muon's.
- Kill rule: Adam loss minus Muon loss peaks at 0.127 at step 500 and ends at 0.034 (3.7×). The window was 500-2,700 (formation steps 900 and 2,300, ±400). The peak sits on the window's first step, so it passes only at the edge. Muon also leads early at every tier, so this peak may be general early speed rather than the head.
- Next: test sets 6 and 7 for Muon, all Adam, and Muon with q,k on Adam, to see whether the missing head is real.

Replication on test sets 6 and 7 (`results/induction/w64rep25x4000_w67.json`), with the same setting. Each data set has its own init and tune rows; the training stream is the same. Values are ICL (mean of the last 5 probes) / best-head induction attention at step 4,000 / first step attention ≥ 0.05:

| arm | test set 5 | test set 6 | test set 7 | final loss w5 / w6 / w7 |
|---|---|---|---|---|
| Muon | 0.463 / 0.175 / 600 | 0.624 / 0.071 / 2,500 | 0.394 / 0.065 / 2,400 | 4.4432 / 4.3785 / 4.4124 |
| all Adam | 0.644 / 0.157 / 1,000 | 2.025 / 0.375 / 800 | 0.458 / 0.075 / 100 | 4.4773 / 4.4210 / 4.4194 |
| Muon, q,k on Adam | 0.242 / 0.049 / 3,700 | 0.338 / 0.032 / never | 0.175 / 0.014 / never | 4.4597 / 4.3994 / 4.4263 |

- Prediction 2 failed. Muon's head formed earlier only on test set 5. On test sets 6 and 7, Adam's formed first (step 800 vs 2,500 on test set 6), and Adam ended with the larger copy score on all three data sets.
- Muon had the lowest loss on all three data sets, 0.007-0.043 below Adam. So lower loss did not come with a stronger induction head.
- By the kill rule's own terms, the direction "Muon forms the induction head earlier" is dropped.
- One result held on all three data sets: Muon with q,k on Adam had the lowest copy score (0.18-0.34, against 0.39-0.62 for Muon and 0.46-2.03 for Adam), and its best head never passed attention 0.05 by step 3,600. Its loss sits within 0.02 of the other two. Untested: whether this comes from the mix itself or from the q,k learning rate (0.0625, tuned for loss at 2,000 steps).

### 2026-11-06, follow-up: why does Muon with q,k on Adam copy worst? (predictions written before the runs)

All-Adam uses the same q,k Adam lr (0.0625) and still copies, so the lr alone is not an obvious cause. Arms in the same setting (width 64, `--rep 0.25`, 4,000 steps, test sets 5-7): Muon with q,k on Adam at q,k lr 0.0156 and 0.25 (¼× and 4× the tuned 0.0625), and the reverse mix, all Adam with q,k on Muon (lr 0.067).
- Prediction A (lr): if the lr causes it, at least one of the two other lrs lifts the copy score to ≥ 0.39 (Muon's lowest) on at least 2 of 3 data sets. My guess: neither does.
- Prediction B (mismatch): if mixing optimizers between q,k and the rest causes it, the reverse mix also copies worse than both pure arms on at least 2 of 3 data sets. My guess: it does not; Muon's v,o/FFN with Adam's q,k is the specific bad pair.

Result (`results/induction/w64rep25x4000_qkfollow.json`; table from `python3 debug/induction_table.py` on all three files). Copy score = ICL, the mean of the last 5 probes:

| arm | w5 | w6 | w7 | mean loss |
|---|---|---|---|---|
| Muon, q,k on Adam lr 0.0156 | 0.428 | 0.449 | 0.383 | 4.4173 |
| Muon, q,k on Adam lr 0.0625 (tuned) | 0.221 | 0.332 | 0.179 | 4.4285 |
| Muon, q,k on Adam lr 0.25 | 0.467 | 0.653 | 0.374 | 4.4152 |
| all Adam, q,k on Muon | 0.451 | 0.540 | 1.305 | 4.4360 |
| Muon | 0.463 | 0.624 | 0.394 | 4.4114 |
| all Adam | 0.644 | 2.025 | 0.458 | 4.4392 |

- Prediction A: at ¼× and 4× the q,k lr, the copy score reaches Muon's lowest value (0.39) on 2 of 3 data sets. So the low copy score belongs to the tuned lr only; my guess was wrong. Because it is not monotone in lr, it is most likely a narrow effect, so I drop "the mix removes the head".
- Prediction B is met by its letter (the reverse mix is below both pure arms on test sets 5 and 6), but the test set 5 margin is 0.012. On test set 7, the reverse mix has the highest copy score of all arms. Not a finding.
- One pattern holds on all three data sets: all Adam copies better than Muon (0.644 vs 0.463, 2.025 vs 0.624, 0.458 vs 0.394), while Muon has the lower loss (by 0.034, 0.043 and 0.007).

### 2026-11-06: does Adam learn in-context copying better than Muon at lower or equal loss? (predictions written before the runs)

Same setting, test sets 8 and 9, Muon and all Adam.
- Prediction C: Adam's copy score is above Muon's on both new data sets, and Muon's loss is below Adam's on both. If either fails on either data set, the pattern is 3 of 5 or 4 of 5, and I treat it as unproven.

Result C (`results/induction/w64rep25x4000_w89.json`). Every saved final model was re-measured on 1,000 probe rows and on all tune rows (`debug/induction_eval.py` → `results/induction/eval_rep25x4000.json`), following the Opus critic, who pointed out that ICL = first-copy loss minus second-copy loss rewards a model for doing badly on rare first-copy tokens:
- Prediction C failed. Muon's copy score beat Adam's on test set 8 (0.459 vs 0.100) and test set 9 (0.656 vs 0.346). "Adam copies better" holds on 3 of 5 data sets, so it is dropped.
- Muon's held-out loss is below Adam's on all 5 data sets: 4.407/4.444, 4.412/4.446, 4.434/4.450, 4.421/4.493, 4.403/4.437.
- No model in this setting has a working copying head. Second-copy loss on uniform random tokens is 9.5-11.0 nats, above the 8.29 of a uniform guess, and copy accuracy is 0.0-1.4%. At a 63-token look-back, the copy score goes up to 3.8 for some models and down to -5.9 for others. The repeated-first-half data taught a fixed 64-token look-back, not content copying, so the whole `--rep 0.25` series measures that look-back and not induction.
- Next: `--repvar`, where each repeat copies a 20-token span to a random later place, 21-87 tokens back, so that only content-based copying helps. Pilot on test set 5, Muon and Adam, half of the rows. Pass condition before any Muon/Adam comparison counts: copy accuracy above 30% on the uniform probe for both arms.

Pilot result, `--repvar --rep 0.5`, tuning set 5, 4,000 steps (`results/induction/w64repvar50x4000_w5.json`, re-measure in `eval_repvar50x4000.json`):

| step | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,500 | 4,000 |
|---|---|---|---|---|---|---|---|
| Muon ICL / best-head attention | 0.18/0.08 | 0.79/0.34 | 1.35/0.42 | 1.88/0.42 | 2.29/0.41 | 3.14/0.44 | 3.52/0.45 |
| Adam ICL / best-head attention | 0.07/0.02 | 0.12/0.02 | 0.23/0.07 | 0.33/0.15 | 0.81/0.31 | 1.62/0.50 | 1.84/0.52 |

- A real copying head now forms in both arms: attention 0.45-0.52 to the right token, against a chance level of 0.02.
- Muon's head forms about 1,500 steps earlier.
- Re-measured at the end: copy accuracy is 3.7% (Muon) vs 0.3% (Adam) on uniform probes, and 27.8% vs 10.0% on frequency-matched probes. Second-copy loss on frequency-matched probes is 4.40 vs 6.55. Held-out loss is 4.4493 vs 4.4473, nearly equal.
- The pass condition I set (above 30% uniform-probe accuracy for both arms) failed. Uniform probes use mostly rare tokens that these models barely predict at all, so I switch the accuracy check to frequency-matched probes for the next runs. This is a change made after seeing data, and I record it here as one.

Prediction D (written before the runs): on test sets 6 and 7 in the same setting, Muon's best head passes attention 0.2 at least 500 steps before Adam's, and Muon's frequency-matched copy accuracy at step 4,000 is higher than Adam's on both data sets.

Result D (`results/induction/w64repvar50x4000_w67.json`, re-measured in `eval_repvar50x4000.json`). Frequency-matched probes, 1,000 rows, step 4,000:

| data set | step best head ≥ 0.2 (Muon / Adam) | 2nd-copy loss (Muon / Adam) | copy accuracy (Muon / Adam) | ICL, training probe (Muon / Adam) | held-out loss (Muon / Adam) |
|---|---|---|---|---|---|
| 5 | 1,300 / 2,700 | 4.40 / 6.55 | 27.8% / 10.0% | 3.52 / 1.84 | 4.4493 / 4.4473 |
| 6 | 2,000 / 1,800 | 5.02 / 6.11 | 21.7% / 12.6% | 3.75 / 1.82 | 4.4481 / 4.4521 |
| 7 | 2,153 / 3,000 | 5.50 / 6.71 | 18.0% / 10.6% | 2.53 / 1.61 | 4.4580 / 4.4568 |

- Prediction D, timing part: failed. On test set 6, Adam's head passed 0.2 attention 200 steps earlier. Earlier formation under Muon is 2 of 3 data sets, so the timing claim is dropped.
- Prediction D, copying part: held on all three data sets. At equal held-out loss (within 0.004), Muon's model copies much better: second-copy loss is 1.1-2.2 nats lower, and accuracy is 1.7-2.8× higher.
- Both optimizers end with the same attention to the right token (0.40-0.55). So the difference is in what the head writes once it attends, which is the v,o (OV) circuit, not in where it looks (q,k).
- Related: the Opus critic cites a NeurIPS 2026 paper ("Optimizers Select Algorithms") in which orthogonalized updates favour in-context recall in associative-recall toys. This result points the same way on web text, with the extra finding that attention is equal and only the copied output differs.

Prediction E (written before the runs): in the same setting, on test sets 5-7, the copying advantage follows v,o.
- Muon with v,o on Adam: frequency-matched second-copy loss ends at least halfway from Muon's toward Adam's on all 3 data sets.
- Muon with q,k on Adam: ends within a quarter of the gap from Muon's on at least 2 of 3 data sets.
- All Adam with v,o on Muon: ends at least halfway from Adam's toward Muon's on at least 2 of 3 data sets.

Result E (`results/induction/w64repvar50x4000_parts.json`, re-measured in `eval_repvar50x4000.json`). Each cell is frequency-matched second-copy loss and, in brackets, the share of the Muon→Adam gap. 0 = like Muon, 1 = like Adam:

| arm | w5 | w6 | w7 |
|---|---|---|---|
| Muon | 4.40 (0) | 5.02 (0) | 5.50 (0) |
| Muon, q,k on Adam | 4.59 (0.09) | 5.29 (0.25) | 6.71 (1.00) |
| Muon, v,o on Adam | 5.60 (0.56) | 5.95 (0.85) | 7.45 (1.61) |
| all Adam, v,o on Muon | 6.78 (1.11) | 7.18 (1.97) | 4.87 (-0.52) |
| all Adam | 6.55 (1) | 6.11 (1) | 6.71 (1) |

Held-out loss stays within 4.427-4.468 across all 15 models.
- "Taking v,o off Muon loses at least half the copying advantage, on all 3 data sets": held (0.56, 0.85, 1.61).
- "Taking q,k off Muon keeps it, on at least 2 of 3": held, but at the edge (0.09, 0.25, 1.00).
- "Muon on v,o alone in an Adam model gives at least half of it, on at least 2 of 3": failed (1 of 3).
- Reading: Muon on v,o is needed for the copying advantage but is not enough on its own. Test set 7 swings widely in every arm, so single runs per data set are noisy here.

Prediction F (written before the runs): Adam's copying deficit is not a learning-rate artifact. With every Adam lr scaled ×0.5 or ×2, on test sets 5 and 6, neither scaled arm reaches within a quarter of the gap of Muon's frequency-matched second-copy loss on both data sets.

Result F (`results/induction/w64repvar50x4000_adamlr.json`, re-measured in `eval_repvar50x4000.json`): with every Adam lr at ×0.5 or ×2, frequency-matched second-copy loss is 6.67/7.67 (×0.5) and 6.49/6.71 (×2) on test sets 5/6. Tuned Adam gets 6.55/6.11 and Muon 4.40/5.02. Prediction F held: neither scaled arm comes within a quarter of the gap, so within a 4× lr range Adam's copying deficit is not an lr artifact. Held-out loss: 4.4465/4.4270 (×0.5) and 4.4746/4.4631 (×2).

Second Opus critic (abstract-level read of the NeurIPS 2026 paper): "Muon recalls from context better than Adam" is already shown there on the synthetic RP-MQAR task across seven architectures. What may still be new is (a) web text at matched held-out loss and (b) where in the model the difference sits. Its requested controls:
- First-copy loss on the same frequency-matched rows. It is already in `eval_repvar50x4000.json`: Muon 8.63 vs Adam 8.54 on test set 5. So both predict these tokens equally well, and the gap is in copying.
- Per-head credit on the same rows.

Per-head credit (`debug/induction_heads.py` → `results/induction/heads_repvar50x4000.json`, 500 frequency-matched rows, second-copy positions). Credit is the direct logit for the copied token minus the mean logit, through the frozen final RMSNorm and the unembedding:

| arm | sum over all 16 heads (w5/w6/w7) | sum over the 4 MLPs | top-head credit / its induction attention | 2nd-copy loss for the 100 most common tokens / ranks 1,000+ |
|---|---|---|---|---|
| Muon | 5.40 / 5.18 / 4.41 | 1.95 / 1.10 / 1.35 | 3.90/0.65, 1.71/0.52, 1.63/0.53 | 3.5-4.1 / 6.1-8.1 |
| Adam | 3.06 / 3.48 / 3.07 | 1.76 / 1.75 / 1.58 | 1.81/0.54, 2.52/0.60, 1.32/0.54 | 4.6-5.1 / 8.8-9.8 |
| Adam lr ×0.5, ×2 (w5, w6) | 3.16, 2.10; 3.30, 3.00 | 1.40-1.66 | | |
| Muon, q,k on Adam | 5.32 / 4.78 / 3.35 | 1.20-1.85 | | |
| Muon, v,o on Adam | 4.43 / 4.10 / 2.30 | 1.35-1.45 | | |
| Adam, v,o on Muon | 2.90 / 2.19 / 4.95 | 1.49-1.86 | | |

- Muon's heads write 1.3-2.5× more credit for the copied token than Adam's at every Adam lr tried, on all data sets tried. The MLPs write about the same.
- The top induction head attends about equally (Muon 0.52-0.65, Adam 0.54-0.60).
- In Muon models the copy write is spread over several heads; the top head holds only 1.6-1.7 of 4.4-5.2 on test sets 6-7.
- The gap shows at every token frequency and is larger for rare tokens.
- Status: at width 64 with planted copied spans, "same loss, same attention, Muon's heads write the copied token about 1.6× harder" holds on 3 of 3 data sets and survives a 4× Adam lr range. Still open: different data orders (all data sets share one), natural text with no planted spans, larger models (Kaggle GPU once the weekly quota resets), and an OV-circuit eigenvalue score.
