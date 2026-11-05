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

## 2026-11-05: Which step shapes each group accepts, prediction before running

Shumaylov et al. (2605.11181) report that on GPT-2 a Muon step with random singular values matches Muon, for the whole model. Their result and our weak-direction test both say the singular directions matter more than their exact sizes. Nobody has split this by group that we could find. `debug/spec_parts.py` keeps Muon's singular directions and changes only their sizes, in one group at a time, each step scaled to Muon's size: gradient's own shape (`p=1`), reversed sizes, random sizes.

Prediction: on value, output and feed-forward, the gradient's own shape loses at least .005 and random and reversed sizes lose less than half of that. On query and key, no shape changes the loss by more than .004.

Result at width 48, 250 steps, data in `results/spec/w48.jsonl`. Base 4.3606, gaps from whole-matrix Muon on worlds 5 to 9, lr factor chosen on worlds 0 and 1.

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

`kaggle/out/map-c`, 9.7 GPU hours, 6.8M weights, base held-out 4.1375. Whole-matrix Muon baseline, gaps on worlds 5 to 9.

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

The cost of moving value and output to Adam roughly doubles with each doubling of width, feed-forward does too, and query and key stay near zero. Muon per head on query and key beats whole-matrix Muon by .0052 and .0038 in all 5 worlds at widths 128 and 256, which is under the .005 win line at 256.

## 2026-11-05: Kaggle map, width 64, 8,000 steps (job map-d)

`kaggle/out/map-d`, 7.25 GPU hours, base held-out 4.3148. Whole-matrix Muon baseline, gaps on worlds 5 to 9.

| Moved to Adam | Gap | Wins of 5 |
| --- | --- | --- |
| Value and output | -.0025 (.0013) | 1 |
| Query and key | -.0033 (.0016) | 1 |
| Gate, up and down | -.0037 (.0015) | 1 |
| Gate alone | +.0032 (.0011) | 4 |
| All seven | -.0041 (.0025) | 0 |
| Muon per head | -.0002 (.0010) | 1 |

At width 64, moving all seven to Adam costs .0433 at 500 steps, .0119 at 2,000 and .0041 at 8,000. The gain shrinks with training length and no group stands out at 8,000 steps.

Kaggle refused the next job on 2026-11-05 with "Maximum weekly GPU quota of 30.00 hours reached", so map-e (the add design at width 128) and width 512 wait for the weekly reset. On the same day they were pushed as private jobs on a second Kaggle account, `jensen404`. The width 128 job failed with no internet (`Could not resolve host: huggingface.co`) because that account has no phone verification.

## 2026-11-05: Each group gets its own step shape, prediction before running

`debug/pick_shape.py` picks a power `p` and lr factor per group on worlds 0 and 1, with `p` from .1, .25, .5, 1 on query and key and from -.25, -.1, .1 on the other two groups, and keeps a group's pick only if it beats the baseline on the tune worlds. Prediction: at width 48 the joint optimizer beats whole-matrix Muon by less than the .005 win line at both 250 and 1,000 steps.

Result at width 48, 250 steps, data in `results/spec/pick_w48.jsonl`. Base 4.3606.

| Group | Picked `p` | lr factor | Gap | Wins of 5 |
| --- | --- | --- | --- | --- |
| Query and key | 1 | .63 | +.0029 (.0013) | 4 |
| Value and output | -.1 | 1 | +.0013 (.0011) | 3 |
| Gate, up and down | .1 | 1 | -.0006 (.0017) | 3 |
| All three together | | | +.0052 (.0008) | 5 |

The prediction failed at this size: the joint optimizer passes the .005 line, though worlds 5 to 9 have now scored many candidates, so it needs other lengths, a larger width and the sealed worlds before it counts. The sealed gate needs Vuk's word and has not been used.

Result at width 48, 1,000 steps, data in `results/spec/pick_w48x1000.jsonl`. Base 4.1323.

| Group | Picked `p` | lr factor | Gap | Wins of 5 |
| --- | --- | --- | --- | --- |
| Query and key | .25 | 2.54 | +.0002 (.0018) | 3 |
| Value and output | -.25 | 1 | +.0014 (.0009) | 3 |
| Gate, up and down | .1 | 1 | +.0002 (.0010) | 3 |
| All three together | | | +.0028 (.0011) | 4 |

The gain falls below .005 at 1,000 steps and the query-and-key pick moves from `p=1` to `p=.25`, so this is a short-training effect and we drop the candidate. No sealed world was used.

## 2026-11-05: Do weak directions keep the weights healthy, prediction before running

`debug/weights.py` trains once per setting on world 0 and reads each group's weight matrices at the end. Stable rank is squared Frobenius norm over squared largest singular value. Prediction at width 48, 1,000 steps: value and output end with a clearly lower stable rank under the gradient's shape than under Muon, at least 20% lower, and query and key differ by under 10%.

Result at width 48, 1,000 steps, world 0, one run per setting, data in `results/spec/weights_w48x1000.json` (250 steps in `weights_w48.json`). Stable rank averaged over layers; a fresh matrix starts near 13 for q, k, v, o and near 20 for gate, up, down.

| Setting | Loss | v | o | q | k | gate | up | down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Muon | 4.1325 | 13.72 | 12.00 | 10.05 | 7.90 | 18.13 | 18.86 | 10.78 |
| Gradient's shape on v, o | 4.1489 | 8.11 | 5.15 | 8.83 | 7.64 | 17.63 | 19.21 | 10.39 |
| Gradient's shape on q, k | 4.1307 | 13.45 | 12.27 | 2.89 | 3.45 | 17.49 | 18.58 | 10.90 |
| Gradient's shape on all | 4.1545 | 8.53 | 5.95 | 3.56 | 3.28 | 14.63 | 15.50 | 3.98 |
| Adam on all | 4.1492 | 8.52 | 6.77 | 6.47 | 5.57 | 10.28 | 12.40 | 4.65 |

The first half of the prediction held: v fell 41% and o fell 57%. The second half failed: q fell 71% and k fell 56%, and the loss did not get worse. Muon does keep the stable rank near its starting value, but query and key lose more rank than value and output at no cost, so low stable rank alone does not explain the cost on value and output. It can still matter only if value and output need rank in a way query and key do not, which this test cannot separate.

## 2026-11-05: When do value and output need Muon, prediction before running

`optimizers/x_muon_switch.py` changes step shape once during training, on v and o only, using each shape's own tuned lr. `debug/switch_parts.py` runs gradient's shape then Muon, and Muon then gradient's shape, switching at 1/4, 1/2 and 3/4 of training, on worlds 5-9 at width 48, 1,000 steps. Gradient's shape throughout costs .0192.

Prediction: the cost comes mostly from late steps, not from lasting damage to the weights. Switching to Muon at 1/2 costs under .006, and switching to the gradient's shape at 1/2 costs over .010. If the first number is over .010, early steps do lasting damage, and the weights explanation comes back.

Result at width 48, 250 steps, worlds 5-9, data in `results/spec/switch_w48.jsonl`. Base 4.3606. Gradient's shape throughout costs .0229 (.0020). Negative means worse than Muon.

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

Vuk's question: for each part of the LLM, either any optimizer works, or a better optimizer exists. `debug/panel_parts.py` changes one part at a time from the tuned baseline to another published optimizer, and tunes that optimizer's learning rate and momentum on worlds 0 and 1. It then scores on worlds 5-9, tune rows, with no sealed world. A frozen control trains the part not at all, which shows the most that part can matter.

- Parts: value and output, query and key, gate up and down, embedding, head, norm scales.
- Optimizers: Muon, Adam, SGD with momentum, Lion, RowNorm, normalized SGD and per-head Muon, where each one applies.

Verdict for each part, fixed now:

- **Any optimizer works:** every tuned optimizer stays within .004 of the baseline.
- **Optimizer matters, baseline is best:** some optimizer loses by over .004 and over 2 errors, and none is a candidate.
- **Candidate for better:** a gain above .005 with the same sign in all 5 worlds.
- **Win:** a candidate at both 250 and 1,000 steps with gain above 7.53 errors, which is the t value for 30 comparisons on 5 worlds. A win still needs larger models and Vuk's word before any sealed test.

Prediction: value and output are "matters, baseline is best", with only Muon-like rules close. Query and key are the part closest to "any optimizer works". Embedding and head matter and keep Adam. Norm scales are "any optimizer works". A candidate may appear at 250 steps on query and key, as Lion did, but no win appears at 1,000 steps.

All parts at width 48 and 250 steps, data in `results/panel/w48.jsonl`, summary in `results/panel/w48_out.json`. Base 4.3606. Each cell is the gap to the baseline, with the error in brackets and wins of 5 worlds after it. Positive beats the baseline. The baseline uses Muon on value, output, query, key and the feed-forward layers, and Adam on embeddings, the head and norms.

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
- Query and key: any tuned optimizer works. Normalized SGD, RowNorm and per-head Muon win 5 of 5 worlds with gains under .005, which the 1,000-step run will check.
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

## Related results

Wang et al., arXiv:2509.26030, add Muon to Adam one group at a time at 160M weights and 10,000 steps with Muon's lr fixed. In their gated table, all Muon scores 3.5125, all Adam 3.8837, query and key only 3.8518, value and output only 3.6874, and value, output and feed-forward with query and key on Adam 3.5312, so Adam on query and key costs them .019.
