# Experiment log

One entry per run, including the failed ones. Newest first.

---

## 2026-07-23 — Beta engine calibration on live API calls

**Goal:** pick the latency, quality and cost operating point using real calls,
and fix the prompt against real failures. Budget $10, spent about $1.35.

**Probe, one wall, four settings.** Default (high) effort fails: it spends the
entire 8k token budget thinking and returns no plan (103 s,
`stop_reason=max_tokens`). `medium`: 44 s, $0.043, valid but placed a foot above
the hands. `low`: 10.7 s, $0.015, a sound 16-move plan. `fast` (thinking off):
15.4 s, and sloppier wording — called a hand moving to a different hold a
"match", tagged a vertical wall "slabby".

**Full comparison,** 11 routes x low/fast/medium, 33 calls, about $1. `low`
matches `fast` on latency and cost and reads better on climbing specifics.
`medium` costs 31-70 s for marginal gains on the quality flags. Picked adaptive
thinking at `low` effort. At prompt v1 every route came back V3 — the model was
hedging to the middle of the grade scale.

**Prompt v2,** validated on 4 routes, all improved or unchanged:

- feet must stay below both hands. Cleared the feet-above-hands flags.
- "match" means both hands on one hold, nothing else.
- commit to a grade from the evidence. Grades then spread (V2 for a jug ladder,
  V4-V5 for the small-hold wall) instead of clustering on V3.
- start-hold suggestion changed from "the two lowest holds" to "the lowest hold
  plus the next lowest within 25% of image width". The old rule picked
  wall-spanning pairs on spray walls, and the model followed them.

**Shipped:** `claude-sonnet-5`, adaptive thinking, effort `low`, max_tokens 8000,
prompt version 2. About 10-20 s per route, up to 35 s on 30+ hold clusters, and
1.5-2.5 cents each at the introductory rate.

## 2026-07-23 — Beta engine: grounding, schema, evaluation harness

**Goal:** turn an isolated route into a move sequence, with no way for the model
to invent holds.

**Grounding is Set-of-Mark.** Route holds are outlined and numbered on the photo,
non-route pixels dimmed, start holds tagged in yellow. The same ids, positions
and sizes go along as JSON. The number is the contract: the schema forces `hold`
to be an integer, the validator forces membership, and one automatic repair
round-trip re-prompts with the exact errors. Tag placement is collision-aware
with leader lines, because on crowded walls chips overlapping their neighbours
made it unclear which hold a number belonged to.

**Schema:** overview, grade (nullable), style, confidence, and moves of
{limb, hold, action, detail, isCrux, confidence}. Structured outputs make the
shape guaranteed, so every remaining failure is a grounding failure.

**Harness:** `beta-prep` builds payloads without spending anything, `beta-run`
generates and scores, `beta-report` compares saved runs. Scoring is geometric
(ends at top, hand descents, same-hand runs, feet above hands, reach outliers)
plus rendered images.

**Clusterer bug found through the harness.** On cool-LED walls, white slopers and
shadowed black holds landed in Blue, producing a 64-hold "blue route" on a spray
wall. Three fixes, each measured against HSV dumps (`climbml colors`) and swept
over all 10 routes:

- half-strength gray-world gains (`sqrt`). Full correction assumes the scene
  averages to gray, which wood-dominant walls do not.
- saturation cutoff scaled by value, 0.16 to 0.28 as value approaches 1.0. A
  bright hold with a faint tint is white plastic under coloured light.
- a dark rule (v < 0.32 and s < 0.40 is black rubber), and the Blue/Purple
  boundary moved from 255 to 240 degrees.

Result on the spray wall: Blue dropped from 64 to 34 holds, and a second cluster
split into 32 Blue and 18 Purple.

## 2026-07-22 — Route isolation: colour clustering

**Goal:** group detected holds into a single route.

**Gap-based hue clustering failed.** Splitting on circular hue gaps works on a
wall with three routes and collapses on a rainbow wall, where hue coverage is
continuous and every cluster chains into one. Replaced with fixed perceptual hue
bins, which cannot chain.

**Sampling matters more than the clustering.** A mean over the bounding box
mostly measures wall. Switched to a median over the interior pixels furthest
from a border-band wall estimate, plus gray-world white balance for warm gym
lighting.

**Prominence.** Picking the largest cluster picks chalky footholds. Added a
mean-area term and a penalty for neutral (white, gray, black) clusters so the
colourful route wins.

Known limit: heavily chalked holds read pale and can join a light cluster.

## 2026-07-21 — Run A: YOLO26n baseline, and corrupted dataset labels

**Goal:** first real model, and the baseline later runs have to beat. Gate: hold
AP50 >= 0.75.

**Config:** `yolo26n.pt` (COCO-pretrained, NMS-free head), 120 epochs, all 120
ran with no early stop, 4.99 h on M3 Pro MPS, imgsz 640, batch 16, seed 0.
Augmentation deltas `flipud=0` and `degrees=10`. Default loss and gains.

**Results** after the label repair below: val mAP50 0.877, mAP50-95 0.685, hold
AP50 0.896, volume AP50 0.859. Test, checked once: mAP50 0.871, mAP50-95 0.672,
hold 0.887, volume 0.855. val and test agree, so the model generalises rather
than fitting the validation split. Per-image at conf 0.25: precision 0.90, recall
0.88. Inference 6.1 ms/image, weights 5.4 MB. Gate passed.

**Corrupted labels in the shipped dataset.** The worst-recall gallery showed
images where detections were correct and recall was near zero. The annotations
had been drawn on a differently-rotated copy of the pixels, an EXIF-orientation
bug in the dataset's pipeline. `climbml repair-labels` detected and repaired 5
validation images (rigid 90 and 180 degree transforms, agreement 0.81-0.94 after
the fix, originals backed up) and flagged 4 pairs misaligned beyond any rigid
transform. Measured hold AP50 had been understated by 13 points: 0.764 to 0.896.
Splits after the repair: 1,232 / 266 / 148.

**Export:** CoreML fp16, 4.8 MB. Parity against PyTorch on 5 dense validation
images: box counts within 2, mean confidence delta under 0.008.

**Next-run ideas:** all 120 epochs ran without early stopping, so there is some
headroom left. Try 150-180. Consider yolo26s and imgsz 960 if small-hold recall
needs a lift on phone photos.

## 2026-07-21 — EDA: dataset composition and class identity

**Goal:** understand the data before training, and identify the undocumented
class ids.

**Results:** 1,650 images, 60,804 boxes. Class 0 is hold (56,723 boxes, median
normalised area 0.0011). Class 1 is volume (4,081 boxes, median 0.0167, 15 times
larger, confirmed on sample grids). Imbalance 13.9:1. Boxes per image: train mean
33.8, test mean 54.3, so test metrics are not directly comparable to train-time
numbers.

**Notes:** the set mixes catalog product shots (one hold on a plain background),
dense real walls up to 272 boxes, and watermarked stock photos with climbers in
them. Mosaic augmentation composites the catalog shots into scene-like images,
which is a reason to leave it on.
