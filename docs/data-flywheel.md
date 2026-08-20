# Retraining from user corrections

A correction interface is also a labelling tool. When someone adds a hold the
model missed, or removes a box that is not a hold, they produce the supervision
this model lacks: their wall, their lighting, their phone camera. That is the
distribution the Roboflow dataset covers least.

This is a design note, not shipped code. It sets out what would have to be true
for those corrections to be worth training on.

## Signals

| Signal | Produced when | Becomes |
|---|---|---|
| Added hold (a tap where no box was) | route refinement | false-negative label |
| Removed box (a detection dismissed) | route refinement | false-positive flag |
| Untouched detections on a refined route | analysis completed | weak positive confirmation |
| Beta rated up or down | after generation | prompt eval data, separate track |

Image pixels are collected only under explicit opt-in, off by default. Wall
photos contain bystanders, so they are personal data: strip EXIF GPS on upload,
delete on account deletion.

## Record shape

One record per refined route:

```json
{
  "model_version": "y26n-640-v12-a",
  "prompt_version": 2,
  "image_ref": "storage path, absent unless the user opted in",
  "detections": [{"cls": 0, "conf": 0.62, "cx": 0.41, "cy": 0.77, "w": 0.05, "h": 0.04}],
  "corrections": [{"action": "add", "cls": 0, "cx": 0.55, "cy": 0.31, "w": 0.06, "h": 0.05}],
  "device": "iPhone16,1",
  "created_at": "2026-07-23T09:14:00Z"
}
```

The model version travels with every record, so cohorts stay comparable and a
regression can be traced to the model generation that caused it.

## The loop

Manual and infrequent, roughly quarterly. Never automatic.

1. **Export** consented records to images and YOLO labels:
   `detections + additions - removals`.
2. **Review.** A human pass in a labelling tool is required. Users are noisy
   labellers: "the holds I used" is not "every hold in the frame", and refining a
   route is not an exhaustive annotation. Reject images that are internally
   inconsistent rather than repairing them.
3. **Version.** The reviewed set becomes `data/v2/`. v1 (Roboflow v12) stays
   frozen. Check for near-duplicates against train and val/test, since user
   photos of the same wall on different days are the obvious leakage path.
4. **Retrain** with the same recipe under a new run name. Log it in
   [experiments.md](experiments.md).
5. **Gate.** The new model must beat the current one on the frozen v1 test split
   and on a curated set of real user photos. Both, not either.
6. **Ship** the weights, and record the new model version in later records.

## Guardrails

- **The v1 test split is never touched.** No user data merges into it. It is the
  only constant yardstick across model generations, and it is worth more frozen
  than it would be as extra training data.
- **Watch class balance.** Corrections skew heavily toward holds, since nobody
  corrects a volume. Cap the drift in the hold:volume ratio per version.
- **Poisoning.** Human review is the only real filter, which is why the loop is
  not automatic.
- **Minimum batch.** Do not retrain below about 300 reviewed images. Under that
  the measured difference is noise.
