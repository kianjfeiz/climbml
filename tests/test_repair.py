"""Label-repair tests.

The corrupted case is built by rotating known-good labels, which is exactly the
defect found in the shipped dataset (docs/experiments.md).
"""

from climbml.data.repair import (
    CONFIDENT,
    TRANSFORMS,
    agreement,
    diagnose,
    read_rows,
    write_rows,
)
from climbml.detect.metrics import yolo_to_xyxy

W = H = 100.0
# Asymmetric on purpose. A box that maps onto another under a rotation would
# make a corrupted file look partly healthy.
GT = [(0, 0.15, 0.25, 0.06, 0.06), (0, 0.35, 0.15, 0.06, 0.06),
      (0, 0.70, 0.35, 0.06, 0.06), (0, 0.55, 0.80, 0.06, 0.06),
      (0, 0.25, 0.90, 0.06, 0.06), (0, 0.85, 0.65, 0.06, 0.06)]
PREDS = [yolo_to_xyxy(cx, cy, w, h, W, H) for _, cx, cy, w, h in GT]


def _rotate(rows, name):
    transform = TRANSFORMS[name]
    return [(cls, *transform(cx, cy, w, h)) for cls, cx, cy, w, h in rows]


def test_healthy_labels_agree_with_predictions():
    assert agreement(GT, PREDS, W, H) == 1.0


def test_healthy_labels_are_not_flagged():
    assert diagnose(GT, PREDS, W, H) is None


def test_rotated_labels_are_diagnosed_and_repairable():
    for name in TRANSFORMS:
        corrupted = _rotate(GT, name)
        finding = diagnose(corrupted, PREDS, W, H)
        assert finding is not None, name
        assert finding.fixable and finding.repaired_agreement >= CONFIDENT
        # The repair is the inverse rotation, so the result is the original.
        repaired = _rotate(corrupted, finding.transform)
        assert agreement(repaired, PREDS, W, H) == 1.0


def test_scrambled_labels_are_reported_but_not_repaired():
    scrambled = [(0, 0.05 + 0.03 * i, 0.05, 0.02, 0.02) for i in range(len(GT))]
    finding = diagnose(scrambled, PREDS, W, H)
    assert finding is not None and not finding.fixable


def test_rows_round_trip(tmp_path):
    label = tmp_path / "img.txt"
    write_rows(label, GT)
    assert read_rows(label) == GT


def test_read_rows_skips_malformed_lines(tmp_path):
    label = tmp_path / "img.txt"
    label.write_text("0 0.1 0.1 0.1 0.1\ngarbage\n\n1 0.2 0.2 0.2 0.2\n")
    assert len(read_rows(label)) == 2
