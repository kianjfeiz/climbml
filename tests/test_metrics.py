import pytest

from climbml.detect.metrics import iou, load_gt, per_image_pr, yolo_to_xyxy

BOX = (0.0, 0.0, 10.0, 10.0)


def test_iou_identical_and_disjoint():
    assert iou(BOX, BOX) == 1.0
    assert iou(BOX, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_half_overlap():
    assert iou(BOX, (5.0, 0.0, 15.0, 10.0)) == pytest.approx(1 / 3)


def test_iou_of_empty_boxes_is_zero():
    assert iou((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)) == 0.0


def test_yolo_to_xyxy():
    assert yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, 100, 200) == (40.0, 60.0, 60.0, 140.0)


def test_per_image_pr_perfect_match():
    preds = [(0, BOX, 0.9)]
    assert per_image_pr(preds, [(0, BOX)]) == (1.0, 1.0)


def test_per_image_pr_wrong_class_is_not_a_match():
    assert per_image_pr([(1, BOX, 0.9)], [(0, BOX)]) == (0.0, 0.0)


def test_per_image_pr_one_prediction_matches_one_gt_only():
    preds = [(0, BOX, 0.9), (0, BOX, 0.8)]      # duplicate detection
    precision, recall = per_image_pr(preds, [(0, BOX)])
    assert (precision, recall) == (0.5, 1.0)


def test_per_image_pr_empty_image_is_perfect():
    assert per_image_pr([], []) == (1.0, 1.0)


def test_per_image_pr_missed_everything():
    assert per_image_pr([], [(0, BOX)]) == (1.0, 0.0)


def test_load_gt_reads_pixel_corners(tmp_path):
    label = tmp_path / "img.txt"
    label.write_text("0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.5 0.5\n")
    boxes = load_gt(label, 100, 200)
    assert boxes[0] == (0, (40.0, 60.0, 60.0, 140.0))
    assert boxes[1][0] == 1


def test_load_gt_missing_file_is_empty(tmp_path):
    assert load_gt(tmp_path / "nope.txt", 100, 100) == []
