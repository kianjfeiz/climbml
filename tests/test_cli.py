import pytest

from climbml.cli import build_parser


def test_every_command_is_registered():
    parser = build_parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    assert commands == {
        "explore", "repair-labels", "train", "eval", "export", "colors",
        "beta-candidates", "beta-prep", "beta-run", "beta-report",
    }


def test_train_defaults():
    args = build_parser().parse_args(["train"])
    assert (args.epochs, args.patience, args.smoke) == (120, 30, False)
    assert args.device is None            # resolved at run time


def test_eval_rejects_an_unknown_split():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "runs/detect/x", "--split", "train"])


def test_beta_run_variant_is_constrained():
    args = build_parser().parse_args(["beta-run", "--variant", "fast"])
    assert args.variant == "fast"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["beta-run", "--variant", "nonsense"])


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
