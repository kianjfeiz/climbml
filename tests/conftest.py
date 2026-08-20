"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthetic import build_wall

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def wall_photo():
    """A synthetic wall: three blue holds up the middle, three red off to the side."""
    return build_wall()


@pytest.fixture
def example_route() -> dict:
    """A real isolated route (holds + starts) captured from the pipeline."""
    return json.loads((FIXTURES / "example_route.json").read_text())


@pytest.fixture
def example_plan() -> dict:
    """A real generated beta for `example_route`."""
    return json.loads((FIXTURES / "example_plan.json").read_text())
