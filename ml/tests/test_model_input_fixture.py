"""The shared model-input fixture is current: it equals what `features.py` produces today."""

import json

from signlnk_ml.training.golden_fixtures import FIXTURE, build


def test_fixture_matches_the_python_reference() -> None:
    assert json.loads(FIXTURE.read_text()) == json.loads(json.dumps(build(), allow_nan=False)), (
        "stale: run `uv run python -m signlnk_ml.training.golden_fixtures`"
    )


def test_fixture_covers_both_coordinate_modes_and_the_resample_cases() -> None:
    cases = json.loads(FIXTURE.read_text())
    assert {c["name"] for c in cases["make_input"]} == {
        "random_clip_xy",
        "random_clip_xyz",
        "edge_cases_xy",
        "edge_cases_xyz",
    }
    assert [c["name"] for c in cases["resample"]] == ["8_to_5", "8_to_8", "3_to_64", "8_to_64"]
