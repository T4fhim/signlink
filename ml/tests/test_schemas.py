"""Contract tests: each schema accepts its example, in JSON Schema and in its Pydantic model."""

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "packages" / "schemas"
SCHEMA_FILES = sorted(p.name for p in SCHEMAS_DIR.glob("*.v[0-9]*.json"))


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _example(schema_file: str) -> dict[str, Any]:
    return _load(SCHEMAS_DIR / "examples" / schema_file.replace(".json", ".example.json"))


def _validator(schema_file: str) -> Draft202012Validator:
    schema = _load(SCHEMAS_DIR / schema_file)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def test_all_five_schemas_present() -> None:
    assert SCHEMA_FILES == [
        "landmark_layout.v1.json",
        "lexicon_entry.v1.json",
        "recognition_output.v1.json",
        "sign_output_request.v1.json",
        "ws_message.v1.json",
    ]


def test_committed_layouts_validate_against_layout_schema() -> None:
    validator = _validator("landmark_layout.v1.json")
    layouts = sorted((SCHEMAS_DIR / "layouts").glob("*.json"))
    assert [p.name for p in layouts] == ["slk-landmarks-v1.json"]
    for path in layouts:
        assert list(validator.iter_errors(_load(path))) == [], path.name


@pytest.mark.parametrize("schema_file", SCHEMA_FILES)
def test_example_validates(schema_file: str) -> None:
    errors = list(_validator(schema_file).iter_errors(_example(schema_file)))
    assert errors == []


@pytest.mark.parametrize("schema_file", SCHEMA_FILES)
def test_missing_required_field_is_rejected(schema_file: str) -> None:
    validator = _validator(schema_file)
    schema = _load(SCHEMAS_DIR / schema_file)
    for key in schema["required"]:
        example = {k: v for k, v in _example(schema_file).items() if k != key}
        assert not validator.is_valid(example), f"{schema_file} accepted example without {key}"


@pytest.mark.parametrize("schema_file", SCHEMA_FILES)
def test_added_fields_are_tolerated(schema_file: str) -> None:
    assert _validator(schema_file).is_valid({**_example(schema_file), "future_field": 1})


@pytest.mark.parametrize("schema_file", SCHEMA_FILES)
def test_generated_pydantic_model_parses_example(schema_file: str) -> None:
    module_name = schema_file.removesuffix(".json").replace(".", "_")
    module = importlib.import_module(f"signlnk_ml.schemas_generated.{module_name}")
    model = getattr(module, _load(SCHEMAS_DIR / schema_file)["title"])
    parsed = model.model_validate(_example(schema_file))
    assert parsed.model_dump(mode="json", exclude_none=True)
    assert model.model_validate({**_example(schema_file), "future_field": 1}) is not None
