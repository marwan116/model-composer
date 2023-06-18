"""Test the model_composer spec."""
import pytest

from model_composer.spec import ModelComposerSpec


@pytest.fixture
def simple_model_composer_spec():
    """Return a simple model composer spec."""
    return {
        "name": "my_composed_model",
        "components": [
            {
                "name": "is_weekday_model",
                "path": "models/is_weekday_model.tf",
                "type": "tensorflow",
                "where": {"input": "is_weekday", "operator": "eq", "value": True},
            },
            {
                "name": "is_weekend_model",
                "path": "models/is_weekend_model.tf",
                "type": "tensorflow",
                "where": {"input": "is_weekday", "operator": "eq", "value": False},
            },
        ],
    }


@pytest.fixture
def complex_model_composer_spec():
    """Return a complex model composer spec."""
    return {
        "name": "my_composed_model",
        "components": [
            {
                "name": "is_weekday_model",
                "path": "models/is_weekday_model.tf",
                "type": "tensorflow",
                "where": {
                    "and_": [
                        {
                            "or_": [
                                {"input": "is_monday", "operator": "eq", "value": True},
                                {
                                    "input": "is_tuesday",
                                    "operator": "eq",
                                    "value": True,
                                },
                                {
                                    "input": "is_wednesday",
                                    "operator": "eq",
                                    "value": True,
                                },
                                {
                                    "input": "is_thursday",
                                    "operator": "eq",
                                    "value": True,
                                },
                                {"input": "is_friday", "operator": "eq", "value": True},
                            ]
                        },
                        {"input": "is_holiday", "operator": "eq", "value": False},
                    ],
                },
            },
            {
                "name": "is_weekend_model",
                "path": "models/is_weekend_model.tf",
                "type": "tensorflow",
                "where": {
                    "input": "is_weekday",
                    "operator": "eq",
                    "value": False,
                },
            },
            {
                "name": "is_holiday_model",
                "path": "models/is_holiday_model.tf",
                "type": "tensorflow",
                "where": {
                    "and_": [
                        {"input": "is_holiday", "operator": "eq", "value": True},
                        {
                            "or_": [
                                {"input": "is_monday", "operator": "eq", "value": True},
                                {
                                    "input": "is_tuesday",
                                    "operator": "eq",
                                    "value": True,
                                },
                                {
                                    "input": "is_wednesday",
                                    "operator": "eq",
                                    "value": True,
                                },
                                {
                                    "input": "is_thursday",
                                    "operator": "eq",
                                    "value": True,
                                },
                                {"input": "is_friday", "operator": "eq", "value": True},
                            ]
                        },
                    ]
                },
            },
        ],
    }


def test_can_parse_a_simple_composed_model_spec(simple_model_composer_spec):
    """Test that we can parse a simple composed model spec."""
    model_def = ModelComposerSpec.parse_obj(simple_model_composer_spec)
    model_def.name == "my_composed_model"
    model_def.components[0].name == "is_weekday_model"
    model_def.components[0].path == "models/is_weekday_model.tf"
    model_def.components[0].type == "tensorflow"

    model_def.components[1].name == "is_weekend_model"
    model_def.components[1].path == "models/is_weekend_model.tf"
    model_def.components[1].type == "tensorflow"


def test_can_parse_a_complex_composed_model_spec(complex_model_composer_spec):
    """Test that we can parse a complex composed model spec."""
    ModelComposerSpec.parse_obj(complex_model_composer_spec)
