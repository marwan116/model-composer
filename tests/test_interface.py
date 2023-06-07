import pytest

@pytest.fixture
def composed_model_definition():
    return {
        "name": "ride_share_pricing",
        "spec": [
            {
                "mask": {
                    "name": "is_weekend",
                    "spec": {"is_weekend": {"eq": True}},
                },
                "model": {
                    "name": "weekend_model",
                    "spec": {"type": "tensorflow", "path": "weekend_model.tf"},
                },
            },
            {
                "mask": {"name": "is_weekday", "spec": {"is_weekend": {"eq": False}}},
                "model": {
                    "name": "weekday_model",
                    "spec": {"type": "tensorflow", "path": "weekday_model.tf"},
                },
            },
        ],
    }


def test_composed_model_definition_can_be_built(composed_model_definition):
    from model_composer.interface import ComposedModelDefinition

    model_def = ComposedModelDefinition.parse_obj(composed_model_definition)
    assert model_def.name == "ride_share_pricing"
    assert len(model_def.spec) == 2

    assert model_def.spec[0].mask.name == "is_weekend"
    assert model_def.spec[0].mask.spec["is_weekend"]["eq"] == True  # noqa: E712
    assert model_def.spec[0].model.name == "weekend_model"
    assert model_def.spec[0].model.spec.type == "tensorflow"

    assert model_def.spec[1].mask.name == "is_weekday"
    assert model_def.spec[1].mask.spec["is_weekend"]["eq"] == False  # noqa: E712
    assert model_def.spec[1].model.name == "weekday_model"
    assert model_def.spec[1].model.spec.type == "tensorflow"
