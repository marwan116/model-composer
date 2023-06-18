"""Tests for the TensorflowModelComposer class."""
from pathlib import Path
from typing import Any, cast, Dict

import pytest
import tensorflow as tf
import yaml

from model_composer.implementations.tensorflow.composer import TensorflowModelComposer
from model_composer.spec import ModelComposerSpec


@pytest.fixture
def model_composer_spec(tmpdir: Path) -> Dict[str, Any]:
    return {
        "name": "my_composed_model",
        "components": [
            {
                "name": "weekday_model",
                "path": str(tmpdir / "weekday_model.tf"),
                "type": "tensorflow",
                "where": {"input": "is_weekday", "operator": "eq", "value": True},
            },
            {
                "name": "weekend_model",
                "path": str(tmpdir / "weekend_model.tf"),
                "type": "tensorflow",
                "where": {"input": "is_weekday", "operator": "eq", "value": False},
            },
        ],
    }


@pytest.fixture
def model_composer_spec_yaml_path(
    tmpdir: Path, model_composer_spec: ModelComposerSpec
) -> Path:
    with open(tmpdir / "composed_model.yaml", "w") as f:
        yaml.dump(model_composer_spec, f)

    return tmpdir / "composed_model.yaml"


@pytest.fixture(params=["explicit_input", "implicit_input"])
def weekend_model(request: pytest.FixtureRequest, tmpdir: Path) -> tf.keras.Model:
    """Build the weekend model."""
    if request.param == "explicit_input":
        weekend_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1,), name="distance"),
                tf.keras.layers.Dense(1, name="price"),
            ]
        )
    elif request.param == "implicit_input":
        weekend_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(1, input_shape=(1,))]
        )
    else:
        raise ValueError(f"Invalid param: {request.param}")

    weekend_model.compile(optimizer="adam", loss="mse")

    weekend_model.fit(
        x={
            input.name: tf.convert_to_tensor([10, 20], dtype=input.dtype)
            for input in [weekend_model.input]
        },
        y=tf.convert_to_tensor([5, 10], dtype=tf.float32),
        epochs=10,
    )
    weekend_model.save(tmpdir / "weekend_model.tf")

    return weekend_model


@pytest.fixture(params=["explicit_input", "implicit_input"])
def weekday_model(request: pytest.FixtureRequest, tmpdir: Path) -> tf.keras.Model:
    """Build the weekday model."""
    if request.param == "explicit_input":
        weekday_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1,), name="distance"),
                tf.keras.layers.Dense(1, name="price"),
            ]
        )
    elif request.param == "implicit_input":
        weekday_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(1, input_shape=(1,))]
        )
    else:
        raise ValueError(f"Invalid param: {request.param}")

    weekday_model.compile(optimizer="adam", loss="mse")

    weekday_model.fit(
        x={
            input.name: tf.convert_to_tensor([10, 20], dtype=input.dtype)
            for input in [weekday_model.input]
        },
        y=tf.convert_to_tensor([5, 10], dtype=tf.float32),
        epochs=10,
    )

    # Save the models
    weekday_model.save(tmpdir / "weekday_model.tf")

    return weekday_model


def test_model_composer_can_build_model_from_spec(
    weekday_model, weekend_model, model_composer_spec
):
    """Test the model composer can build a tensorflow model from the spec."""
    composer = TensorflowModelComposer()
    spec = ModelComposerSpec.parse_obj(model_composer_spec)
    model = composer.build(spec)
    assert isinstance(model, tf.keras.Model)


def test_model_composer_can_build_model_from_yaml(
    weekday_model, weekend_model, model_composer_spec_yaml_path
):
    """Test the model composer can build a tensorflow model from a spec file."""
    composer = TensorflowModelComposer()
    model = composer.from_yaml(model_composer_spec_yaml_path)
    assert isinstance(model, tf.keras.Model)


def test_composed_model_output_is_correct(
    tmpdir, weekday_model, weekend_model, model_composer_spec
):
    """Test the composed model behavior is as expected."""
    weekday_model = cast(
        tf.keras.Model, tf.keras.models.load_model(tmpdir / "weekday_model.tf")
    )
    weekend_model = cast(
        tf.keras.Model, tf.keras.models.load_model(tmpdir / "weekend_model.tf")
    )

    x = {
        input.name: tf.convert_to_tensor([10, 20], dtype=tf.float32)
        for input in [weekday_model.input, weekend_model.input]
    }
    weekday_model_output = weekday_model.predict(x)
    weekend_model_output = weekend_model.predict(x)

    composer = TensorflowModelComposer()
    spec = ModelComposerSpec.parse_obj(model_composer_spec)
    model = composer.build(spec)

    x["is_weekday"] = tf.convert_to_tensor([True, False], dtype=tf.bool)
    model_output = model.predict(x)
    expected = tf.convert_to_tensor([weekday_model_output[0], weekend_model_output[1]])
    tf.assert_equal(model_output, expected)
