import tensorflow as tf
import pytest

from model_composer.interface import ComposedModelDefinition
from model_composer._tensorflow import TensorflowModelComposer


@pytest.fixture
def composed_model_definition(tmpdir):
    return {
        "name": "ride_share_pricing",
        "spec": [
            {
                "mask": {
                    "name": "is_weekend",
                    "spec": {"is_weekend": {"notin": ["False"]}},
                },
                "model": {
                    "name": "weekend_model",
                    "spec": {
                        "type": "tensorflow",
                        "path": str(tmpdir / "weekend_model.tf"),
                    },
                },
            },
            {
                "mask": {
                    "name": "is_weekday",
                    "spec": {"is_weekend": {"in": ["False"]}},
                },
                "model": {
                    "name": "weekday_model",
                    "spec": {
                        "type": "tensorflow",
                        "path": str(tmpdir / "weekday_model.tf"),
                    },
                },
            },
        ],
    }


@pytest.fixture
def weekend_model(tmpdir):
    # Build the weekend model
    weekend_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), name="distance"),
            tf.keras.layers.Dense(1, name="price"),
        ]
    )

    weekend_model.compile(optimizer="adam", loss="mse")

    weekend_model.fit(
        x={"distance": tf.convert_to_tensor([10, 20], dtype=tf.float32)},
        y=tf.convert_to_tensor([10, 20], dtype=tf.float32),
        epochs=10,
    )
    weekend_model.save(tmpdir / "weekend_model.tf")


@pytest.fixture
def weekday_model(tmpdir):
    # Build the weekday model
    weekday_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), name="distance"),
            tf.keras.layers.Dense(1, name="price"),
        ]
    )

    weekday_model.compile(optimizer="adam", loss="mse")

    weekday_model.fit(
        x={"distance": tf.convert_to_tensor([10, 20], dtype=tf.float32)},
        y=tf.convert_to_tensor([5, 10], dtype=tf.float32),
        epochs=10,
    )

    # Save the models
    weekday_model.save(tmpdir / "weekday_model.tf")


def test_model_composer_can_build_model(
    weekday_model, weekend_model, composed_model_definition
):
    composer = TensorflowModelComposer()
    definition = ComposedModelDefinition.parse_obj(composed_model_definition)
    model = composer.build(definition)
    assert isinstance(model, tf.keras.Model)


def test_composed_model_output_is_correct(
    tmpdir, weekday_model, weekend_model, composed_model_definition
):
    weekday_model = tf.keras.models.load_model(tmpdir / "weekday_model.tf")
    weekend_model = tf.keras.models.load_model(tmpdir / "weekend_model.tf")

    x = {"distance": tf.convert_to_tensor([10, 20], dtype=tf.float32)}
    weekday_model_output = weekday_model.predict(x)
    weekend_model_output = weekend_model.predict(x)

    composer = TensorflowModelComposer()
    definition = ComposedModelDefinition.parse_obj(composed_model_definition)
    model = composer.build(definition)

    x["is_weekend"] = tf.convert_to_tensor(["False", "True"], dtype=tf.string)
    model_output = model.predict(x)
    expected = tf.convert_to_tensor([weekday_model_output[0], weekend_model_output[1]])
    print(f"{model_output=} {expected=}")

    tf.assert_equal(model_output, expected)
