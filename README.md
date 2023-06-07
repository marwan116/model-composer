# Model Composer

## Motivation
Easily compose a model ensemble from your machine learning models

It is common in many usecases to have different models to deal with cyclic and seasonal drifts or more generally different slices of the data.
For instance, rideshare prices will fluctuate on weekdays versus weekends. If we have two models, one trained on weekday data and one trained on weekend data, we would like to compose them into a single model that can be used to predict prices for any day of the week. The composed model can be stored and deployed as a single "object" and used to make predictions. This is favorable to having to store and deploy two separate models for the following reasons:

- The composed model is defined once and therefore easier to maintain than having to implement the logic to compose the models in every service that needs to make predictions.

- The composed model's performance will most likely be better than the performance of the individual models. This is because the composed model will be able to leverage graph optimizations and will only have to process the data once. Refer to our performance section for more details.

## Supported ML frameworks
- Tensorflow

## Roadmap
- Support unnamed input layers in Tensorflow models
- Support for more ML frameworks (Pytorch, Scikit-learn, etc.)

## How to setup

```bash
pip install model_composer
```


## How to use

We define our composed model using a composed model spec which can be serialized in a human-readable file like yaml.

example.yaml
```yaml
name: "ride_share_pricing"
spec:
  - mask:
    name: "is_weekend"
    spec:
      - is_weekend:
        eq: true
    model:
      name: "weekend_model"
        spec:
          type: "tensorflow"
          path: "weekend_model.tf"

  - mask:
    name: "is_weekday"
    spec:
      - is_weekend:
        eq: false
    model:
        name: "weekday_model"
        spec:
          type: "tensorflow"
          path: "weekday_model.tf"
```

We build each model separately and save them to disk. We then load the composed model spec and use it to compose the models.

```python
import tensorflow as tf

# Build the weekend model
weekend_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), name="distance"),
    tf.keras.layers.Dense(1, name="price")
])

weekend_model.compile(optimizer="adam", loss="mse")

weekend_model.fit(
    x={"distance": tf.convert_to_tensor([10, 20], dtype=tf.float32)},
    y=tf.convert_to_tensor([10, 20], dtype=tf.float32),
    epochs=10
)
weekend_model.save("weekend_model.tf")


# Build the weekday model
weekday_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,), name="distance"),
    tf.keras.layers.Dense(1, name="price")
])

weekday_model.compile(optimizer="adam", loss="mse")

weekday_model.fit(
    x={"distance": tf.convert_to_tensor([10, 20], dtype=tf.float32)},
    y=tf.convert_to_tensor([5, 10], dtype=tf.float32),
    epochs=10
)

# Save the models
weekday_model.save("weekday_model.tf")
```

We can then load the composed model spec and use it to make predictions.

```python
from model_composer import ComposedModel

composed_model = ComposedModel.from_file("example.yaml")

# Make predictions
composed_model.predict({"is_weekend": [True, False], "distance": [10, 20]})
```

We can visualize the composed model using the `visualize` method.

```python
composed_model.visualize()
```
