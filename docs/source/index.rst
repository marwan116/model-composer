Model Composer
==============

Motivation
----------

This use-case prompted the development of ``model-composer``:

-  You have two tensorflow models, one trained on weekday data and one
   trained on weekend data
-  You would like to compose a single tensorflow model that can be used
   to generate predictions for any day of the week.
-  You want the composed model to be natively defined in tensorflow -
   i.e. a single “computational graph” that can be easily loaded and
   used to make predictions.
-  You want a single composed model becasuse:

   -  It is easier to maintain than having to implement the logic to
      compose the models in every service that needs to make
      predictions.
   -  It ensures the performance of the composed model will remain
      consistent with a native tensorflow model of a similar complexity.
   -  It is easier to deploy a single model than multiple models

Documentation
-------------

The official documentation is hosted on ReadTheDocs:
https://model-composer.readthedocs.io

Install
-------

Using pip:

::

   pip install model-composer

Extras
~~~~~~

Make use of extras to install the model composer implementations that
you need:

.. code:: bash

   pip install model-composer[tensorflow]  # compose tensorflow models
   pip install model-composer[cloudpathlib]  # load models from cloud storage
   pip install model-composer[all]  # all extras

Quick start
-----------

Declare your composed model in a yaml file which defines the components
and how they should be composed.

``yaml title="example.yaml" name: "ride_share_pricing" components:   - name: weekday_model     path: weekday_model.tf     type: tensorflow     where:       input: is_weekday       operator: eq       value: true   - name: weekend_model     path: weekend_model.tf     type: tensorflow     where:       input: is_weekday       operator: eq       value: false``

Each component needs to have the following properties: - ``name``: The
name of the component model - ``path``: The path to the component model
on disk - ``type``: The type of the component model. - ``where``: The
condition at which the component model should be used.

We build the weekend model and save it to disk.

.. code:: python

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

We build the weekday model and save it to disk.

.. code:: python

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

We can now build our composed model from the example yaml spec.

.. code:: python

   import tensorflow as tf
   from model_composer import TensorflowModelComposer

   composed_model = TensorflowModelComposer().from_yaml("example.yaml")

   assert isinstance(composed_model, tf.keras.Model)

   composed_model.save("composed_model.tf")

   loaded_model = tf.keras.models.load_model("composed_model.tf")

   composed_model.predict({
     "is_weekday": tf.convert_to_tensor([True, False], dtype=tf.bool),
     "distance": tf.convert_to_tensor([10, 20], dtype=tf.float32)
   })

Roadmap
-------

-  Support for more ML frameworks:

   -  PyTorch
   -  Scikit-learn


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   model_composer.implementations
   model_composer.implementations.tensorflow


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
