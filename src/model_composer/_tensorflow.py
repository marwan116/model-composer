"""Build a composed tensorflow model from a definition."""
from typing import List, Optional, TypeVar, Union
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from model_composer.interface import (
    ModelComposerInterface,
    ComposedModelDefinition,
    CompSpec,
)
from keras.engine.keras_tensor import KerasTensor
from typeguard import typechecked

InputLayerT = TypeVar(
    "InputLayerT",
    bound=Union[
        tf.keras.layers.InputLayer,
        tf.Tensor,
        Tensor,
        tf.SparseTensor,
        tf.RaggedTensor,
        KerasTensor,
    ],
)

TensorLikeT = TypeVar(
    "TensorLikeT",
    bound=Union[
        tf.Tensor,
        Tensor,
        tf.SparseTensor,
        tf.RaggedTensor,
        KerasTensor,
    ],
)


@typechecked
class TensorflowModelComposer(ModelComposerInterface):
    """Build a composed tensorflow model from a definition."""

    def _infer_tf_dtype(self, comp_spec: CompSpec) -> tf.dtypes.DType:
        """Infer the tensorflow dtype from a comparison spec."""
        dtypes = set()
        for comp_operator, comp_value in comp_spec.items():
            if comp_operator in ["in", "notin"]:
                elem_value = comp_value[0]
                if isinstance(elem_value, int):
                    dtypes.add(tf.int32)
                elif isinstance(elem_value, float):
                    dtypes.add(tf.float32)
                elif isinstance(elem_value, str):
                    dtypes.add(tf.string)
                else:
                    raise TypeError(f"Unsupported type {type(elem_value)}")

            elif comp_operator in ["ge", "gt", "le", "lt"]:
                if isinstance(comp_value, int):
                    dtypes.add(tf.int32)
                elif isinstance(comp_value, float):
                    dtypes.add(tf.float32)
                elif isinstance(comp_value, str):
                    dtypes.add(tf.string)
                else:
                    raise TypeError(f"Unsupported type {type(comp_value)}")

            elif comp_operator in ["eq", "ne"]:
                if isinstance(comp_value, bool):
                    dtypes.add(tf.bool)
                elif isinstance(comp_value, int):
                    dtypes.add(tf.int32)
                elif isinstance(comp_value, float):
                    dtypes.add(tf.float32)
                elif isinstance(comp_value, str):
                    dtypes.add(tf.string)
                else:
                    raise TypeError(f"Unsupported type {type(comp_value)}")

        if len(dtypes) == 0:
            raise NotImplementedError(f"Cannot infer dtype from {comp_spec=}")
        if len(dtypes) > 1:
            raise ValueError(f"{comp_spec=} requires more than one type: {dtypes=}")

        return dtypes.pop()

    def _attempt_to_fetch_input_from_models(
        self,
        input_name: str,
        inputs: List[InputLayerT],
        expected_dtype: tf.dtypes.DType,
    ) -> Optional[InputLayerT]:
        """Attempt to fetch an input from a list of models."""
        for input_layer in inputs:
            if input_layer.name == input_name:
                if input_layer.dtype != expected_dtype:
                    raise ValueError(
                        f"Expected dtype {expected_dtype} for input {input_name}, "
                        f"got {input_layer.dtype}"
                    )
                return input_layer

    def _fetch_or_build_input(
        self,
        col_name: str,
        comp_spec: CompSpec,
        inputs: List[InputLayerT],
    ) -> InputLayerT:
        """Fetch or build an input layer for a given column."""
        expected_dtype = self._infer_tf_dtype(comp_spec)

        found_layer = self._attempt_to_fetch_input_from_models(
            input_name=col_name, inputs=inputs, expected_dtype=expected_dtype
        )

        if found_layer is not None:
            return found_layer

        return tf.keras.layers.Input(shape=(1,), name=col_name, dtype=expected_dtype)

    def _get_model_inputs(self, model: tf.keras.Model) -> List[InputLayerT]:
        """Get the inputs for a model."""
        if isinstance(model.input, list):
            return model.input

        elif isinstance(
            model.input,
            (
                tf.keras.layers.InputLayer,
                tf.Tensor,
                tf.SparseTensor,
                tf.RaggedTensor,
                KerasTensor,
            ),
        ):
            return [model.input]

        else:
            raise NotImplementedError(
                f"Unsupported model input type {type(model.input)}"
            )

    def _resolve_inputs(self, models: List[tf.keras.Model]) -> List[InputLayerT]:
        """Resolve the inputs for a list of models."""
        inputs = []
        for model in models:
            model_inputs = self._get_model_inputs(model)
            for model_input in model_inputs:
                if model_input.name not in {i.name for i in inputs}:
                    inputs.append(model_input)
        return inputs

    def _build_condition(self, comp_spec: CompSpec, input: InputLayerT) -> TensorLikeT:
        for comp_op, comp_val in comp_spec.items():
            if comp_op == "eq":
                return tf.math.equal(input, comp_val)
            elif comp_op == "ne":
                return tf.math.not_equal(input, comp_val)
            elif comp_op == "gt":
                return tf.math.greater(input, comp_val)
            elif comp_op == "ge":
                return tf.math.greater_equal(input, comp_val)
            elif comp_op == "lt":
                return tf.math.less(input, comp_val)
            elif comp_op == "le":
                return tf.math.less_equal(input, comp_val)
            elif comp_op == "in":
                return tf.reshape(
                    tf.reduce_any(
                        tf.concat(
                            values=[
                                tf.math.equal(input, ind_val) for ind_val in comp_val
                            ],
                            axis=-1,
                        ),
                        axis=-1,
                    ),
                    (-1, 1),
                )
            elif comp_op == "notin":
                return tf.reshape(
                    tf.reduce_all(
                        tf.concat(
                            values=[
                                tf.math.not_equal(input, ind_val)
                                for ind_val in comp_val
                            ],
                            axis=-1,
                        ),
                        axis=-1,
                    ),
                    (-1, 1),
                )

    def build(self, definition: ComposedModelDefinition) -> "tf.keras.Model":
        """Create a composed model from a definition."""
        mask_and_model = []
        for slice_spec in definition.spec:
            model_spec = slice_spec.model.spec
            mask_spec = slice_spec.mask.spec

            if model_spec.type != "tensorflow":
                raise NotImplementedError(
                    f"Only tensorflow models are supported, got {model_spec.type}"
                )

            model = tf.keras.models.load_model(model_spec.path, compile=False)
            mask_and_model.append((mask_spec, model))

        inputs = self._resolve_inputs(models=[model for _, model in mask_and_model])
        model_and_condition_inputs = []
        for mask_spec, model in mask_and_model:
            model_conditions = []
            for col_name, comp_spec in mask_spec.items():
                condition_input = self._fetch_or_build_input(
                    col_name=col_name,
                    comp_spec=comp_spec,
                    inputs=inputs,
                )

                if condition_input.name not in {i.name for i in inputs}:
                    inputs.append(condition_input)

                condition = self._build_condition(
                    comp_spec=comp_spec, input=condition_input
                )
                model_conditions.append(condition)

            model_condition = tf.reshape(
                tf.reduce_any(tf.concat(values=model_conditions, axis=-1), axis=-1),
                (-1, 1),
            )
            # model_condition = model_conditions[0]
            model_and_condition_inputs.append((model, model_condition))

        # create an index tensor to keep track of the order of the inputs
        index_tensor = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(tf.range(tf.shape(x)[0]), axis=-1),
        )(inputs[0])

        # build out the model composed now that the conditions are set up
        outputs = []
        index_masked = []

        for model, condition in model_and_condition_inputs:
            model_inputs = self._get_model_inputs(model)
            model_inputs_reconstructed = [
                i for i in inputs if i.name in {j.name for j in model_inputs}
            ]
            input_sliced = tf.boolean_mask(model_inputs_reconstructed, condition)
            index_masked.append(tf.boolean_mask(index_tensor, condition))
            output = model(input_sliced)
            outputs.append(output)

        index_after_mask = tf.keras.layers.concatenate(index_masked, axis=0)
        composed_output = tf.keras.layers.concatenate(outputs, axis=0)
        composed_output_reordered = tf.gather(
            composed_output, tf.argsort(index_after_mask)
        )
        composed_model = tf.keras.models.Model(
            inputs=inputs, outputs=composed_output_reordered
        )

        return composed_model
