"""Build a composed tensorflow model from a spec."""
import re
import tempfile
from pathlib import Path
from typing import cast, List, Optional, TYPE_CHECKING, Union

from typeguard import typechecked

from model_composer._utils import copy_path, make_path
from model_composer.interface import ModelComposerInterface
from model_composer.spec import (
    ComparisonSpec,
    ConditionSpecT,
    IntersectionConditionSpec,
    ModelComposerSpec,
    UnionConditionSpec,
)

if TYPE_CHECKING:
    import tensorflow as tf
    from keras.engine.keras_tensor import KerasTensor
    from tensorflow.python.framework.ops import Tensor


def _lazy_import_tensorflow() -> None:
    global tf
    global KerasTensor
    global Tensor

    import tensorflow as tf
    from keras.engine.keras_tensor import KerasTensor
    from tensorflow.python.framework.ops import Tensor


InputLayerT = Union[
    "tf.keras.layers.InputLayer",
    "KerasTensor",
]

TensorLikeT = Union[
    "tf.Tensor",
    "Tensor",
    "tf.SparseTensor",
    "tf.RaggedTensor",
    "KerasTensor",
]


@typechecked
class TensorflowModelComposer(ModelComposerInterface):
    """Build a composed tensorflow model from a spec."""

    def _sanitize_layer_name(self, name: str) -> str:
        """Sanitize a layer name."""
        # lowercase, replace spaces with underscores
        # use regex to remove non-alphanumeric characters
        return re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_"))

    def _infer_tf_dtype(self, comparison_spec: ComparisonSpec) -> "tf.dtypes.DType":
        """Infer the tensorflow dtype from a comparison spec."""
        dtypes = set()

        comp_operator = comparison_spec.operator
        comp_value = comparison_spec.value

        if comp_operator in ["in", "notin"]:
            if not isinstance(comp_value, list):
                raise TypeError(f"Expected list for {comparison_spec=}")
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
            raise NotImplementedError(f"Cannot infer dtype from {comparison_spec=}")
        if len(dtypes) > 1:
            raise ValueError(
                f"{comparison_spec=} requires more than one type: {dtypes=}"
            )

        return dtypes.pop()

    def _attempt_to_fetch_input_from_models(
        self,
        input_name: str,
        inputs: List[InputLayerT],
        expected_dtype: "tf.dtypes.DType",
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
        return None

    def _fetch_or_build_input(
        self,
        comparison_spec: ComparisonSpec,
        inputs: List[InputLayerT],
    ) -> InputLayerT:
        """Fetch or build an input layer for a given column."""
        expected_dtype = self._infer_tf_dtype(comparison_spec)

        found_layer = self._attempt_to_fetch_input_from_models(
            input_name=comparison_spec.input,
            inputs=inputs,
            expected_dtype=expected_dtype,
        )

        if found_layer is not None:
            return found_layer

        return cast(
            InputLayerT,
            tf.keras.layers.Input(
                shape=(1,), name=comparison_spec.input, dtype=expected_dtype
            ),
        )

    def _get_model_inputs(self, model: "tf.keras.Model") -> List[InputLayerT]:
        """Get the inputs for a model."""
        if isinstance(model.input, list):
            return model.input

        elif isinstance(
            model.input,
            (
                tf.keras.layers.InputLayer,
                KerasTensor,
            ),
        ):
            return [model.input]

        else:
            raise NotImplementedError(
                f"Unsupported model input type {type(model.input)}"
            )

    def _resolve_inputs(self, models: List["tf.keras.Model"]) -> List[InputLayerT]:
        """Resolve the inputs for a list of models."""
        inputs: List[InputLayerT] = []
        for model in models:
            model_inputs: List[InputLayerT] = self._get_model_inputs(model)
            for model_input in model_inputs:
                if model_input.name not in {i.name for i in inputs}:
                    inputs.append(model_input)
        return inputs

    def _build_condition(
        self, comparison_spec: ComparisonSpec, input: InputLayerT
    ) -> TensorLikeT:
        comp_op = comparison_spec.operator
        comp_val = comparison_spec.value

        if comp_op == "eq":
            comparison_op = tf.math.equal(input, comp_val)
        elif comp_op == "ne":
            comparison_op = tf.math.not_equal(input, comp_val)
        elif comp_op == "gt":
            comparison_op = tf.math.greater(input, comp_val)
        elif comp_op == "ge":
            comparison_op = tf.math.greater_equal(input, comp_val)
        elif comp_op == "lt":
            comparison_op = tf.math.less(input, comp_val)
        elif comp_op == "le":
            comparison_op = tf.math.less_equal(input, comp_val)
        elif comp_op == "in":
            if not isinstance(comp_val, list):
                raise TypeError(f"Expected list for {comparison_spec=}")
            comparison_op = tf.reshape(
                tf.reduce_any(
                    tf.concat(
                        values=[tf.math.equal(input, ind_val) for ind_val in comp_val],
                        axis=-1,
                    ),
                    axis=-1,
                ),
                (-1, 1),
            )
        elif comp_op == "notin":
            if not isinstance(comp_val, list):
                raise TypeError(f"Expected list for {comparison_spec=}")
            comparison_op = tf.reshape(
                tf.reduce_all(
                    tf.concat(
                        values=[
                            tf.math.not_equal(input, ind_val) for ind_val in comp_val
                        ],
                        axis=-1,
                    ),
                    axis=-1,
                ),
                (-1, 1),
            )
        else:
            raise NotImplementedError(f"Unsupported operator {comp_op}")

        return cast(TensorLikeT, comparison_op)

    def _build_condition_from_spec(
        self, condition_spec: ConditionSpecT, inputs: List[InputLayerT]
    ) -> TensorLikeT:
        if isinstance(condition_spec, ComparisonSpec):
            condition_input = self._fetch_or_build_input(
                comparison_spec=condition_spec,
                inputs=inputs,
            )

            if condition_input.name not in {i.name for i in inputs}:
                inputs.append(condition_input)

            return self._build_condition(
                comparison_spec=condition_spec,
                input=condition_input,
            )

        elif isinstance(condition_spec, UnionConditionSpec):
            conditions = []
            for sub_condition_spec in condition_spec.or_:
                conditions.append(
                    self._build_condition_from_spec(
                        condition_spec=sub_condition_spec,
                        inputs=inputs,
                    )
                )

            return tf.reshape(
                tf.reduce_any(tf.concat(values=conditions, axis=-1), axis=-1),
                (-1, 1),
            )

        elif isinstance(condition_spec, IntersectionConditionSpec):
            conditions = []
            for sub_condition_spec in condition_spec.and_:  # type: ignore
                conditions.append(
                    self._build_condition_from_spec(
                        condition_spec=sub_condition_spec,
                        inputs=inputs,
                    )
                )
            return tf.reshape(
                tf.reduce_all(tf.concat(values=conditions, axis=-1), axis=-1),
                (-1, 1),
            )

    def build(self, spec: ModelComposerSpec) -> "tf.keras.Model":
        """Create a composed model from a spec."""
        _lazy_import_tensorflow()

        spec_and_model = []
        for component_spec in spec.components:
            if component_spec.type != "tensorflow":
                raise NotImplementedError(
                    f"Only tensorflow models are supported, got {component_spec.type}"
                )

            model_path_object = make_path(component_spec.path)

            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = Path(tmpdir) / model_path_object.name
                copy_path(model_path_object, str(local_path))
                model = tf.keras.models.load_model(local_path, compile=False)

            spec_and_model.append((component_spec, model))

        inputs = self._resolve_inputs(models=[model for _, model in spec_and_model])

        model_and_condition_inputs = []
        for component_spec, model in spec_and_model:
            model_condition = self._build_condition_from_spec(
                condition_spec=component_spec.where,
                inputs=inputs,
            )

            model_and_condition_inputs.append(
                (model, model_condition, component_spec.name)
            )

        # create an index tensor to keep track of the order of the inputs
        index_tensor = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(tf.range(tf.shape(x)[0]), axis=-1),
        )(inputs[0])

        # build out the model composed now that the conditions are set up
        outputs = []
        index_masked = []

        for model, condition, mask_name in model_and_condition_inputs:
            model_inputs = self._get_model_inputs(model)
            model_inputs_reconstructed = [
                i for i in inputs if i.name in {j.name for j in model_inputs}
            ]

            input_sliced = []
            for input in model_inputs_reconstructed:
                input_sliced.append(tf.boolean_mask(input, condition))

            index_masked.append(tf.boolean_mask(index_tensor, condition))
            model._name = self._sanitize_layer_name(mask_name)
            output = model(input_sliced)
            outputs.append(output)

        index_after_mask = tf.keras.layers.concatenate(index_masked, axis=0)
        composed_output = tf.keras.layers.concatenate(outputs, axis=0)
        composed_output_reordered = tf.gather(
            composed_output, tf.argsort(index_after_mask)
        )
        composed_model = tf.keras.models.Model(
            inputs=inputs, outputs=composed_output_reordered, name=spec.name
        )

        return composed_model
