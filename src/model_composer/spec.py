"""Specification of a model-composer."""
from __future__ import annotations

from typing import Dict, List, Literal, Union

from pydantic import BaseModel, Field

InputNameStr = str
ModelNameStr = str
JSONPrimitiveT = Union[None, int, float, str, bool]
JSONSerializableValueT = Union[
    JSONPrimitiveT, List[JSONPrimitiveT], Dict[str, JSONPrimitiveT]
]
BinaryOpT = Literal["_and", "_or"]

__all__ = ["ModelComposerSpec"]

ConditionSpecT = Union[
    "ComparisonSpec", "UnionConditionSpec", "IntersectionConditionSpec"
]


class ComparisonSpec(BaseModel):
    """A comparison between an input and a value."""

    input: InputNameStr
    operator: Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "notin"]
    value: JSONSerializableValueT


class UnionConditionSpec(BaseModel):
    """A list of conditions that are OR'ed together."""

    or_: List[Union[ComparisonSpec, "IntersectionConditionSpec"]]


class IntersectionConditionSpec(BaseModel):
    """A list of conditions that are AND'ed together."""

    and_: List[Union[ComparisonSpec, UnionConditionSpec]]


UnionConditionSpec.update_forward_refs()


class ComponentModelSpec(BaseModel):
    """A list of slice specifications."""

    name: ModelNameStr
    path: str
    type: Literal["tensorflow", "sklearn"]
    where: ConditionSpecT = Field(repr=False)


class ModelComposerSpec(BaseModel):
    """Model composer specification.

    example:
        >>> spec = ModelComposerSpec.parse_obj({
        ...     "name": "my_composed_model",
        ...     "components": [
        ...         {
        ...             "name": "is_weekend_model",
        ...             "path": "models/is_weekend_model.tf",
        ...             "type": "tensorflow",
        ...             "where": {
        ...                 "input": "is_weekday",
        ...                 "operator": "eq",
        ...                 "value": False
        ...             },
        ...         },
        ...         {
        ...             "name": "is_weekday_model",
        ...             "path": "models/is_weekday_model.tf",
        ...             "type": "tensorflow",
        ...             "where": {
        ...                 "input": "is_weekday",
        ...                 "operator": "eq",
        ...                 "value": True
        ...             },
        ...         },
        ...     ]
        ... })
    """

    name: str
    components: List[ComponentModelSpec]
