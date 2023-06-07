"""Interface of model-composer."""
from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from model_composer._utils import DictLikeBaseModel, ListLikeBaseModel

InputStr = str


class CompSpec(
    DictLikeBaseModel[Literal["in", "notin", "eq", "ne", "ge", "gt", "le", "lt"], Any]
):
    """Specification of a comparison operator."""


class _MaskSpec(DictLikeBaseModel[InputStr, CompSpec]):
    """A mapping of inputs to their comparison specs."""


class MaskSpec(BaseModel):
    """Specification for a slice - the mask and model."""

    name: str
    spec: _MaskSpec


class _ModelSpec(BaseModel):
    type: Literal["tensorflow", "sklearn"]
    path: str


class ModelSpec(BaseModel):
    """Specification of a model."""

    name: str
    spec: _ModelSpec


class SliceSpec(BaseModel):
    """Specification for a slice."""

    mask: MaskSpec
    model: ModelSpec


class SliceSpecList(ListLikeBaseModel[SliceSpec]):
    """A list of slice specifications."""


class ComposedModelDefinition(BaseModel):
    """Definition of a composed model."""

    name: str
    spec: SliceSpecList


class Model(Protocol):
    """A model protocol - left empty to keep type flexible to different ML libraries."""


class ModelComposerInterface(ABC):
    """Interface of a model composer."""

    @abstractmethod
    def build(self, definition: ComposedModelDefinition) -> Model:
        """Build a model object given a definition."""
