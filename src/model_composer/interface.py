"""Interface of model-composer."""
from abc import ABC, abstractmethod
from typing import Protocol

from model_composer.spec import ModelComposerSpec


class Model(Protocol):
    """A model protocol - left empty to keep type flexible to different ML libraries."""


class ModelComposerInterface(ABC):
    """Interface of a model composer."""

    @abstractmethod
    def build(self, spec: ModelComposerSpec) -> Model:
        """Build a model object given a spec."""

    def from_yaml(self, path: str) -> Model:
        """Build a model object given a path to a YAML file."""
        import yaml

        with open(path, "r") as f:
            spec_dict = yaml.safe_load(f)

        spec = ModelComposerSpec.parse_obj(spec_dict)

        return self.build(spec)
