"""Implements a strict base model that forbids extra fields."""
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    MutableSequence,
    overload,
    SupportsIndex,
    TypeVar,
    Union,
)

from pydantic import BaseConfig, BaseModel as _BaseModel, Extra
from pydantic.generics import GenericModel as _GenericModel

BaseT = TypeVar("BaseT", bound="BaseModel")


class BaseModel(_BaseModel):
    """BaseModel class."""

    # Note: Config has to be explicitly defined here, otherwise the mypy plugin won't
    # work
    class Config(BaseConfig):
        """Default Config for pydantic dataclass and base models."""

        arbitrary_types_allowed: bool = True
        extra: Extra = Extra.forbid
        keep_untouched = (cached_property,)

    def evolve(self: BaseT, **kwargs: Any) -> BaseT:
        """Drop-in replacement for attrs.evolve."""
        return self.copy(update=kwargs, deep=True)


class BaseFrozenModel(BaseModel):
    """BaseFrozenModel class."""

    class Config(BaseConfig):
        """Default Config for pydantic dataclass and base models."""

        arbitrary_types_allowed: bool = True
        extra: Extra = Extra.forbid
        keep_untouched = (cached_property,)
        frozen: bool = True


GenericT = TypeVar("GenericT", bound="GenericModel")


class GenericModel(_GenericModel):
    """BaseModel class."""

    class Config(BaseConfig):
        """Default Config for pydantic dataclass and base models."""

        arbitrary_types_allowed: bool = True
        extra: Extra = Extra.forbid
        keep_untouched = (cached_property,)

    def evolve(self: GenericT, **kwargs: Any) -> GenericT:
        """Drop-in replacement for attrs.evolve."""
        return self.copy(update=kwargs, deep=True)


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


class DictLikeBaseModel(
    GenericModel, Generic[KeyT, ValueT], MutableMapping[KeyT, ValueT]
):
    """Dict-like Base Model."""

    __root__: Dict[KeyT, ValueT] = {}

    def __getitem__(self, item: KeyT) -> ValueT:
        """Get an item from the underlying dict."""
        return self.__root__.__getitem__(item)

    def __setitem__(self, item: KeyT, val: ValueT) -> None:
        """Set value on item from the underlying dictionary."""
        return self.__root__.__setitem__(item, val)

    def __delitem__(self, item: KeyT) -> None:
        """Delete index from ."""
        return self.__root__.__delitem__(item)

    def __iter__(self) -> Iterator[KeyT]:  # type: ignore[override]
        """Yield elements from keys."""
        for elem in self.__root__:
            yield elem

    def __len__(self) -> int:
        """Return length of mapping."""
        return len(self.__root__)

    def __repr__(self) -> str:
        """Return instance `repr`."""
        return f"{self.__class__.__name__}({self.__root__})"


class ListLikeBaseModel(GenericModel, Generic[ValueT], MutableSequence[ValueT]):
    """List-like Base Model."""

    __root__: List[ValueT] = []

    @overload
    def __getitem__(self, index: SupportsIndex) -> ValueT:  # noqa: D105
        ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[ValueT]:  # noqa: D105
        ...

    def __getitem__(
        self, index: Union[SupportsIndex, slice]
    ) -> Union[ValueT, MutableSequence[ValueT]]:
        """Get an item from the underlying list."""
        return self.__root__[index]

    @overload
    def __setitem__(self, index: SupportsIndex, value: ValueT) -> None:  # noqa: D105
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ValueT]) -> None:  # noqa: D105
        ...

    def __setitem__(self, index, value):  # type: ignore
        """Set value on item from the underlying list."""
        self.__root__[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Delete index from ."""
        return self.__root__.__delitem__(index)

    def __len__(self) -> int:
        """Return length of underlying list."""
        return len(self.__root__)

    def __iter__(self) -> Iterator[ValueT]:  # type: ignore[override]
        """Yield elements from keys."""
        for elem in self.__root__:
            yield elem

    def insert(self, index: SupportsIndex, value: ValueT) -> None:
        """Insert object before index."""
        return self.__root__.insert(index, value)

    def __repr__(self) -> str:
        """Return instance `repr`."""
        return f"{self.__class__.__name__}({self.__root__})"
