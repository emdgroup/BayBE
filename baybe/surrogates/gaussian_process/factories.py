"""Component factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, Union, get_origin

from attrs import define, field

from baybe.kernels.base import Kernel
from baybe.searchspace import SearchSpace
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from torch import Tensor

_T = TypeVar("_T")


class ComponentFactory(Protocol, Generic[_T]):
    """The interface expected for Gaussian process component factories."""

    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> _T:
        """Create a Gaussian process component for the given DOE context."""
        ...


def is_component_factory(type: Any) -> bool:
    """Determine if a given type modulo its generics is a component factory."""
    return type is ComponentFactory or get_origin(type) is ComponentFactory


# Register de-/serialization hooks
converter.register_structure_hook_func(
    is_component_factory, get_base_structure_hook(ComponentFactory)
)
converter.register_unstructure_hook_func(is_component_factory, unstructure_base)


@define(frozen=True)
class SingletonFactory(ComponentFactory, SerialMixin, Generic[_T]):
    """A singleton factory that returns a fixed pre-defined object upon request."""

    instance: _T = field()
    """The fixed object to be returned by the factory."""

    def __call__(  # noqa: D102
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> _T:
        # See base class.

        return self.instance


def to_kernel_factory(
    x: Union[Kernel, ComponentFactory[Kernel]], /
) -> ComponentFactory[Kernel]:
    """Wrap a kernel into a singleton factory (with factory passthrough)."""
    return x.to_factory() if isinstance(x, Kernel) else x
