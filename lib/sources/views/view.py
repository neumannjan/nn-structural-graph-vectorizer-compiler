import functools
from typing import Type

from lib.sources.base import Network
from lib.sources.minimal_api.base import MinimalAPINetwork
from lib.sources.minimal_api.network_and_ordinals import MinimalAPINetworkAndOrdinals
from lib.sources.minimal_api.ordinals import MinimalAPIOrdinals
from lib.sources.minimal_api_bridge import NetworkImpl
from lib.sources.minimal_api_bridge_reverse import MinimalAPINetworkFromNetwork, MinimalAPIOrdinalsFromNetwork
from lib.utils import ExtendsDynamicError, InheritDynamic, extends_dynamic


class _ViewInheritDynamic(InheritDynamic):
    def __get__(self, obj: object, objtype: Type | None = None):
        try:
            return super().__get__(obj, objtype)
        except ExtendsDynamicError as e:
            cls = objtype or obj.__class__

            the_base = None
            for b in (ViewBasis, NetworkViewBasis, OrdinalsViewBasis):
                if b in cls.__bases__:
                    the_base = b
                    break

            if the_base is None:
                raise ExtendsDynamicError(
                    f"Unknown error. You are using {_ViewInheritDynamic.__name__} when you shouldn't be."
                ) from e

            the_decorator = {
                ViewBasis: view_basis,
                NetworkViewBasis: network_view_basis,
                OrdinalsViewBasis: ordinals_view_basis,
            }[the_base]

            raise ExtendsDynamicError(
                f"Your class {cls.__name__} failed to inherit the methods. "
                f"Did you forget to apply the @{the_decorator.__name__} decorator on it?"
            ) from e


class NetworkViewBasis(MinimalAPINetwork):
    """
    A base class for a view on a `MinimalAPINetwork`.

    See documentation for the `@view` decorator for usage.
    """

    def __init__(self, network: MinimalAPINetwork) -> None:
        self.network = network

    get_layers = _ViewInheritDynamic()
    get_layers_map = _ViewInheritDynamic()
    get_ids = _ViewInheritDynamic()
    get_inputs = _ViewInheritDynamic()
    get_input_lengths = _ViewInheritDynamic()
    get_layer_neurons = _ViewInheritDynamic()
    get_input_weights = _ViewInheritDynamic()
    get_biases = _ViewInheritDynamic()
    get_values_numpy = _ViewInheritDynamic()
    get_values_torch = _ViewInheritDynamic()
    get_transformations = _ViewInheritDynamic()
    get_aggregations = _ViewInheritDynamic()
    slice = _ViewInheritDynamic()
    select_ids = _ViewInheritDynamic()


def network_view_basis(cls):
    """Decorate a `NetworkViewBasis` implementation to make it work."""
    if len(cls.__bases__) == 0 or cls.__bases__[0] != NetworkViewBasis:
        raise ValueError(
            f"{cls.__name__}: Must extend {NetworkViewBasis.__name__} when "
            f"using the @{network_view_basis.__name__} decorator!"
        )

    return extends_dynamic("network")(cls)


class OrdinalsViewBasis(MinimalAPIOrdinals):
    """
    A base class for a view on a `MinimalAPIOrdinals`.

    See documentation for the `@view` decorator for usage.
    """

    def __init__(self, ordinals: MinimalAPIOrdinals) -> None:
        self.ordinals = ordinals

    get_ordinal = _ViewInheritDynamic()
    get_all_ordinals = _ViewInheritDynamic()
    get_id = _ViewInheritDynamic()
    get_ordinals_for_layer = _ViewInheritDynamic()


def ordinals_view_basis(cls):
    """Decorate an `OrdinalsViewBasis` implementation to make it work."""
    if len(cls.__bases__) == 0 or cls.__bases__[0] != OrdinalsViewBasis:
        raise ValueError(
            f"{cls.__name__}: Must extend {OrdinalsViewBasis.__name__} when "
            f"using the @{ordinals_view_basis.__name__} decorator!"
        )
    return extends_dynamic("ordinals")(cls)


def _build_network_and_ordinals(
    network: MinimalAPINetwork, ordinals: MinimalAPIOrdinals
) -> MinimalAPINetworkAndOrdinals:
    cls = type("MinimalAPINetworkAndOrdinalsImpl", (MinimalAPINetworkAndOrdinals,), {})

    def __getattribute__(self, key: str):
        if hasattr(network, key):
            return getattr(network, key)

        if hasattr(ordinals, key):
            return getattr(ordinals, key)

        raise KeyError(key)

    cls.__getattribute__ = __getattribute__
    return cls()


class ViewBasis(NetworkViewBasis, OrdinalsViewBasis):
    """
    A base class for a view on a `MinimalAPINetwork` and `MinimalAPIOrdinals` together.

    See documentation for the `@view` decorator for usage.
    You may also use `NetworkViewBasis` and `OrdinalsViewBasis` individually instead of `ViewBasis`.
    """

    def __init__(self, network: MinimalAPINetwork, ordinals: MinimalAPIOrdinals) -> None:
        NetworkViewBasis.__init__(self, network)
        OrdinalsViewBasis.__init__(self, ordinals)
        self._ViewBasis__network_and_ordinals = _build_network_and_ordinals(network, ordinals)


def view_basis(cls):
    """Decorate a `ViewBasis` implementation to make it work."""
    if len(cls.__bases__) == 0 or cls.__bases__[0] != ViewBasis:
        raise ValueError(
            f"{cls.__name__}: Must extend {ViewBasis.__name__} when " f"using the @{view_basis.__name__} decorator!"
        )

    return extends_dynamic("_ViewBasis__network_and_ordinals")(cls)


class View(NetworkImpl):
    """
    A base class for a view on a Network.

    See documentation for the `@view` decorator for usage.
    """

    def __init__(self, network: Network) -> None:
        self.network = network
        view_name = View.__name__
        cls_name = self.__class__.__name__
        decorator_name = view.__name__
        raise Exception(f"{cls_name}: {view_name} must be used with @{decorator_name} decorator!")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.network)})"


def view(
    view_basis: Type[NetworkViewBasis] | Type[OrdinalsViewBasis] | Type[ViewBasis],
    ordinals_view_basis: Type[OrdinalsViewBasis] | None = None,
):
    """Build a view for a Network. This is a decorator.

    Example usage:
    ```
    @view_basis
    class MyViewBasis(ViewBasis):
        # this is my place to override the minimal API of a given network

        def get_ids(self, neurons):
            # TODO: my custom overridden implementation of get_ids
            # You may use self.network.[any method] and self.ordinals.[any method]
            # to access the unaltered (original) minimal API of the original network.
            # You may use self.[any method] to access the altered minimal API.
            ...


    @view(MyViewBasis)
    class MyView(View):
        # no need to do anything here
        pass

    # usage:
    network: Network = ...
    network_viewed = MyView(network)
    # use network_viewed like you would use network
    ```

    The `ViewBasis` is a 'minimal API' implementation of both a network and its ordinals,
    where all methods that are not explicitly overridden use the same implementation
    as the original `self.network` and `self.ordinals`.
    To make the method reuse work, the `@view_basis` decorator is needed.

    The `View` class together with the `@view` decorator builds a 'full API' view for a 'full API' network
    from the 'minimal API' `ViewBasis` version.

    Please learn about the difference between minimal API and full API in the documentation for
    `MinimalAPINetwork` and `Network` respectively.

    You may also use custom constructor parameters (besides the already present ones):

    ```
    @view_basis
    class MyViewBasis(ViewBasis):
        def __init__(self, network, ordinals, something_else):
            super().__init__(network, ordinals)
            self.something_else = something_else

        # TODO: override methods here


    @view(MyViewBasis)
    class MyView(View):
        def __init__(self, network, something_else):
            # optional constructor here as well, with empty body
            # only needed to get correct code completion in various IDEs
            # otherwise not needed, @view decorator takes care of the constructor fully

            # no need to do anything here
            pass

    # usage
    network: Network = ...
    network_viewed = MyView(network, something_else=...)
    ```
    """

    def the_decorator(cls: Type):
        @functools.wraps(cls)
        def _f(cls):
            if not cls.__bases__ == (View,):
                cls_name = cls.__name__
                view_name = View.__name__
                decorator_name = view.__name__
                raise Exception(f"{cls_name}: @{decorator_name} decorator requires that the class extends {view_name}!")

            def __init__(self, network: Network, *kargs, **kwargs):
                self.network = network
                if issubclass(view_basis, ViewBasis):
                    assert ordinals_view_basis is None or view_basis == ordinals_view_basis
                    new_minimal_api = view_basis(
                        MinimalAPINetworkFromNetwork(network),
                        MinimalAPIOrdinalsFromNetwork(network),
                        *kargs,
                        **kwargs,
                    )
                    custom_ordinals = new_minimal_api
                elif issubclass(view_basis, NetworkViewBasis):
                    new_minimal_api = view_basis(MinimalAPINetworkFromNetwork(network), *kargs, **kwargs)

                    if ordinals_view_basis is not None:
                        custom_ordinals = ordinals_view_basis(MinimalAPIOrdinalsFromNetwork(network), *kargs, **kwargs)
                    else:
                        custom_ordinals = None
                elif issubclass(view_basis, OrdinalsViewBasis):
                    assert ordinals_view_basis is None or view_basis == ordinals_view_basis
                    new_minimal_api = MinimalAPINetworkFromNetwork(network)
                    custom_ordinals = view_basis(MinimalAPIOrdinalsFromNetwork(network), *kargs, **kwargs)
                else:
                    raise Exception(
                        f"network_view_basis must either subclass {ViewBasis.__name__} or {NetworkViewBasis.__name__}."
                    )

                NetworkImpl.__init__(self, new_minimal_api, custom_ordinals)

            cls.__init__ = __init__
            return cls

        return _f(cls)

    return the_decorator
