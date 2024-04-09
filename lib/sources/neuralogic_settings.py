import warnings

from neuralogic.core.settings import Settings as _NSettings


class NeuralogicSettings(_NSettings):
    def __new__(
        cls,
        *,
        compute_neuron_layer_indices: bool | None = None,
        **kwargs,
    ) -> "NeuralogicSettings":
        if compute_neuron_layer_indices is False:
            warnings.warn("`NeuralogicSettings.compute_neuron_layer_indices` will be set to True")

        return _NSettings(**kwargs, compute_neuron_layer_indices=True)  # pyright: ignore
