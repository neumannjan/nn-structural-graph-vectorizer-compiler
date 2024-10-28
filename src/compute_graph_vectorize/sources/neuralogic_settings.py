from neuralogic.core.settings import Settings as _NSettings


class NeuralogicSettings(_NSettings):
    def __new__(cls, **kwargs) -> "NeuralogicSettings":
        return _NSettings(**kwargs)  # pyright: ignore
