from typing import Any, Self

import msgspec
from rich.console import Console

rich_console = Console(highlight=False)


class BaseConfig(msgspec.Struct):
    def __save__(self, config_path):
        config_yaml = msgspec.yaml.encode(self)
        with open(config_path, "wb") as f:
            f.write(config_yaml)

    @classmethod
    def __load__(cls, config_path) -> Self:
        with open(config_path, "rb") as f:
            config_yaml = f.read()
        return msgspec.yaml.decode(config_yaml, type=cls)

    def __repr__(self, num_tabs: int = 0, use_rich=False, subtitle: str = "") -> str:
        CLASS_PREFIX = "[b red]" if use_rich else ""
        FIELD_PREFIX = "[blue]" if use_rich else ""
        VALUE_PREFIX = "[green]" if use_rich else ""
        SUFFIX = "[/]" if use_rich else ""
        TAB = "  "

        class_name = f"{CLASS_PREFIX}{TAB * num_tabs}{self.__class__.__name__}{SUFFIX}{subtitle}"

        reprs = []
        for k in self.__struct_fields__:
            repr_k = f"{FIELD_PREFIX}{k}:{SUFFIX}"

            value = getattr(self, k)
            if isinstance(value, BaseConfig):
                subtitle_k = f"{FIELD_PREFIX}({k}){SUFFIX}"
                repr_field = value.__repr__(num_tabs=num_tabs + 1, use_rich=use_rich, subtitle=subtitle_k)

            else:
                repr_v = f"{VALUE_PREFIX}{value}{SUFFIX}"
                repr_field = f"{TAB * (num_tabs + 1)}{repr_k} {repr_v}"
            reprs.append(repr_field)

        return f"{class_name}\n{'\n'.join(reprs)}"

    def __rich__(self) -> str:
        return self.__repr__(use_rich=True)

    def __print__(self) -> None:
        rich_console.print(self.__rich__())

    def __to_dict__(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


def print_config(config: BaseConfig) -> None:
    config.__print__()
