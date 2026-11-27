"""
infra.registry —— 通用注册中心(只剩纯机制,不知业务)

提供 `Registry[T]`:按 name 装/查/列举任意 type[T]。各能力模块(llm / memory / ...)
持有自己的 Registry 实例,在自己的 __init__.py 里 re-export `register_*` /
`get_*` / `list_*` 装饰器与查询函数。infra 这层不再写死"世界上有 LLM 和
memory backend",避免业务知识溢出到基础设施层。

工具不走 registry —— 工具由 MCP 在运行时动态发现。
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """name → type[T] 的通用注册表;典型用法见各能力模块的 __init__.py。"""

    def __init__(self) -> None:
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """装饰器:`@my_registry.register("foo") class Foo: ...`"""
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{name!r} 已经注册过了")
            self._items[name] = cls
            return cls
        return deco

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(
                f"{name!r} 未注册;已注册:{sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        """按字母序返回所有已注册的 name。"""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)
