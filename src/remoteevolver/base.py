"""
remoteevolver.base —— Agent 抽象

只规定接口：从用户输入跑到最终答案。
具体怎么循环、何时反思，由子类（ReActAgent 等）决定。
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def run(self, task: str) -> str:
        """处理单条用户任务，返回最终自然语言答复。"""
        ...
