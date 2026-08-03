"""报告附件的薄封装。

装了 allure 就走 allure，没装就退化成 no-op —— 测试框架不应该因为报告工具没装
就跑不起来。CI 里装 allure-pytest 生成报告，本地快速调试可以什么都不装。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

try:  # pragma: no cover - 取决于环境
    import allure  # type: ignore
    from allure_commons.types import AttachmentType  # type: ignore
    _HAS_ALLURE = True
except Exception:  # pragma: no cover
    allure = None  # type: ignore
    AttachmentType = None  # type: ignore
    _HAS_ALLURE = False


def has_allure() -> bool:
    return _HAS_ALLURE


def attach_text(name: str, body: str) -> None:
    if _HAS_ALLURE:
        allure.attach(body, name=name, attachment_type=AttachmentType.TEXT)


def attach_png(name: str, body: bytes) -> None:
    if _HAS_ALLURE:
        allure.attach(body, name=name, attachment_type=AttachmentType.PNG)


@contextmanager
def step(title: str) -> Iterator[None]:
    if _HAS_ALLURE:
        with allure.step(title):
            yield
    else:
        yield
