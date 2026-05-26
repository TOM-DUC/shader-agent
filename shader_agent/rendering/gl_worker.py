"""GL 单线程执行器（修复 "cannot create program"）。

为什么需要
==========
moderngl 的 standalone context 是**线程绑定**的：它只在创建它的那个线程里
"current"。一旦在别的线程调用 `ctx.program()` / `ctx.buffer()` / 渲染，
就会报 `cannot create program`（glCreateProgram 返回 0，因为当前线程没有
活跃的 GL context）。

Gradio 的 `queue()` 用线程池分发请求：第 1 次请求在线程 A 建好 context、
渲染成功；第 2 次请求落到线程 B，复用了绑定在 A 上的 context，于是
`ctx.program()` 直接失败——这正是"第一次能出图、之后都不出图、
生成时三次编译全 cannot create program"的根因。

解决方案
========
把**所有** GL 操作（创建 context、创建 buffer、编译 program、渲染读回）
都丢到一个**独立的常驻线程**上串行执行。context 在这个线程上创建、也只在
这个线程上使用，线程亲和性问题彻底消失。

代价：所有 GL 调用串行化。但单帧渲染只有几十毫秒，且 GL context 本就无法
真正并发共享，串行是正确且安全的。
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable


class _GLWorker:
    """单线程任务队列：提交的可调用对象都在同一条线程上执行。"""

    def __init__(self) -> None:
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            t = threading.Thread(target=self._loop, name="gl-worker", daemon=True)
            t.start()
            self._thread = t

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """在 GL 线程上同步执行 fn，阻塞直到返回；异常会原样抛回调用方。"""
        # 若已经在 GL 线程上（嵌套调用），直接执行避免自我死锁
        if threading.current_thread() is self._thread:
            return fn(*args, **kwargs)
        self._ensure_started()
        done = threading.Event()
        box: dict[str, Any] = {}
        self._q.put((fn, args, kwargs, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _loop(self) -> None:
        while True:
            fn, args, kwargs, box, done = self._q.get()
            try:
                box["value"] = fn(*args, **kwargs)
            except BaseException as e:  # noqa: BLE001 —— 要把异常带回调用线程
                box["error"] = e
            finally:
                done.set()


# 进程级单例
gl_worker = _GLWorker()


def run_on_gl(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """把一个 GL 操作派发到专用线程执行并取回结果。"""
    return gl_worker.submit(fn, *args, **kwargs)
