"""依赖装配层（原先散在 `ui/runners.py` 里的启动期装配逻辑）。

拆出来的原因：装配"用真 LLM 还是桩、用真 GL 还是软件渲染、用向量库还是内存
语料"属于**运行时环境决策**，既服务于 Gradio UI，也服务于 HTTP 接口和自动化
测试。留在 UI 模块里会导致接口层反向依赖 UI，测试也没法干净地替换依赖。

三种 profile：
  - ``real`` : 全部使用真实依赖，任一不可用即报错（用于生产/预发自检）
  - ``auto`` : 真实依赖优先，不可用时逐项降级（默认，本地开发体验最好）
  - ``test`` : 全部替换为确定性替身（Stub LLM / Stub 渲染 / Fake 检索），
               无需 API Key、无需 GPU，供 CI 的接口自动化测试使用

`ui/runners.py` 现在只是本模块的一层薄封装，保持对旧调用方的兼容。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from shader_agent.agents.analyzer import ShaderAnalyzer
from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.orchestrator import Orchestrator
from shader_agent.config.settings import settings
from shader_agent.observability import is_enabled as lf_enabled
from shader_agent.utils.logger import logger

Profile = Literal["auto", "real", "test"]

DEFAULT_PROFILE: str = os.environ.get("SHADER_AGENT_PROFILE", "auto").strip().lower()


# ============================================================
# 装配选项 + 单例缓存
# ============================================================

@dataclass
class AssemblyOptions:
    """一次装配的全部可调开关（UI 顶部"运行选项"与接口请求参数共用）。"""
    render_backend: str = "auto"        # "auto" | "mock" | "real"
    use_vector_store: str = "auto"      # "auto" | "off"
    use_llm_cache: bool = True
    enable_self_critique: bool = False
    max_fix_loops: int = 1
    top_k: int = 1
    profile: str = ""                   # "" = 取 DEFAULT_PROFILE

    def resolved_profile(self) -> str:
        p = (self.profile or DEFAULT_PROFILE or "auto").lower()
        return p if p in ("auto", "real", "test") else "auto"


@dataclass
class _Assembly:
    """缓存一次装配后的产物。"""
    analyzer: ShaderAnalyzer
    generator: ShaderGenerator
    orchestrator: Orchestrator
    backend_label: str
    vstore_label: str
    retriever: Any = None
    diagnostics: list[str] = field(default_factory=list)
    profile: str = "auto"
    llm_ready: bool = False
    stub: Any = None                    # test profile 下的 StubLLM，供测试断言调用次数


Assembly = _Assembly  # 对外的正式名字（保留旧下划线名以兼容既有 import）

_ASSEMBLY_CACHE: dict[str, _Assembly] = {}


def _cache_key(opts: AssemblyOptions) -> str:
    return (
        f"{opts.resolved_profile()}|{opts.render_backend}|{opts.use_vector_store}|"
        f"{opts.use_llm_cache}|{opts.enable_self_critique}|"
        f"{opts.max_fix_loops}|{opts.top_k}"
    )


# ============================================================
# 各依赖的构建
# ============================================================

def _build_llm_fns(use_cache: bool):
    """构造 chat / json / code / vision / text_critique 五个 llm_fn。

    缺 DEEPSEEK_API_KEY 时返回全 None，Analyzer/Generator 自动走 fallback 路径。

    凭据判断只认 `settings.has_llm_credentials` 这一个口径。原先这里写的是
    `os.environ.get("DEEPSEEK_API_KEY") or settings.deepseek_api_key`——两个来源
    的清洗规则不同（settings 会 strip 并剥引号，os.environ 不会），
    `DEEPSEEK_API_KEY=" "` 这类脏值会让这里判定"有 key"、随后真实请求 401，
    错误现场离根因很远。
    """
    if not settings.has_llm_credentials:
        return None, None, None, None, None, "缺 DEEPSEEK_API_KEY；Analyzer/Generator 走 fallback"
    try:
        from shader_agent.llm.llm_fn import (
            make_chat_fn, make_code_fn, make_json_fn,
            make_text_critique_fn, make_vision_critique_fn,
        )
        return (
            make_chat_fn(use_cache=use_cache),
            make_json_fn(use_cache=use_cache),
            make_code_fn(use_cache=use_cache),
            make_vision_critique_fn(use_cache=use_cache),
            make_text_critique_fn(use_cache=use_cache),
            "",
        )
    except Exception as e:
        return None, None, None, None, None, f"llm_fn 装配失败: {e}"


_VSTORE_SINGLETON: Any = None
_KSTORE_SINGLETON: Any = None
_PSTORE_SINGLETON: Any = None


def _build_vector_store(use_vstore: str):
    """按需懒加载 vector store（进程级单例，避免重复打开 ChromaDB）。"""
    global _VSTORE_SINGLETON
    if use_vstore == "off":
        return None, "已关闭向量库"
    try:
        if _VSTORE_SINGLETON is None:
            from shader_agent.corpus.vector_store import ShaderVectorStore
            _VSTORE_SINGLETON = ShaderVectorStore()
        vs = _VSTORE_SINGLETON
        shader_n = vs.count()
        chunk_n = vs.chunk_count()
        if shader_n == 0 and chunk_n == 0:
            return None, "向量库为空。运行 `python -m scripts.build_corpus` 先建库。"
        if chunk_n > 0:
            return vs, f"已连接（{chunk_n} 子块 / {shader_n} 条）"
        return vs, f"已连接（{shader_n} 条）"
    except Exception as e:
        return None, f"向量库不可用: {e}"


def _warmup_models_sequential(retriever: Any, reranker: Any) -> None:
    """串行预热嵌入模型与重排模型（显存不足时避免同时加载导致 OOM）。"""
    try:
        from shader_agent.embeddings.bge_embedder import get_embedder
        get_embedder().embed_one("warmup")
        logger.info("[warmup] embedder 预热完成")
    except Exception as e:
        logger.warning(f"[warmup] embedder 预热跳过: {e}")
    try:
        reranker.rerank("warmup", [{"text": "warmup", "fused_score": 0.0}])
        logger.info("[warmup] reranker 预热完成")
    except Exception as e:
        logger.warning(f"[warmup] reranker 预热跳过: {e}")


def _build_retriever(vstore: Any):
    """围绕向量库装配混合检索器（向量 + BM25 + 父文档 + 可选重排）。"""
    global _KSTORE_SINGLETON, _PSTORE_SINGLETON
    if vstore is None:
        return None, "无向量库，混合检索不可用"
    try:
        from shader_agent.corpus.keyword_store import KeywordStore
        from shader_agent.corpus.parent_store import ParentDocumentStore
        from shader_agent.corpus.reranker import get_reranker
        from shader_agent.corpus.retriever import HybridRetriever

        if _KSTORE_SINGLETON is None:
            _KSTORE_SINGLETON = KeywordStore.load()
        if _PSTORE_SINGLETON is None:
            _PSTORE_SINGLETON = ParentDocumentStore()
        kstore = _KSTORE_SINGLETON
        pstore = _PSTORE_SINGLETON
        reranker = get_reranker(
            model_name=settings.retrieval.reranker_model,
            enabled=settings.retrieval.use_rerank,
        )
        retr = HybridRetriever(
            vector_store=vstore, keyword_store=kstore,
            parent_store=pstore, reranker=reranker,
        )
        if os.environ.get("SHADER_AGENT_SYNC_WARMUP", "0") == "1":
            _warmup_models_sequential(retr, reranker)
        chunk_n = vstore.chunk_count()
        if chunk_n > 0 and pstore.count() > 0:
            label = f"混合检索（{chunk_n} 子块 / BM25 {kstore.count()}）"
        else:
            label = "混合检索（子块库为空，降级为向量检索）"
        return retr, label
    except Exception as e:
        return None, f"混合检索装配失败，降级为向量检索: {e}"


def _build_render_backend(prefer: str):
    """决定 compiler/renderer 用真 GL 还是 mock。"""
    from shader_agent.rendering import GLSLCompiler, GLSLRenderer
    from shader_agent.rendering.mock import MockCompiler, MockRenderer

    if prefer == "mock":
        return MockCompiler(), MockRenderer(), "mock（用户强制）"

    if prefer in ("auto", "real"):
        c, c_err = GLSLCompiler.try_create()
        r, r_err = GLSLRenderer.try_create()
        if c is not None and r is not None:
            return c, r, "moderngl 真 GL"
        if prefer == "auto":
            return MockCompiler(), MockRenderer(), (
                f"mock（真 GL 不可用：{(c_err or r_err or '').splitlines()[0][:80]}）"
            )
        raise RuntimeError(f"真 GL 不可用：{c_err or r_err}")

    return MockCompiler(), MockRenderer(), "mock（未知 backend 选项）"


def _build_test_backends(opts: AssemblyOptions) -> dict[str, Any]:
    """test profile：全部替换为确定性替身。"""
    from shader_agent.testing.fake_render import StubCompiler, StubRenderer
    from shader_agent.testing.fake_retriever import FakeRetriever, FakeVectorStore
    from shader_agent.testing.stub_llm import make_stub_llm_fns

    chat_fn, json_fn, code_fn, vision_fn, critique_fn, stub = make_stub_llm_fns()
    vstore = None if opts.use_vector_store == "off" else FakeVectorStore()
    retriever = None if vstore is None else FakeRetriever()
    if opts.render_backend == "mock":
        from shader_agent.rendering.mock import MockCompiler, MockRenderer
        compiler, renderer, label = MockCompiler(), MockRenderer(), "mock（1x1 占位）"
    else:
        compiler, renderer, label = StubCompiler(), StubRenderer(), "stub（确定性软件渲染）"
    return {
        "chat_fn": chat_fn, "json_fn": json_fn, "code_fn": code_fn,
        "vision_fn": vision_fn, "text_critique_fn": critique_fn,
        "stub": stub, "vstore": vstore, "retriever": retriever,
        "compiler": compiler, "renderer": renderer,
        "backend_label": label,
        "vstore_label": "内存语料（fake，6 条）" if vstore else "已关闭向量库",
        "retriever_label": "FakeRetriever（确定性排序）" if retriever else "无检索",
    }


# ============================================================
# 主入口
# ============================================================

def get_assembly(opts: AssemblyOptions) -> _Assembly:
    """按 opts 装配 Analyzer/Generator/Orchestrator；命中缓存则直接复用。"""
    key = _cache_key(opts)
    if key in _ASSEMBLY_CACHE:
        return _ASSEMBLY_CACHE[key]

    profile = opts.resolved_profile()
    diag: list[str] = [f"profile: {profile}"]

    if profile == "test":
        b = _build_test_backends(opts)
        chat_fn, json_fn, code_fn = b["chat_fn"], b["json_fn"], b["code_fn"]
        vision_fn, text_critique_fn = b["vision_fn"], b["text_critique_fn"]
        vstore, retriever = b["vstore"], b["retriever"]
        compiler, renderer = b["compiler"], b["renderer"]
        backend_label, vs_msg = b["backend_label"], b["vstore_label"]
        stub = b["stub"]
        diag += ["LLM: stub（确定性）", "向量库: " + vs_msg,
                 "检索: " + b["retriever_label"], "渲染后端: " + backend_label]
        llm_ready = True
    else:
        stub = None
        chat_fn, json_fn, code_fn, vision_fn, text_critique_fn, llm_msg = \
            _build_llm_fns(opts.use_llm_cache)
        llm_ready = chat_fn is not None
        if llm_msg:
            diag.append("LLM: " + llm_msg)
            if profile == "real":
                # 缺凭据与"装配失败"是两回事：前者给出配置指引（MissingCredentialsError
                # 自带三条可执行建议），后者把原始失败原因原样抛出，不要笼统成一句
                # "LLM 不可用"——那会让排查从看日志变成猜。
                settings.require_llm_credentials("real profile 启动")
                raise RuntimeError(f"real profile 下 LLM 装配失败：{llm_msg}")
        else:
            diag.append(
                f"LLM: chat_model={settings.llm.chat_model}, coder={settings.llm.coder_model}"
            )
        vstore, vs_msg = _build_vector_store(opts.use_vector_store)
        diag.append("向量库: " + vs_msg)
        retriever, retr_msg = _build_retriever(vstore)
        diag.append("检索: " + retr_msg)
        render_prefer = "real" if profile == "real" else opts.render_backend
        compiler, renderer, backend_label = _build_render_backend(render_prefer)
        diag.append("渲染后端: " + backend_label)

    if lf_enabled():
        diag.append(f"可观测性: Langfuse 已启用 (env={settings.observability.environment})")
    else:
        diag.append("可观测性: 未启用（未装 langfuse 或未配置 LANGFUSE_PUBLIC_KEY）")

    analyzer = ShaderAnalyzer(
        vector_store=vstore,
        retriever=retriever,
        llm_fn=chat_fn,
        walkthrough_llm=json_fn,
        summary_llm=json_fn,
        effect_llm=chat_fn,
        compare_llm=chat_fn,
        model_name=settings.llm.chat_model,
        top_k=opts.top_k,
        strategy="fourstage",
    )

    generator = ShaderGenerator(
        vector_store=vstore,
        retriever=retriever,
        llm_fn=code_fn,
        compiler=compiler,
        renderer=renderer,
        critique_fn=vision_fn,
        text_critique_fn=text_critique_fn,
        enable_self_critique=opts.enable_self_critique and (
            text_critique_fn is not None or vision_fn is not None
        ),
        model_name=settings.llm.coder_model,
        max_fix_loops=opts.max_fix_loops,
        top_k=opts.top_k,
    )

    asm = _Assembly(
        analyzer=analyzer,
        generator=generator,
        orchestrator=Orchestrator(analyzer=analyzer, generator=generator),
        backend_label=backend_label,
        vstore_label=vs_msg,
        retriever=retriever,
        diagnostics=diag,
        profile=profile,
        llm_ready=llm_ready,
        stub=stub,
    )
    _ASSEMBLY_CACHE[key] = asm
    return asm


def clear_assembly_cache() -> int:
    n = len(_ASSEMBLY_CACHE)
    _ASSEMBLY_CACHE.clear()
    return n
