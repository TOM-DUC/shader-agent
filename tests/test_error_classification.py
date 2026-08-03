"""错误归类与 draft 失败分流的守护用例。

这些用例守的是"故障发生时，调用方拿到的错误码能不能指向正确的处置手段"。
它们全部离线、无依赖，秒级返回。

对应的三个真实缺陷（坏掉之后会怎样，都写在各用例的 docstring 里）：

  1. 缺凭据被归成 INTERNAL 50001 —— "没配 key"被当成"内部错误要报警"；
  2. 检索故障被归成 LLM_TIMEOUT / INTERNAL —— 排查方向被带到大模型上；
  3. 修正轮 LLM 抛异常时整体报错 —— 首轮已经产出的代码被一起丢掉。

跑法：
    pytest tests/test_error_classification.py -v
"""
from __future__ import annotations

import pytest

from shader_agent.agents.generator import ShaderGenerator
from shader_agent.agents.schemas import GeneratedShader, Message
from shader_agent.config.settings import MissingCredentialsError, Settings
from shader_agent.service.errors import (
    ErrorCode,
    ServiceError,
    classify_retrieval_error,
    classify_upstream_error,
)


# ============================================================
# 1) 凭据缺失 → 50301，不是 50001
# ============================================================

def test_missing_credentials_maps_to_llm_unavailable() -> None:
    """`MissingCredentialsError` 必须归到 LLM_UNAVAILABLE。

    坏掉之后的表现：新同事 clone 下来没配 key，一跑生成拿到 500 INTERNAL，
    于是去翻服务端日志找"内部错误"——而正确的下一步只是配一个环境变量。
    异常自带的三条可执行建议也必须原样透传，不能被格式化掉。
    """
    exc = MissingCredentialsError("调用大模型需要 DEEPSEEK_API_KEY，但未配置。")
    err = classify_upstream_error(exc)
    assert err.code is ErrorCode.LLM_UNAVAILABLE
    assert err.http_status == 503
    assert "DEEPSEEK_API_KEY" in err.message


def test_llm_fn_not_ready_maps_to_llm_unavailable() -> None:
    """auto profile 无 key 时，DraftCodeAction 抛的是普通 RuntimeError。

    它的文案里是 `DEEPSEEK_API_KEY`（下划线），而不是上游常见的 `api key`
    （空格）。关键词表少了下划线那条，这条路径就会静默落进 50001。
    """
    exc = RuntimeError(
        "Generator 的 LLM 函数（code_fn）未就绪。"
        "请检查 DEEPSEEK_API_KEY 是否配置，或等待 LLM 初始化完成后重试。"
    )
    assert classify_upstream_error(exc).code is ErrorCode.LLM_UNAVAILABLE


# ============================================================
# 2) 上游异常的既有归类不能被上面的改动带偏
# ============================================================

@pytest.mark.parametrize(
    "message, expected",
    [
        ("StubLLMTimeout: stub llm: request timed out after 120s", ErrorCode.LLM_TIMEOUT),
        ("stub llm: 429 Too Many Requests (rate limit exceeded)", ErrorCode.RATE_LIMITED),
        ("stub llm: 401 invalid_api_key", ErrorCode.LLM_UNAVAILABLE),
        ("chromadb: could not connect to persistent client", ErrorCode.RETRIEVAL_UNAVAILABLE),
    ],
)
def test_known_upstream_errors_keep_their_codes(message: str, expected: ErrorCode) -> None:
    """故障注入用例断言的四个错误码，在这里再守一道。

    接口层的故障注入用例要起服务、走 HTTP，反馈慢；这条纯函数用例秒级返回，
    改关键词表时先被它挡住。
    """
    assert classify_upstream_error(RuntimeError(message)).code is expected


def test_service_error_passes_through_unchanged() -> None:
    """已经是 ServiceError 的不许被二次归类，否则错误码会在传递中漂移。"""
    original = ServiceError(ErrorCode.UNSUPPORTED_SHADER, "用了 iChannel0")
    assert classify_upstream_error(original) is original


# ============================================================
# 3) 检索调用点：不靠猜
# ============================================================

def test_retrieval_timeout_is_not_reported_as_llm_timeout() -> None:
    """检索超时走通用归类会命中 timeout 关键词，被判成 LLM_TIMEOUT 50401。

    坏掉之后的表现：向量库慢查询导致的故障，值班同学照着错误码去查大模型，
    方向完全错了。调用点已经知道"这是检索"，就不该再靠文本猜。
    """
    exc = TimeoutError("vector query timed out after 30s")
    assert classify_upstream_error(exc).code is ErrorCode.LLM_TIMEOUT
    assert classify_retrieval_error(exc).code is ErrorCode.RETRIEVAL_UNAVAILABLE


def test_retrieval_error_with_unfamiliar_wording_still_maps_to_50303() -> None:
    """换库、换版本后报错措辞会变，不能依赖某一句具体文案。"""
    exc = RuntimeError("some brand new storage backend blew up")
    assert classify_upstream_error(exc).code is ErrorCode.INTERNAL
    err = classify_retrieval_error(exc)
    assert err.code is ErrorCode.RETRIEVAL_UNAVAILABLE
    assert err.retryable is True


def test_retrieval_rate_limit_keeps_its_own_code() -> None:
    """限流在检索场景同样成立，不该被一律压平成 50303。"""
    exc = RuntimeError("429 Too Many Requests")
    assert classify_retrieval_error(exc).code is ErrorCode.RATE_LIMITED


# ============================================================
# 4) draft 失败按轮次分流
# ============================================================

def test_first_round_draft_failure_raises() -> None:
    """首轮就失败 = 什么都没产出，必须抛给上游归类。

    坏掉之后的表现：接口返回 200 + code=""，用户以为成功了，实际拿到空白；
    而真正的原因（超时/限流/鉴权）永远到不了 classify_upstream_error。
    """
    def boom(_messages):
        raise TimeoutError("stub llm: request timed out")

    gen = ShaderGenerator(llm_fn=boom, max_fix_loops=1)
    with pytest.raises(RuntimeError) as ei:
        gen.handle(Message(role="user", content="蓝色波纹"))
    # 原始错误文本必须保留，否则 service 层归类无从下手
    assert "timed out" in str(ei.value)


def test_fix_round_failure_keeps_the_first_round_result() -> None:
    """修正轮失败时手里已有一份编不过的代码，如实返回比整体报错更有用。

    坏掉之后的表现：首轮产出的代码因为修复轮撞上一次限流被整个丢掉，
    用户拿到 429 而不是"这是初稿，编译没过，原因如下"。
    """
    calls = {"n": 0}

    def flaky(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            # 无 mainImage + 括号不平：静态校验必然拦下，从而进入修正轮
            return "// EXPLAIN: bad\nfloat oops(){ return 1.0;\n"
        raise RuntimeError("stub llm: 429 Too Many Requests")

    gen = ShaderGenerator(llm_fn=flaky, max_fix_loops=1)
    out = gen.handle(Message(role="user", content="蓝色波纹"))
    g = GeneratedShader(**out.payload)

    assert calls["n"] == 2, "修正轮应当被真正尝试过"
    assert "oops" in g.code, "首轮成品不许被丢掉"
    assert g.compile_result.ok is False
    assert "fix-round aborted" in g.compile_result.errors, "中止原因必须留痕"
    # 中止轮没有产出，不计入轮数——这个字段要进指标看板，虚高就失去意义
    assert g.iterations == 1


# ============================================================
# 5) dotenv 开关的两条路径必须一起关
# ============================================================

def test_load_dotenv_switch_controls_env_file_too(monkeypatch) -> None:
    """`SHADER_AGENT_LOAD_DOTENV` 必须同时管住 load_dotenv 与 pydantic env_file。

    坏掉之后的表现：`SHADER_AGENT_LOAD_DOTENV=0` 只跳过了 os.environ 注入，
    Settings 仍从 env_file 读到真 key——同一个意图有了两个开关，
    只关其中一个的人会得到"我明明关了 .env，怎么还认得出 key"的困惑，
    而配置层守护用例在有 .env 的开发机上会假绿。
    """
    import shader_agent.config.settings as st

    monkeypatch.setattr(st, "LOAD_DOTENV", False)
    assert st._env_file_arg() is None

    monkeypatch.setattr(st, "LOAD_DOTENV", True)
    assert st._env_file_arg() == str(st.PROJECT_ROOT / ".env")


def test_settings_still_constructible_without_any_env(tmp_path) -> None:
    """开关关闭 + config.yaml 缺失，仍然要能拿到一份全默认配置。"""
    monkey = Settings.load(config_path=tmp_path / "not-exists.yaml")
    assert monkey.llm.chat_model
    assert Settings(_env_file=None, deepseek_api_key="").has_llm_credentials is False
