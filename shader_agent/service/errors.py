"""统一错误码与业务异常。

设计原则：
- **业务失败不等于接口失败**：编译不通过、检索为空属于业务结果，走 200 + data；
  只有"调用方参数错"或"依赖不可用"才抛 ServiceError。
- 错误码是稳定契约（写进接口文档与测试用例 YAML），HTTP 状态码只做粗分类，
  精确断言一律用 `code` 字段，避免测试因框架默认状态码变动而漂移。
- 错误码规则：前三位对齐 HTTP 语义，后两位为业务序号，便于一眼判类。
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

from shader_agent.config.settings import MissingCredentialsError


class ErrorCode(IntEnum):
    """接口统一错误码。"""

    OK = 0

    # --- 4xx：调用方问题 ---
    INVALID_PARAM = 40001          # 参数校验失败（缺字段 / 类型错 / 越界）
    EMPTY_INPUT = 40002            # 关键输入为空（code / description 空白）
    INPUT_TOO_LARGE = 40003        # 输入超出长度上限
    UNSUPPORTED_SHADER = 40004     # 使用了本地不支持的特性（iChannel/多 pass）
    SHADER_COMPILE_ERROR = 40005   # 送进来的 shader 编不过（渲染类接口）
    UNAUTHORIZED = 40101           # 缺少或错误的 API Key
    NOT_FOUND = 40401
    RATE_LIMITED = 42901           # 上游或本服务限流

    # --- 5xx：服务端 / 依赖问题 ---
    INTERNAL = 50001               # 未归类的内部异常
    LLM_ERROR = 50002              # 大模型返回异常（非法 JSON / 空响应）
    GENERATION_FAILED = 50003      # 多轮修复后仍未产出可用结果
    LLM_UNAVAILABLE = 50301        # 未配置 key / 客户端装配失败
    RENDER_UNAVAILABLE = 50302     # 渲染后端不可用（无 GL 且禁用 mock）
    RETRIEVAL_UNAVAILABLE = 50303  # 向量库 / 检索器不可用
    LLM_TIMEOUT = 50401            # 上游超时


#: 错误码 → 建议 HTTP 状态码
_HTTP_STATUS: dict[int, int] = {
    ErrorCode.OK: 200,
    ErrorCode.INVALID_PARAM: 422,
    ErrorCode.EMPTY_INPUT: 422,
    ErrorCode.INPUT_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_SHADER: 422,
    ErrorCode.SHADER_COMPILE_ERROR: 422,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.INTERNAL: 500,
    ErrorCode.LLM_ERROR: 500,
    ErrorCode.GENERATION_FAILED: 500,
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.RENDER_UNAVAILABLE: 503,
    ErrorCode.RETRIEVAL_UNAVAILABLE: 503,
    ErrorCode.LLM_TIMEOUT: 504,
}


def http_status_of(code: int | ErrorCode) -> int:
    return _HTTP_STATUS.get(int(code), 500)


class ServiceError(Exception):
    """业务层统一异常。API 层捕获后转成统一响应体。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str = "",
        *,
        detail: Any = None,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.message = message or code.name.lower()
        self.detail = detail
        self.retryable = retryable
        super().__init__(f"[{int(code)}] {self.message}")

    @property
    def http_status(self) -> int:
        return http_status_of(self.code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": int(self.code),
            "message": self.message,
            "detail": self.detail,
            "retryable": self.retryable,
        }


# ---------------------------------------------------------------
# 上游异常 → 错误码 的统一归类
# ---------------------------------------------------------------

_TIMEOUT_HINTS = ("timeout", "timed out", "readtimeout", "connecttimeout")
_RATELIMIT_HINTS = ("rate limit", "ratelimit", "429", "too many requests", "tpm", "rpm")
#: 注意 `api_key`（下划线）与 `api key`（空格）都要覆盖：前者来自我们自己的
#: 报错文本（"请检查 DEEPSEEK_API_KEY 是否配置"），后者来自 OpenAI 兼容上游。
#: 少了下划线那条，"没配 key"会被归成 INTERNAL 50001——变成"内部错误要报警"，
#: 而它其实只是"这台机器没配凭据"，两者的处置手段完全不同。
_AUTH_HINTS = ("api key", "api_key", "unauthorized", "401", "invalid_api_key",
               "未就绪", "未配置")
_RETRIEVAL_HINTS = ("chromadb", "chroma", "could not connect to persistent client",
                    "vector store", "vectorstore", "keyword store", "parent store",
                    "retriever", "retrieval")


def classify_upstream_error(exc: BaseException) -> ServiceError:
    """把第三方依赖（LLM / 渲染 / 检索）抛出的裸异常归一化成 ServiceError。

    这样上层只面对有限的错误码集合，测试也能对"超时""限流""不可用"分别断言，
    而不是对着一堆 500 猜原因。
    """
    text = f"{type(exc).__name__}: {exc}".lower()

    if isinstance(exc, ServiceError):
        return exc
    if isinstance(exc, MissingCredentialsError):
        # 缺凭据是确定性的，不该靠关键词猜。异常自带三条可执行建议，原样透传。
        return ServiceError(ErrorCode.LLM_UNAVAILABLE, str(exc))
    if isinstance(exc, TimeoutError) or any(h in text for h in _TIMEOUT_HINTS):
        return ServiceError(
            ErrorCode.LLM_TIMEOUT, f"上游超时：{exc}", retryable=True,
        )
    if any(h in text for h in _RATELIMIT_HINTS):
        return ServiceError(
            ErrorCode.RATE_LIMITED, f"上游限流：{exc}", retryable=True,
        )
    if any(h in text for h in _AUTH_HINTS):
        return ServiceError(ErrorCode.LLM_UNAVAILABLE, f"上游鉴权失败：{exc}")
    if any(h in text for h in _RETRIEVAL_HINTS):
        # 检索类连接/不可用错误要归到 RETRIEVAL_UNAVAILABLE（50303），
        # 而不是笼统的 INTERNAL（50001）——前者对调用方意味着"检索依赖挂了，
        # 可降级"，后者是"内部错误，要报警"，运维手段完全不同。
        return ServiceError(
            ErrorCode.RETRIEVAL_UNAVAILABLE, f"检索不可用：{exc}", retryable=True,
        )
    if isinstance(exc, (ConnectionError, OSError)):
        return ServiceError(
            ErrorCode.LLM_UNAVAILABLE, f"上游连接失败：{exc}", retryable=True,
        )
    return ServiceError(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")


#: 在检索调用点上，"这是检索故障"是已知事实，不该再靠错误文本去猜。
#: 靠猜会有两个盲区：① 检索器报错措辞换了（换库、换版本）就退回 50001；
#: ② 检索**超时**会先命中 _TIMEOUT_HINTS，被归成 LLM_TIMEOUT 50401——
#: 调用方据此去查大模型，方向完全错了。
_LLM_FLAVORED_CODES = frozenset({
    ErrorCode.INTERNAL, ErrorCode.LLM_TIMEOUT, ErrorCode.LLM_UNAVAILABLE,
})


def classify_retrieval_error(exc: BaseException) -> ServiceError:
    """检索链路专用归类：先走通用归类，再把"看起来像 LLM/内部"的纠正为 50303。

    保留通用归类的结果是有意的：限流（42901）等错误码在检索场景同样成立，
    没必要一律压平成 50303。
    """
    err = classify_upstream_error(exc)
    if err.code in _LLM_FLAVORED_CODES:
        return ServiceError(
            ErrorCode.RETRIEVAL_UNAVAILABLE, f"检索不可用：{exc}", retryable=True,
        )
    return err
