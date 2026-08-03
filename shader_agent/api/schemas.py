"""接口层数据契约。

约定：
- **所有**接口返回同一个信封 `{code, message, request_id, elapsed_ms, data}`；
  `code=0` 才代表成功，HTTP 状态码只做粗分类。测试断言一律看 `code`。
- 请求模型开启 `extra="forbid"`：多传字段直接 422，避免"字段名写错但接口
  静默忽略"这类最难查的问题。
- 数值边界写在模型里（而不是散在业务代码），既是文档也是校验，
  测试可以直接按边界值组织正常/异常/边界三类用例。
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Complexity = Literal["minimal", "simple", "moderate", "complex"]


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- 统一响应信封 ----------------

class ApiResponse(BaseModel):
    code: int = Field(0, description="业务错误码，0 表示成功")
    message: str = Field("ok", description="人类可读的结果说明")
    request_id: str = Field("", description="链路追踪 ID，与响应头 X-Request-Id 一致")
    elapsed_ms: float = Field(0.0, description="服务端处理耗时（毫秒）")
    data: Optional[Any] = Field(None, description="业务数据，失败时为 null 或错误详情")


# ---------------- 请求体 ----------------

class ValidateRequest(_Req):
    code: str = Field(..., min_length=1, max_length=20000, description="GLSL 源码")
    require_dynamic: Optional[bool] = Field(
        None, description="是否要求使用 iTime；None 表示不检查")


class CompileRequest(_Req):
    code: str = Field(..., min_length=1, max_length=20000)


class RenderRequest(_Req):
    code: str = Field(..., min_length=1, max_length=20000)
    width: int = Field(768, ge=16, le=1920)
    height: int = Field(576, ge=16, le=1080)
    time: float = Field(1.5, ge=0.0, le=3600.0, description="iTime 取值")


class AnalyzeRequest(_Req):
    code: str = Field(..., min_length=1, max_length=20000)
    top_k: int = Field(3, ge=1, le=10, description="检索参考样本数量")
    with_render: bool = Field(False, description="是否同时返回原始 shader 的渲染图")
    with_reference_code: bool = Field(False, description="参考样本是否附带完整源码")


class GenerateRequest(_Req):
    description: str = Field(..., min_length=1, max_length=2000, description="中文需求描述")
    palette: str = Field("", max_length=100)
    complexity: Complexity = "simple"
    dynamic: bool = True
    effect_type: str = Field("", max_length=50)
    constraints: list[str] = Field(default_factory=list, max_length=10)
    max_fix_loops: int = Field(1, ge=0, le=3, description="编译失败后的最大修复轮数")
    top_k: int = Field(1, ge=0, le=10)
    enable_self_critique: bool = False
    with_render: bool = False


class RemixRequest(_Req):
    code: str = Field(..., min_length=1, max_length=20000)
    instruction: str = Field(..., min_length=1, max_length=2000)
    analyze_first: bool = True
    max_fix_loops: int = Field(1, ge=0, le=3)
    with_render: bool = False


class SearchRequest(_Req):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    tags: list[str] = Field(default_factory=list, max_length=8)


class FaultRequest(_Req):
    """故障注入（仅 test profile 注册该路由）。"""
    llm_mode: Optional[str] = None
    llm_fail_times: Optional[int] = Field(None, ge=0, le=10)
    llm_latency_ms: Optional[int] = Field(None, ge=0, le=10000)
    compiler_mode: Optional[str] = None
    compiler_fail_times: Optional[int] = Field(None, ge=0, le=10)
    renderer_mode: Optional[str] = None
    renderer_latency_ms: Optional[int] = Field(None, ge=0, le=10000)
    retriever_mode: Optional[str] = None
    retriever_latency_ms: Optional[int] = Field(None, ge=0, le=10000)
