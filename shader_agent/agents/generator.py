"""ShaderGenerator 角色。

工作流：
  observe(message with user text or GenerationSpec)
    → parse_spec        (若 message.payload 已是 GenerationSpec 则跳过)
    → retrieve_examples
    → draft_code
    → validate_code
    → 若失败且 iterations < max_fix_loops，回到 draft_code(prev_code+errors)
    → self_critique     (enable_self_critique=False 时跳过)
    → 输出 Message{payload=GeneratedShader}
"""
from __future__ import annotations

from typing import Any, Callable

from shader_agent.agents.actions.generator_actions import (
    DraftCodeAction,
    DraftCodeIn,
    ParseSpecAction,
    ParseSpecIn,
    RetrieveExamplesAction,
    RetrieveExamplesIn,
    SelfCritiqueAction,
    SelfCritiqueIn,
    ValidateCodeAction,
    ValidateCodeIn,
)
from shader_agent.agents.role import Role
from shader_agent.agents.schemas import (
    CompileResult,
    GeneratedShader,
    GenerationSpec,
    Message,
    SimilarShader,
)
from shader_agent.utils.logger import logger


_GENERATOR_SYSTEM_PROMPT = (
    "你是 ShaderGenerator。你的职责是根据用户的需求 spec，"
    "生成可在 Shadertoy 编辑器直接运行的 GLSL fragment shader。"
)


class ShaderGenerator(Role):
    role_name = "ShaderGenerator"
    system_prompt = _GENERATOR_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        vector_store: Any = None,
        retriever: Any = None,
        llm_fn: Callable[[list[dict[str, str]]], str] | None = None,
        compiler: Any = None,
        # 可选的多模态自评
        critique_fn: Callable | None = None,
        # 纯文本自评（无多模态也能用；分析编译错误）
        text_critique_fn: Callable | None = None,
        renderer: Any = None,               # 约定 .render(code) -> bytes(PNG)
        enable_self_critique: bool = False,
        model_name: str = "",
        max_fix_loops: int = 2,
        top_k: int = 3,
    ) -> None:
        self._vector_store = vector_store
        self._retriever = retriever
        self._llm_fn = llm_fn
        self._compiler = compiler
        self._critique_fn = critique_fn
        self._text_critique_fn = text_critique_fn
        self._renderer = renderer
        self._enable_self_critique = enable_self_critique
        self._model_name = model_name
        self._max_fix_loops = max_fix_loops
        self._top_k = top_k
        super().__init__()

    def _setup_actions(self) -> None:
        self.register_action(ParseSpecAction())
        self.register_action(RetrieveExamplesAction(
            vector_store=self._vector_store,
            retriever=self._retriever,
        ))
        self.register_action(DraftCodeAction(llm_fn=self._llm_fn))
        self.register_action(ValidateCodeAction(compiler=self._compiler))
        self.register_action(SelfCritiqueAction(
            critique_fn=self._critique_fn,
            text_critique_fn=self._text_critique_fn,
        ))

    def handle(self, message: Message) -> Message:
        self.observe(message)

        spec = self._extract_spec(message)
        if spec is None:
            r1 = self.run_action("parse_spec", ParseSpecIn(user_text=message.content))
            if not r1.ok or r1.data is None:
                err = Message(
                    role="generator",
                    content=f"无法解析用户需求: {r1.error}",
                    parent_id=message.msg_id,
                )
                self.memory.add(err)
                return err
            spec = r1.data.spec

        # retrieve examples
        examples: list[SimilarShader] = []
        if self._vector_store is not None or self._retriever is not None:
            r2 = self.run_action(
                "retrieve_examples",
                RetrieveExamplesIn(spec=spec, top_k=self._top_k),
            )
            if r2.ok and r2.data is not None:
                examples = list(r2.data.items)

        # draft + validate 循环
        prev_code = ""
        prev_errors = ""
        iterations = 0
        final_code = ""
        final_explain = ""
        # 首轮（设计/改写轮）的解释。修正轮只针对编译错误，不应覆盖它。
        design_explain = ""
        compile_result = CompileResult(ok=False, errors="not_executed")

        for i in range(self._max_fix_loops + 1):
            r3 = self.run_action(
                "draft_code",
                DraftCodeIn(
                    spec=spec,
                    examples=examples,
                    prev_code=prev_code,
                    prev_errors=prev_errors,
                ),
            )
            if not r3.ok or r3.data is None:
                # draft 失败说明 LLM 链路有问题（超时/限流/鉴权/非法输出）。
                # 按轮次分流，两种失败的正确处置不同：
                #
                # · 首轮失败：什么都没产出。不能静默降级成"空代码成品"——否则
                #   接口层拿到的是一份 code="" 的成功响应，用户以为成功了，
                #   实际什么都没有。把原始错误原样抛出，由 service 层的
                #   classify_upstream_error 归类成可操作的错误码
                #   （50401/42901/50301 等）。
                # · 修正轮失败：手里已经有一份（编不过的）代码。整体报错等于把
                #   它一起扔掉，而"如实返回 compile_ok=false + 错误原文"本来就是
                #   这个系统对反复修复失败时的既定契约，没有理由因为失败发生在
                #   LLM 侧而不是编译器侧就换一套行为。
                logger.warning(f"[generator] draft failed (round {i + 1}): {r3.error}")
                if i == 0 or not final_code:
                    raise RuntimeError(r3.error) from None
                compile_result = CompileResult(
                    ok=False,
                    errors=(f"{compile_result.errors or ''}\n"
                            f"[fix-round aborted] {r3.error}").strip(),
                )
                break
            # 轮次计数放在成功之后：中止轮没有产出，计进去会让 iterations 虚高，
            # 而这个字段是要进指标看板的。
            iterations = i + 1
            draft = r3.data
            if i == 0 and draft.explanation:
                design_explain = draft.explanation
            r4 = self.run_action("validate_code", ValidateCodeIn(code=draft.code))
            if not r4.ok or r4.data is None:
                final_code = draft.code
                final_explain = draft.explanation
                compile_result = CompileResult(
                    ok=False, errors=r4.error or "validate exception"
                )
                break
            compile_result = r4.data.result
            final_code = draft.code
            final_explain = draft.explanation
            if compile_result.ok:
                break
            # 失败 → 准备下一轮
            prev_code = draft.code
            prev_errors = compile_result.errors
            logger.info(
                f"[generator] fix loop {i+1}/{self._max_fix_loops+1}: "
                f"errors={prev_errors[:120]}"
            )

        # 解释归一化：最终始终优先用首轮的设计/改写解释（描述最后成功的成品）。
        if design_explain:
            final_explain = design_explain

        # 自评（可选）
        critique_score = 0.0
        critique_rationale = ""
        if self._enable_self_critique and final_code:
            rendered_b64 = ""
            if self._renderer is not None:
                try:
                    png = self._renderer.render(final_code)
                    if isinstance(png, (bytes, bytearray)):
                        import base64
                        rendered_b64 = base64.b64encode(png).decode("ascii")
                except Exception as e:
                    logger.warning(f"[generator] renderer failed: {e}")
            r5 = self.run_action(
                "self_critique",
                SelfCritiqueIn(
                    code=final_code, spec=spec, rendered_image_b64=rendered_b64,
                    compile_ok=bool(compile_result.ok),
                    compile_errors=compile_result.errors or "",
                ),
            )
            if r5.ok and r5.data is not None:
                critique_score = float(r5.data.score)
                critique_rationale = r5.data.rationale

        gen = GeneratedShader(
            code=final_code,
            explanation=final_explain,
            spec=spec,
            compile_result=compile_result,
            iterations=iterations,
            references_used=examples,
            self_critique_score=critique_score,
            self_critique_rationale=critique_rationale,
            model_used=self._model_name,
        )
        out = gen.to_message(parent_id=message.msg_id)
        self.memory.add(out)
        return out

    @staticmethod
    def _extract_spec(message: Message) -> GenerationSpec | None:
        if message.payload_type == GenerationSpec.PAYLOAD_TYPE and message.payload:
            try:
                return GenerationSpec(**message.payload)
            except Exception:
                return None
        return None
