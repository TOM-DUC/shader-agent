"""ShaderAnalyzer 角色。

两种策略（由 strategy 参数选择）：
  - "single"     : 单 ExplainShaderAction，一次 LLM 调用产出讲解
  - "fourstage"  : 四段式（walkthrough → summary → effect → compare），默认策略

公共流水线：
  observe(message with code)
    → parse_shader            (静态)
    → retrieve_similar        (混合检索 / 向量检索)
    → [strategy 分支]
    → synthesize_report
    → 输出 Message{payload=AnalysisReport}
"""
from __future__ import annotations

from typing import Any, Callable, Literal

from shader_agent.agents.actions.analyzer_actions import (
    ExplainShaderAction,
    ExplainShaderIn,
    ExplainShaderOut,
    ParseShaderAction,
    ParseShaderIn,
    ParseShaderOut,
    RetrieveSimilarAction,
    RetrieveSimilarIn,
    SynthesizeReportAction,
    SynthesizeReportIn,
)
from shader_agent.agents.actions.analyzer_actions_v2 import (
    CompareAction, CompareIn,
    EffectInferAction, EffectInferIn,
    SummaryAction, SummaryIn,
    WalkthroughAction, WalkthroughIn,
)
from shader_agent.agents.role import Role
from shader_agent.agents.schemas import (
    AnalysisReport,
    Message,
    SimilarShader,
)
from shader_agent.utils.logger import logger


_ANALYZER_SYSTEM_PROMPT = (
    "你是 ShaderAnalyzer。你的职责是阅读 Shadertoy 风格的 GLSL fragment shader，"
    "并产出一份准确、结构化、可被另一个 Agent 直接消费的分析报告。"
    "你必须客观、保守，不要捏造代码里不存在的实现细节。"
)


Strategy = Literal["single", "fourstage"]


class ShaderAnalyzer(Role):
    role_name = "ShaderAnalyzer"
    system_prompt = _ANALYZER_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        vector_store: Any = None,
        retriever: Any = None,
        llm_fn: Callable[[list[dict[str, str]]], str] | None = None,
        # 允许为 4 段单独指定 llm_fn（例如让 summary 用 reasoner，
        # walkthrough 用 chat）。任一为 None 则回退到 llm_fn
        walkthrough_llm: Callable | None = None,
        summary_llm: Callable | None = None,
        effect_llm: Callable | None = None,
        compare_llm: Callable | None = None,
        model_name: str = "",
        top_k: int = 3,
        strategy: Strategy = "fourstage",
        parallel: bool = True,
    ) -> None:
        self._vector_store = vector_store
        self._retriever = retriever
        self._llm_fn = llm_fn
        self._walkthrough_llm = walkthrough_llm or llm_fn
        self._summary_llm = summary_llm or llm_fn
        self._effect_llm = effect_llm or llm_fn
        self._compare_llm = compare_llm or llm_fn
        self._model_name = model_name
        self._top_k = top_k
        self._strategy = strategy
        # 把四段中相互独立的环节并行化（见 _run_fourstage）。
        self._parallel = parallel
        super().__init__()

    def _setup_actions(self) -> None:
        self.register_action(ParseShaderAction())
        self.register_action(RetrieveSimilarAction(
            vector_store=self._vector_store,
            retriever=self._retriever,
        ))
        # 单 explain（策略 single 使用）
        self.register_action(ExplainShaderAction(llm_fn=self._llm_fn))
        # 四段式
        self.register_action(WalkthroughAction(llm_fn=self._walkthrough_llm))
        self.register_action(SummaryAction(llm_fn=self._summary_llm))
        self.register_action(EffectInferAction(llm_fn=self._effect_llm))
        self.register_action(CompareAction(llm_fn=self._compare_llm))
        self.register_action(SynthesizeReportAction())

    # ---------- 主入口 ----------
    def handle(self, message: Message) -> Message:
        self.observe(message)
        code = self._extract_code(message)
        if not code.strip():
            err = Message(role="analyzer",
                         content="未能从输入中提取到 shader 代码。",
                         parent_id=message.msg_id)
            self.memory.add(err)
            return err

        # 1. parse
        r1 = self.run_action("parse_shader", ParseShaderIn(code=code))
        parse_out: ParseShaderOut = r1.data if r1.ok and r1.data else ParseShaderOut()

        # 2. retrieve
        similar: list[SimilarShader] = []
        if self._vector_store is not None or self._retriever is not None:
            # 用静态解析出的函数名/内置变量做标签线索，提升标签匹配度
            want_tags = self._infer_tags(parse_out)
            r2 = self.run_action(
                "retrieve_similar",
                RetrieveSimilarIn(code=code, top_k=self._top_k, want_tags=want_tags),
            )
            if r2.ok and r2.data is not None:
                similar = list(r2.data.items)

        # 3. 策略分支
        if self._strategy == "single":
            explain = self._run_single_explain(code, parse_out, similar)
            comparison = ""
        else:
            explain, comparison = self._run_fourstage(code, parse_out, similar)

        # 4. synthesize
        r_syn = self.run_action(
            "synthesize_report",
            SynthesizeReportIn(
                code=code,
                explain=explain,
                similar=similar,
                model_used=self._model_name,
            ),
        )
        report: AnalysisReport = r_syn.data if r_syn.ok and r_syn.data else AnalysisReport(
            source_code=code, algorithm_summary="(synthesize failed)"
        )

        # 把 comparison 挂到 section_walkthrough 的特殊键
        if comparison:
            report.section_walkthrough = dict(report.section_walkthrough or {})
            report.section_walkthrough["对照参考样本"] = comparison

        out = report.to_message(parent_id=message.msg_id)
        self.memory.add(out)
        return out

    # ---------- 策略 single ----------
    def _run_single_explain(
        self,
        code: str,
        parse_out: ParseShaderOut,
        similar: list[SimilarShader],
    ) -> ExplainShaderOut:
        r = self.run_action(
            "explain_shader",
            ExplainShaderIn(code=code, parse_result=parse_out, similar=similar),
        )
        if r.ok and r.data:
            return r.data
        logger.warning(f"[analyzer] single explain failed: {r.error}")
        return ExplainShaderOut(algorithm_summary="(explain failed)",
                                techniques=["2d-pattern"])

    # ---------- 策略 fourstage ----------
    def _run_fourstage(
        self,
        code: str,
        parse_out: ParseShaderOut,
        similar: list[SimilarShader],
    ) -> tuple[ExplainShaderOut, str]:
        """跑四段，结果聚合成 ExplainShaderOut + comparison 文本。

        依赖关系：
            walkthrough ─┐
                         ├─► summary ─► {effect, compare}
        其中 effect 与 compare 只依赖 summary，彼此独立，可并行。
        当 self._parallel=True 时：
            Round1: walkthrough ∥ summary（summary 不等 walkthrough，
                    用代码直接产出摘要，牺牲极少上下文换取一半延迟）
            Round2: effect ∥ compare
        把原本 4 次串行 LLM 往返压缩为 2 轮，分析耗时约减半。
        """
        if self._parallel:
            return self._run_fourstage_parallel(code, parse_out, similar)
        return self._run_fourstage_serial(code, parse_out, similar)

    # ---------- 并行版（默认） ----------
    def _run_fourstage_parallel(
        self,
        code: str,
        parse_out: ParseShaderOut,
        similar: list[SimilarShader],
    ) -> tuple[ExplainShaderOut, str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as ex:
            # Round 1: walkthrough ∥ summary（summary 先不带 walkthrough 上下文）
            f_wk = ex.submit(
                self.run_action, "walkthrough",
                WalkthroughIn(code=code, parse_result=parse_out),
            )
            f_sum = ex.submit(
                self.run_action, "summary",
                SummaryIn(code=code, parse_result=parse_out, walkthrough={}),
            )
            r_wk = f_wk.result()
            r_sum = f_sum.result()

        wk = r_wk.data if r_wk.ok and r_wk.data else None
        walkthrough = wk.walkthrough if wk else {}
        key_vars = wk.key_variables if wk else {}

        sm = r_sum.data if r_sum.ok and r_sum.data else None
        summary = sm.algorithm_summary if sm else ""
        techniques = sm.techniques if sm else []

        with ThreadPoolExecutor(max_workers=2) as ex:
            # Round 2: effect ∥ compare（都只依赖 summary）
            f_ef = ex.submit(
                self.run_action, "effect_infer",
                EffectInferIn(code=code, parse_result=parse_out, summary=summary),
            )
            f_cm = ex.submit(
                self.run_action, "compare",
                CompareIn(code=code, summary=summary, similar=similar),
            )
            r_ef = f_ef.result()
            r_cm = f_cm.result()

        ef = r_ef.data if r_ef.ok and r_ef.data else None
        visual_effect = ef.visual_effect if ef else ""
        cm = r_cm.data if r_cm.ok and r_cm.data else None
        comparison = cm.comparison if cm else ""

        explain = ExplainShaderOut(
            algorithm_summary=summary,
            key_variables=key_vars,
            techniques=techniques,
            visual_effect=visual_effect,
            section_walkthrough=walkthrough,
        )
        return explain, comparison

    # ---------- 串行版（向后兼容；parallel=False 时使用） ----------
    def _run_fourstage_serial(
        self,
        code: str,
        parse_out: ParseShaderOut,
        similar: list[SimilarShader],
    ) -> tuple[ExplainShaderOut, str]:
        # 3a. walkthrough
        r_wk = self.run_action(
            "walkthrough",
            WalkthroughIn(code=code, parse_result=parse_out),
        )
        wk = r_wk.data if r_wk.ok and r_wk.data else None
        walkthrough = wk.walkthrough if wk else {}
        key_vars = wk.key_variables if wk else {}

        # 3b. summary
        r_sum = self.run_action(
            "summary",
            SummaryIn(code=code, parse_result=parse_out, walkthrough=walkthrough),
        )
        sm = r_sum.data if r_sum.ok and r_sum.data else None
        summary = sm.algorithm_summary if sm else ""
        techniques = sm.techniques if sm else []

        # 3c. effect_infer
        r_ef = self.run_action(
            "effect_infer",
            EffectInferIn(code=code, parse_result=parse_out, summary=summary),
        )
        ef = r_ef.data if r_ef.ok and r_ef.data else None
        visual_effect = ef.visual_effect if ef else ""

        # 3d. compare
        r_cm = self.run_action(
            "compare",
            CompareIn(code=code, summary=summary, similar=similar),
        )
        cm = r_cm.data if r_cm.ok and r_cm.data else None
        comparison = cm.comparison if cm else ""

        explain = ExplainShaderOut(
            algorithm_summary=summary,
            key_variables=key_vars,
            techniques=techniques,
            visual_effect=visual_effect,
            section_walkthrough=walkthrough,
        )
        return explain, comparison

    # ---------- 工具 ----------
    @staticmethod
    def _infer_tags(parse_out: ParseShaderOut) -> list[str]:
        """从静态解析结果粗推技术标签，作为检索时的标签匹配线索。"""
        funcs = " ".join(parse_out.custom_functions).lower()
        tags: list[str] = []
        if "march" in funcs or "raymarch" in funcs:
            tags.append("raymarching")
        if any(f.lower().startswith("sd") for f in parse_out.custom_functions):
            tags.append("sdf")
        if "noise" in funcs or "fbm" in funcs or "hash" in funcs:
            tags.append("noise")
        if "normal" in funcs or "light" in funcs:
            tags.append("lighting")
        if "iTime" in parse_out.used_builtins:
            tags.append("animation")
        return tags

    @staticmethod
    def _extract_code(message: Message) -> str:
        if isinstance(message.payload, dict):
            c = message.payload.get("code")
            if isinstance(c, str) and c.strip():
                return c
        return message.content or ""
