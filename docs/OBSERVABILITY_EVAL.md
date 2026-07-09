# 可观测性与评估（Langfuse + DeepEval）

本文档说明本项目如何把"能跑"变成"可量化、可回归"。

## 为什么要加这两层

在此之前，系统的质量只能靠人眼看渲染图判断，存在三个问题：

1. **看不见过程**。一次生成背后是 1~3 次 LLM 调用、1 次混合检索、0~2 轮编译修正，
   耗时和 token 花在哪一步不清楚。
2. **说不出好坏**。改了检索权重或提示词之后，"变好了"只是主观感受，没有数字。
3. **防不住回退**。改 A 修好了，B 悄悄坏了，直到演示时才发现。

Langfuse 解决第 1 个问题（**过程可见**），DeepEval 解决第 2、3 个问题
（**结果可量化、回归可拦截**）。两者通过 `trace_id` 打通：评估分数直接挂回
产生它的那一次真实运行。

## 架构

```
                    ┌──────────────────────────────────────┐
   用户请求 / golden │  Orchestrator  (root span = 1 trace) │
                    └───────────────┬──────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      span: retrieval        span: action.*         span: action.*
      （融合分拆解）          （每个 Action）         （每个 Action）
              │                     │
              │                     ▼
              │           generation: deepseek.chat[model]
              │           （token / 延迟 / 成本，由 langfuse.openai 自动采集）
              │
              └──────────────► 命中的 shader_id + vec/bm25/tag/quality 分量

                                    │
                                    ▼
                          DeepEval 计算指标
                                    │
                    score_trace_by_id(trace_id, metric, value)
                                    │
                                    ▼
                   Langfuse 看板：质量分 与 耗时/token 同视图对比
```

### 埋点位置

| 位置 | 观测类型 | 记录内容 |
|---|---|---|
| `agents/orchestrator.py` | root span | 任务名、端到端耗时、compile_ok、iterations |
| `ui/runners.py::run_generate` | root span | UI 侧生成任务（不经 orchestrator） |
| `agents/actions/base.py::Action.run` | span | 每个 Action 的输入摘要、耗时、成败 |
| `corpus/retriever.py::retrieve` | span | 命中 shader、融合分及 vec/bm25/tag/quality 四路分量 |
| `llm/deepseek_client.py` | generation | 模型、prompt、completion、token、延迟、成本 |

LLM 调用不需要手写埋点：`observability/langfuse_client.py` 会把底层 OpenAI 客户端
换成 `langfuse.openai.OpenAI`（drop-in），自动记录为 generation。

### 并行分析的上下文传播

Analyzer 用 `ThreadPoolExecutor` 把四段式压成两轮。OpenTelemetry 的上下文是
`contextvar`，**不会自动跨线程传播**，不处理的话四个子 span 会各自开成孤立 trace。
`observability.bind_current_context()` 在提交前捕获上下文、在 worker 线程内 attach，
使并行子 span 正确挂回父 trace。

## 零侵入与降级

这是接入的第一原则：**未安装 langfuse / deepeval，或未配置密钥时，现有链路行为完全不变。**

| 缺失项 | 行为 |
|---|---|
| 未装 `langfuse` | `get_openai_class()` 返回原生 `openai.OpenAI`；所有 span/score 变 no-op |
| 装了但没配 `LANGFUSE_PUBLIC_KEY` | `observability.enabled=auto` 判定为关闭，全链路 no-op |
| `observability.enabled=off` | 强制关闭，不产生任何 langfuse 副作用 |
| 未装 `deepeval` | 评估仍可运行，只跑确定性指标，报告照常产出 |
| 无 `DEEPSEEK_API_KEY` | LLM 裁判指标自动跳过 |

`trace_span` 有一条硬性约束：**绝不吞掉业务异常**。span 只负责记录，异常原样向上传播
（`tests/test_observability.py` 中有回归测试锁死这一行为）。

## 评估指标设计

指标刻意分成两层。着色器有**客观可验证的正确性**——能不能编译、有没有 `mainImage`、
有没有引用 `iChannel0`——这类事实不该交给 LLM 去猜。

### 第一层：确定性指标（零 LLM 成本、零方差、可进 CI）

| 指标 | 含义 |
|---|---|
| `Compile Success` | GLSL 通过编译/静态校验（0/1） |
| `Shadertoy Convention` | 五项硬约束：入口签名、无外部纹理、无自定义 uniform、WebGL1 兼容、括号配平 |
| `Fix Loop Efficiency` | 修正轮数效率，一次成功=1.0 |
| `Retrieval Relevancy` | 越过融合分阈值的命中占比 × 平均融合分 |
| `Negative Rejection` | 负样例（无关 query）是否正确地**不返回**参考 |

修正轮数效率的定义：

\[
s = \max\left(0,\ 1 - \frac{it - 1}{\text{max\_loops}}\right)
\]

检索相关性的定义（命中率与平均分各占一半，避免"只有一条高分"或"全是擦线低分"两种极端）：

\[
s = 0.5 \cdot \frac{|\{i : f_i \ge \tau\}|}{n} + 0.5 \cdot \min(1,\ \bar{f})
\]

其中 \(f_i\) 是第 \(i\) 条命中的融合分，\(\tau\) 是 `retrieval.min_score` 阈值。

### 第二层：LLM-as-a-judge（GEval，DeepSeek 评审）

| 指标 | 评什么 |
|---|---|
| `Spec Adherence` | 代码是否落实了 effect_type / palette / dynamic / complexity |
| `Explanation Faithfulness` | 中文解释是否忠实于代码，有无臆造未实现的技术 |
| `Analysis Faithfulness` | 分析报告是否忠实于源码，有无捏造实现细节 |
| `Retrieval Context Relevancy` | 检索到的参考对完成需求是否真的有帮助 |

评审模型走 `eval/judge_model.py` 的 `DeepSeekJudge`，它把 deepeval 的评审请求
转接到项目自带的 DeepSeek 客户端，从而复用**重试、磁盘缓存、超时、统计**，
并让评审调用本身也被 Langfuse 记录。**不需要 `OPENAI_API_KEY`**。

GEval 一律用 `evaluation_steps` 而非 `criteria`：给定步骤比让模型自己推导评分标准
方差更小、更可复现。

## 黄金数据集

`eval/datasets.py`，小而精，每条对应一类**失败模式**而非同质样例堆砌：

- 生成（5 条）：覆盖 raymarching / noise / fractal / 2d-pattern / post-processing 五种
  effect_type，外加两个边界样例——`dynamic=false`（静态）与显式硬约束（"不要用纹理"）。
- 分析（2 条）：用内置 seed shader 作源码，完全离线。
- 检索（4 条）：含一条**负样例**（问 Python CSV），理想行为是阈值生效、不返回参考。

## 使用

### 安装与配置

```bash
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 填入 Langfuse 密钥（可选，不填则 tracing 降级为 no-op）：

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # 自托管改成你的地址
```

### 验证 tracing 链路

```bash
python -m scripts.verify_observability --dry-run   # 不调 LLM
python -m scripts.verify_observability             # 真跑一次 LLM 调用
```

### 跑评估

```bash
# 只跑确定性指标：零 LLM 成本、零方差，适合 CI
python -m scripts.run_eval --no-judge

# 只评检索链路（最快，不调 LLM 生成）
python -m scripts.run_eval --tasks retrieval

# 全量（含 LLM 裁判，会真调 DeepSeek）
python -m scripts.run_eval

# 快速冒烟：每类只跑前 2 条
python -m scripts.run_eval --limit 2

# CI 门禁：通过率低于 0.8 则退出码为 1
python -m scripts.run_eval --no-judge --min-pass-rate 0.8
```

产物：`data/reports/eval_{ts}/report.md` 与 `report.json`。

### 消融实验：RAG 到底有没有用

`--no-vector-store` 关掉检索，与默认跑一次对比，即可用数字回答
"混合检索对生成质量的贡献有多大"：

```bash
python -m scripts.run_eval --tasks generation --name with_rag
python -m scripts.run_eval --tasks generation --no-vector-store --name no_rag
```

对比两份 `report.md` 的 `Spec Adherence` 与 `Compile Success` 均值即可。

## CI 集成建议

```yaml
- name: Offline eval gate
  run: python -m scripts.run_eval --tasks retrieval --no-judge --min-pass-rate 0.8
```

只跑确定性指标 + 检索任务，不需要 API key、不消耗 token、耗时秒级，
却能拦住"检索权重改错导致召回崩掉"这类回退。
