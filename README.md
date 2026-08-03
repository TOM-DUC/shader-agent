# Shader Agent

<p>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/status-beta-yellow" alt="Status: Beta">
  <img src="https://img.shields.io/badge/LLM-DeepSeek-red" alt="LLM: DeepSeek">
</p>

基于 DeepSeek 大模型的 Shadertoy 智能助手。它能解读现成的 GLSL 片元着色器、按自然语言需求生成新的着色器、在现有代码基础上做最小化改写，并通过混合检索知识库（向量 + BM25 + 重排 + 融合）与工作记忆，让生成质量随使用不断提升。

本项目强调的是工程化的检索增强生成（RAG）系统设计，而非堆砌重型框架。整个系统保持轻量，没有引入 Neo4j、Elasticsearch、Celery 或外部 Agent 框架，所有能力都建立在可控、可解释、可降级的组件之上。

## 截图

> 🖼️ 截图待补充 — 运行 `python -m scripts.run_ui` 后即可在浏览器中看到界面。

| 标签页 | 功能 |
|--------|------|
| **Analyzer** | 解读现成着色器，展示分段讲解与对照参考样本 |
| **Generator** | 按需求生成新着色器，展示检索参考与用户反馈入口 |
| **Remixer** | 基于原代码改写，支持改写链与采用反馈 |

## 核心特性

着色器解读。读入一段 Shadertoy 风格的片元着色器，输出结构化分析报告，包含算法摘要、关键变量、技术标签、视觉效果推断、分段讲解，以及从知识库检索到的对照参考样本。

按需生成。把自然语言需求解析成结构化的生成规格，检索相似的高质量参考样本作为上下文，起草代码后进入编译校验循环，必要时自动修正，最后可选地做一次自评。

代码改写。在原始代码基础上做最小化修改，默认不引入外部参考，保留用户要求保留的部分。改写会形成一条改写链，每一轮都作为独立情节记录，并通过父链与上一轮相连。

混合检索知识库。把每个着色器拆成父子知识块，子块用于细粒度检索，命中后回溯完整着色器。检索同时走向量召回与 BM25 关键词召回，再叠加标签匹配度与质量分做融合排序，可选交叉编码器精排，最后用相关度阈值过滤，宁缺毋滥。

## 架构总览

数据侧的知识库构建流程：外部着色器数据经过清洗、静态分析与质量验证，切分成父子知识块，分别写入向量库、BM25 关键词索引与父文档存储。检索时三路（向量 + BM25 + 标签/质量融合）结果排序，产出可信的参考内容。

运行侧的任务流程：用户任务由 Analyzer、Generator、Remixer 协作完成，经过编译与渲染验证。系统检索知识库获取相关参考样本，注入 LLM 上下文辅助生成与解读。

## 目录结构

```
shader_agent/
  config/        配置入口，集中管理 LLM、检索、记忆、可观测性等参数
  llm/           DeepSeek 客户端与 llm_fn 装配
  corpus/        知识库：模型、清洗、打标、静态分析、分块、向量库、关键词库、父文档库、混合检索、重排
  embeddings/    本地嵌入模型封装
  agents/        Role、Action、工作记忆（WorkingMemory）、Analyzer、Generator、Orchestrator
  observability/ Langfuse 可观测性：tracing 客户端、span 辅助
  rendering/     headless GLSL 编译与渲染
  ui/            Gradio 三标签页界面与运行装配
  utils/         日志等工具
scripts/         建库、下载嵌入模型、各模块验证脚本、UI 启动
tests/           离线单元测试
config.yaml      非敏感配置
.env.example     环境变量模板
requirements.txt 依赖清单
```

## 快速开始

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux 或 macOS 下用 source .venv/bin/activate 激活环境。

配置 API key：

```bash
cp .env.example .env
```

编辑 .env，填入真实的 `DEEPSEEK_API_KEY`。如果有 Shadertoy API key，可一并填入 `SHADERTOY_API_KEY`，没有也能用内置的种子着色器跑通全流程。

> ⚠️ `.env` 文件包含敏感密钥，已默认被 `.gitignore` 排除，**切勿**提交到版本库。

下载本地嵌入模型（首次较慢，模型约 2GB）：

```bash
python -m scripts.download_embedder
```

构建知识库。完全离线只用种子着色器即可冒烟：

```bash
python -m scripts.build_corpus --no-api
```

建库会依次完成数据收集、清洗去重、主题打标、静态分析与质量评分，然后写入向量库、子块向量、BM25 关键词索引与父文档存储，最后跑一轮混合检索冒烟测试，打印每条查询命中的着色器及其融合分构成。

启动界面：

```bash
python -m scripts.run_ui
```

## 混合检索说明

检索粒度是子块。一个着色器被拆成 overview、structure、algorithm、每个自定义函数、以及完整代码摘要等多个子块。查询如何求法线时，命中的是具体的 calcNormal 函数子块，而不是整条着色器的平均语义。

关键词检索采用面向 GLSL 的分词器配合 BM25。通用英文分词器会把 sdSphere、calcNormal 当成单个不可分的词，导致函数名片段几乎无法命中。本项目的分词器会拆分驼峰与下划线，同时保留原始整词与 Shadertoy 内置变量，使整词命中与片段命中都能加分。选择内存 BM25 而非 SQLite 全文检索，是因为语料规模在数十到数百条，无需额外数据库文件，且对代码标识符的检索质量更高。

融合排序按可配置权重组合四个信号，默认是向量相关度占一半，关键词、标签匹配度、质量分各占其余。融合分低于阈值时不返回参考。重排器优先使用交叉编码器，当依赖缺失、无网络或无显存时自动降级为基于融合分与词重叠的确定性排序，保证全链路在任何环境下都能跑通。

所有这些权重、阈值与开关都集中在 config.yaml 的 retrieval 段，可不改代码直接调参。


## 可观测性（Langfuse）

本项目通过 Langfuse 对每次任务生成完整 trace 链路：

```
一次生成请求
├── root span: 整个任务耗时
│   ├── span: 混合检索（记录融合分四路分量）
│   ├── span: DraftCode 起草
│   ├── span: CompileFix 编译修正
│   └── generation: LLM 调用（自动记录 token/延迟/成本）
```

每次 Action 执行、检索调用和 LLM 请求都会自动记录。未安装 Langfuse 或未配置密钥时全链路自动降级为 no-op，不影响现有功能。

配置方式（可选，不填则 no-op）：

```bash
# 在 .env 中填入
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

验证链路：

```bash
python -m scripts.verify_observability --dry-run
```

## 配置要点

config.yaml 集中管理非敏感配置。llm 段配置对话与代码模型、生成参数与重试。corpus 段配置数据源与清洗阈值。embedding 段配置本地嵌入模型。vector_store 段配置集合名与距离度量。retrieval 段配置混合检索的召回规模、融合权重、阈值与重排开关。observability 段配置 tracing 开关与环境标签。

敏感字段如 API key 放在 .env，不进版本库。

## 测试

```bash
pytest
```

测试以离线为主，覆盖数据模型、清洗、打标、Action、Role、编排、渲染与界面运行层，默认不触发网络与真实大模型调用。

## 自动化测试与质量保障体系

在原有业务之上，项目新增了一套完整的质量保障体系：HTTP API 层、确定性测试替身、六层校验、故障注入、CI 门禁与性能基线，让 AI 应用可以像普通后端服务一样被自动化测试。

### 三态运行 Profile

装配层（`shader_agent/service/assembly.py`）把"用真 LLM 还是桩、用真 GL 还是软件渲染、用向量库还是内存语料"统一为环境决策：

| profile | LLM | 编译/渲染 | 检索 | 用途 |
| --- | --- | --- | --- | --- |
| `real` | 真实 DeepSeek | 真 GL，不可用即报错 | ChromaDB + bge-m3 | 生产 / 预发自检 |
| `auto` | 真实，缺 key 则降级 | 真 GL，不可用降级 mock | 有则用，无则降级 | 本地开发（默认） |
| `test` | 确定性桩 | 确定性软件渲染 | 6 条内存语料 | **CI 与自动化测试** |

`test` profile 下无需 API Key、无需 GPU、无需下载模型权重，单条用例毫秒级。确定性替身位于 `shader_agent/testing/`：`stub_llm`（输出由 prompt 决定，可断言"要蓝色就必须真的是蓝色"）、`fake_render`（numpy 复算等价图像，支持感知哈希回归）、`fake_retriever`（6 条内存语料，打分规则确定）、`faults`（故障注入开关）。

### 六层校验体系

一条用例只做到它该做的那一层，成本从低到高：

| 层 | 内容 | 位置 |
| --- | --- | --- |
| L0 配置层 | 凭据可选性、环境隔离、配置报错可读性 | `tests/test_settings_config.py` |
| L1 协议层 | 统一信封 `{code, message, request_id, elapsed_ms, data}`、错误码、request_id | `tests_api/utils/assertions.py` |
| L2 契约层 | JSON Schema：字段名 / 类型 / 必填 | `tests_api/data/schemas/*.json` |
| L3 规则层 | GLSL 静态规则（与服务的 `glsl_rules.py` 独立实现，互为交叉验证） | `tests_api/utils/glsl_checker.py` |
| L4 编译层 | GLSL 真的能不能编过（生成接口自报的 `compile_ok` 与独立 `/compile` 复核一致） | `tests_api/testcases/*` |
| L5 图像层 | PNG 可解码、像素统计防全黑、主色通道、感知哈希回归与动画帧差异 | `tests_api/utils/image_checker.py` |

### 接口与错误码契约

- 业务门面 `ShaderService`（`shader_agent/service/shader_service.py`）是 UI、HTTP、测试共用的唯一业务入口；装配逻辑独立为 `service/assembly.py`。
- 新增 FastAPI 接口层（`shader_agent/api/`）：`/healthz`、`/readyz`、`/api/v1/meta`、`/api/v1/shader/validate|compile|render|analyze|generate|remix`、`/api/v1/retrieval/search`，以及仅 test profile 注册的故障注入路由 `/api/v1/_test/faults`。
- 所有接口返回统一信封；错误码是稳定契约（`40001~40005` 参数/输入、`42901` 限流、`50002/50003` 模型/生成、`50301/50302/50303` LLM/渲染/检索不可用、`50401` 上游超时），HTTP 状态码只做粗分类。
- **一条设计约定**：编译不通过、检索为空属于业务结果，返回 `200 + code=0`，结果放在 `data` 里；只有调用方用错接口或依赖挂了才返回非 0。

### 故障注入矩阵

通过 `faults` fixture 模拟外部依赖异常，每条用例后自动复位：

- LLM 超时 → `50401/504`；限流 → `42901/429`；鉴权失败 → `50301/503`；超时后重试自愈
- LLM 吐非法 JSON → 降级到静态解析不崩；产出编不过的代码 → 修复循环救回且 `iterations` 如实记录
- 编译器持续失败 → **如实**返回 `compile_ok=false` + 错误原文（不许假装成功）
- 渲染器不可用 → 渲染接口 503，但生成主流程照常返回代码；输出全黑帧 → 图像层断言必须抓到
- 检索为空 → 退化为无参考生成；检索报错 → 检索接口如实返回 `50303`

### 测试框架

- **API Object 封装**（`tests_api/api_objects/`）：用例里不出现 URL、不出现 httpx；同一套用例既可跑进程内 ASGI（默认，不占端口），也可打已部署实例（`pytest tests_api --base-url=http://host:8000`）。
- **YAML 数据驱动**（`tests_api/data/*_cases.yaml`）：新增一条边界用例只加几行 YAML，`case_id` 直接作为 pytest 用例名。
- **CI 分层触发**（`.github/workflows/quality.yml`）：`config`（配置守护，最快）→ `smoke`（冒烟）→ `api-tests`（全量 + 并行 + 有限重试 + 覆盖率 + Allure 报告）→ nightly 性能压测（Locust，失败率 >2% 或 P95 >5s 即判红）。
- 覆盖率配置见 `.coveragerc`，pytest 配置见 `pytest.ini`（marker：`config` / `smoke` / `api` / `contract` / `fault` / `e2e` / `image` / `gpu` / `live`）。

### 常用命令

```bash
# 环境体检：不起服务、不调模型，30 秒内回答"这台机器缺什么"（退出码可直接当 CI 门禁）
python -m shader_agent.config.doctor

# 配置层守护用例（秒级，不依赖任何服务）
pytest tests -m config

# 冒烟
pytest tests_api -m smoke

# 全量接口自动化（test profile，无需 key/GPU）
pytest tests_api

# 只跑故障注入
pytest tests_api -m fault

# 启动 HTTP API（test profile）
python -m uvicorn shader_agent.api.main:app --port 8000
```

## 设计取舍

为什么不用更重的组件。语料规模不大，关系不复杂，单机 SQLite 加 Chroma 加内存 BM25 已足够，引入图数据库或搜索引擎只会增加部署与维护成本。

为什么检索做成可降级。没有 GPU的情况，交叉编码器与大嵌入模型未必跑得起来。混合检索器在子块库、关键词索引或父文档表任一缺失时，会自动退回到着色器级向量检索，保证基本可用。

为什么用父子分块而非整段向量化。每条 shader 拆成 overview / structure / algorithm / 函数级子块，检索"如何求法线"时命中具体函数块而非整段平均语义，精度更高。子块命中后通过父文档表回溯完整代码，兼顾粒度与完整性。

## 贡献

欢迎贡献！无论是 Bug 报告、功能建议还是代码提交，请先开 issue 讨论，再提交 PR。

- 代码风格：遵循项目现有风格，保持轻量可读
- 测试：新增功能应包含对应测试
- 提交信息：使用 `feat:` / `fix:` / `refactor:` / `docs:` 前缀

## 许可

本项目基于 [MIT License](LICENSE) 开源。
