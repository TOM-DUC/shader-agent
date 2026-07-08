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
  config/        配置入口，集中管理 LLM、检索、记忆等参数
  llm/           DeepSeek 客户端与 llm_fn 装配
  corpus/        知识库：模型、清洗、打标、静态分析、分块、向量库、关键词库、父文档库、混合检索、重排
  embeddings/    本地嵌入模型封装
  agents/        Role、Action、工作记忆（WorkingMemory）、Analyzer、Generator、Orchestrator
  rendering/     headless GLSL 编译与渲染
  ui/            Gradio 三标签页界面与运行装配
  utils/         日志等工具
scripts/         建库、下载嵌入模型、各模块验证脚本、UI 启动
tests/           离线单元测试
config.yaml      非敏感配置
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


## 配置要点

config.yaml 集中管理非敏感配置。llm 段配置对话与代码模型、生成参数与重试。corpus 段配置数据源与清洗阈值。embedding 段配置本地嵌入模型。vector_store 段配置集合名与距离度量。retrieval 段配置混合检索的召回规模、融合权重、阈值与重排开关。memory 段配置记忆数据库路径、集合名、晋升门槛、注入上限与经验排序权重。

敏感字段如 API key 放在 .env，不进版本库。

## 测试

```bash
pytest
```

测试以离线为主，覆盖数据模型、清洗、打标、Action、Role、编排、渲染与界面运行层，默认不触发网络与真实大模型调用。

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
