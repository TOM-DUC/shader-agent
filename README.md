# Shader Agent

基于 DeepSeek LLM 的 Shadertoy 智能教学助手。

## 项目结构

```
Shader_agent/
├── gpt-researcher/              # 参考项目：GPT Researcher
├── MetaGPT/                     # 参考项目：MetaGPT
├── desktop-shadertoy/           # 参考项目：Desktop Shadertoy
├── shader_agent/                # 核心代码包
│   ├── config/                  # 配置管理
│   ├── llm/                     # LLM 客户端
│   ├── corpus/                  # Shadertoy 语料库（阶段二）
│   ├── embeddings/              # 本地嵌入模型（阶段二）
│   ├── agents/                  # Role / Action / Memory / Orchestrator（阶段三）
│   ├── tools/                   # 工具集（阶段四）
│   └── utils/                   # 工具函数
├── scripts/                     # 验证和辅助脚本
├── tests/                       # 测试
├── data/                        # 数据目录
├── logs/                        # 日志目录
├── requirements.txt             # 依赖
├── config.yaml                  # 非敏感配置
├── .env.example                 # 环境变量模板
└── README.md
```

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt

# 配置 API key
cp .env.example .env
# 编辑 .env，填入真实的 DEEPSEEK_API_KEY

# 验证
python -m scripts.verify_deepseek
pytest
```

## 架构决策记录（ADR）

### 阶段三编排框架：自写轻量状态机 + 角色类

**决策**：阶段三采用"自写轻量状态机 + 角色类"，参考 MetaGPT 的 `Role / Action / Message` 思想，但不直接依赖 MetaGPT 框架。

**理由**：

1. MetaGPT 把 LLM、配置、记忆都框死在自己的体系里，直接 fork 会和我们 DeepSeek 客户端 + pydantic-settings 的体系产生两套配置；
2. 项目本质只有两个 agent，状态机足够，无需引入 LangGraph 这类额外依赖；
3. 借鉴 GPT-Researcher 的 `plan → search → synthesize` 流水线作为 Analyzer 的 Action 序列，借鉴 MetaGPT 的角色消息总线思想，把两者的精华吸收为我们自己的 `Role` 基类即可。

**何时复议**：当 agent 数量超过 4 个，或需要并发协作 / 复杂回退时，再评估迁移到 MetaGPT 或 LangGraph。

---

## 阶段二：Shadertoy 知识库

### 数据流水线

```
Shadertoy API ─┐
                ├─► fetch ─► clean/dedup ─► topic-tag ─► bge-m3 embed ─► ChromaDB
seed shaders ──┘                                                            │
                                                                            ▼
                                                                Analyzer 检索增强
```

### 关键决策

- **嵌入模型用本地 bge-m3，而不是 DeepSeek embedding API**：DeepSeek 截至阶段二未提供官方 embedding API；bge-m3 多语言（中英都强）、1024 维、HF 公开权重，自托管更稳。
- **采集策略：API 优先 + seed 兜底**：用户没有 Shadertoy API key 也能跑通整条流水线，方便迭代调试。
- **打标双轨**：规则版（默认，毫秒级，覆盖 80%+）+ LLM 复核（可选，对规则未命中样本兜底）。
- **文档拼接策略**：向量化时**不灌全代码**，只拼 `name + description + tags + 代码前 1500 字符`，避免长代码淹没语义。

### 准备工作

申请并填入：

```env
# .env
SHADERTOY_API_KEY=YOUR_KEY  # 可选，留空则只用 seed
```

预下载嵌入模型（约 2.3GB，首次下载耗时）：

```bash
python -m scripts.download_embedder
```

### 构建语料库

```bash
# 默认构建（含 API 拉取，若配了 key）
python -m scripts.build_corpus

# 仅 seed（快速冒烟，约 30 秒可完）
python -m scripts.build_corpus --no-api

# 清空向量库后重建
python -m scripts.build_corpus --reset

# 启用 LLM 复核打标
python -m scripts.build_corpus --enable-llm-tagging
```

### 知识库扩容（无 API key 场景）

Shadertoy 新账号等级不足无法申请 API key 时，本项目提供三条互补扩容路径，
合并后可达 60+ 条样本，足够支撑 Analyzer/Generator 的检索 + in-context example：

| 路径 | 来源 | 命令 | 说明 |
| --- | --- | --- | --- |
| ① 扩充种子 | 内嵌 33 条手写 shader | 默认开启 | 覆盖 8 个 TOPIC_VOCAB 主题，无版权问题 |
| ② 本地导入 | `data/external_shaders/*.glsl` + `.meta.json` | `--from-local-dir auto` | 从 GitHub 开放 shader 集合手动下载放进来 |
| ③ 公开端点抓取 | Shadertoy `POST /shadertoy` | `--from-urls` / `--from-id-list` | 无需 key，速率 1.5 s/req，遵守 TOS |

```bash
# 路径 ②：从本地目录批量导入
mkdir -p data/external_shaders
# 放 .glsl 文件进去（可选配同名 .meta.json sidecar）
python -m scripts.build_corpus --from-local-dir auto

# 路径 ③：从 Shadertoy URL 列表抓取
python -m scripts.build_corpus --from-urls \
    https://www.shadertoy.com/view/XlSSRV \
    https://www.shadertoy.com/view/MdX3Rr

# 或从文本文件读 id/URL 列表
cat > data/wanted_ids.txt <<EOF
https://www.shadertoy.com/view/XlSSRV
MdX3Rr
WdSXWy
EOF
python -m scripts.build_corpus --from-id-list data/wanted_ids.txt

# 组合：seed + 本地 + 抓取（推荐用法）
python -m scripts.build_corpus --no-api \
    --from-local-dir auto \
    --from-id-list data/wanted_ids.txt
```

`.meta.json` sidecar 示例（与同名 `.glsl` 放同一目录）：

```json
{
  "name": "Volumetric Cloud",
  "description": "Single-pass volumetric cloud demo.",
  "tags_raw": ["noise", "fbm", "raymarching", "volumetric"],
  "author": "alice",
  "license": "CC-BY-NC-SA-3.0",
  "likes": 250
}
```

⚠️ **TOS 提醒**：路径 ③ 使用 Shadertoy 自家前端的非文档化端点；请严格控制速率
（默认 ≥1.5 s/req），只抓主动给出的 id，不要爬全站，不要在 CI/后台无限循环。
抓回的 shader 默认 CC-BY-NC-SA-3.0，仅限学习与非商用；公开二次分发请逐条 attribution。

### 验收

```bash
python -m scripts.verify_corpus
pytest tests/test_corpus.py -q
```

### 产物位置

```
data/
├── shadertoy_corpus/
│   ├── raw/         # 原始 Shadertoy JSON（断点续传缓存）
│   └── clean/       # 清洗 + 打标后的 ShaderRecord JSON
├── vector_db/       # ChromaDB 持久化
└── models/          # bge-m3 权重缓存
```

---

## 阶段三：Agent 骨架与共享协议

### 五件套抽象

```
Message  ←  Role 之间通用消息载体（带 payload + payload_type）
Memory   ←  每个 Role 私有的会话记忆（仅内存）
Action   ←  最小工作单元；输入/输出有 pydantic schema
Role     ←  系统提示 + Action 集合 + Memory + handle(message)
Orchestrator ← 串行调度两个 Role 完成组合任务
```

### 关键契约：Analyzer → Generator

```python
AnalysisReport  # Analyzer 产物
├── source_code           : 被分析的原始 GLSL
├── algorithm_summary     : 中文算法摘要
├── key_variables         : { var: 用途 }    ← Generator 改写时复用
├── techniques            : [tag, ...]      ← 受控词表
├── visual_effect         : 视觉效果一句话
├── section_walkthrough   : 分段讲解
└── similar_shaders       : 检索到的相似样本

GenerationSpec.reference_report = AnalysisReport   ← 组合任务的关键穿透字段
```

### Action 清单（共 8 个，本阶段定义骨架）

**Analyzer（4 个）**

| Action | 输入 | 输出 | 是否调 LLM |
|---|---|---|---|
| ParseShaderAction | code | uniforms / funcs / builtins / LOC | 否 |
| RetrieveSimilarAction | code | list[SimilarShader] | 否（用 vector_store） |
| ExplainShaderAction | code+parse+similar | summary / vars / techniques / sections | 是（阶段四接通） |
| SynthesizeReportAction | 上面三者 | AnalysisReport | 否 |

**Generator（4 个）**

| Action | 输入 | 输出 | 是否调 LLM |
|---|---|---|---|
| ParseSpecAction | 用户文本 | GenerationSpec | 否（规则） |
| RetrieveExamplesAction | spec | list[SimilarShader] | 否 |
| DraftCodeAction | spec+examples+prev_err | code + explanation | 是（阶段五接通） |
| ValidateCodeAction | code | CompileResult | 否（阶段六接真编译器） |

### 三种组合任务（由 Orchestrator 串）

```python
orch.analyze_only(code)                # 仅分析
orch.generate_only(user_text)          # 仅生成
orch.analyze_then_generate(code, ask)  # 关键路径：先分析后改写
```

### 运行与验收

```bash
# 单测（24 项）
pytest tests/test_agents.py -q

# 端到端 dry-run（用 fallback / stub，不调真 LLM 也不真编译）
python -m scripts.verify_agents
```

阶段三**有意不接 DeepSeek 与真编译器**，目的是先把骨架与协议固定下来。阶段四、五的工作就是把
`ExplainShaderAction(llm_fn=...)` 与 `DraftCodeAction(llm_fn=...)` 的 `llm_fn` 真正接到
`DeepSeekClient.chat` / `.code` 即可，不必修改任何 Role / Action 结构。

---

## 阶段四：Analyzer Agent 实现

### 四段式工作流

阶段三的单 `ExplainShaderAction` 在阶段四被拆为 4 个 Action，灵感来自 GPT-Researcher 的"plan → search → synthesize"分阶段思想——单 prompt 多目标会让模型偷懒，分段强制每步都"完成"。

```
parse → retrieve_similar
        ├─ WalkthroughAction   逐段讲解 + 关键变量  [JSON 模式]
        ├─ SummaryAction       算法摘要 + 技术标签  [JSON 模式]
        ├─ EffectInferAction   视觉效果推断       [自由文本]
        └─ CompareAction       与参考样本的对照    [自由文本]
                            ↓
                     SynthesizeReportAction → AnalysisReport
```

Analyzer 提供 `strategy` 参数：

- `"fourstage"` — 默认，阶段四的四段式
- `"single"` — 阶段三的单 prompt（向后兼容，不破坏既有测试）

### LLM 适配层

`shader_agent/llm/llm_fn.py` 提供两个工厂，把 `DeepSeekClient.chat` 包装成符合 Action 协议的 `llm_fn(messages) -> str`：

```python
from shader_agent.llm.llm_fn import make_chat_fn, make_json_fn, stats

chat_fn = make_chat_fn()                     # 普通对话
json_fn = make_json_fn(temperature=0.0)      # 强制 JSON 输出
print(stats.snapshot())                      # 调用次数 / 缓存命中 / token 估算
```

特性：
- **缓存**：基于 `(model, temperature, json_mode, messages)` 的 sha1 落盘到 `data/cache/llm/`，调试反复跑同一 shader 不烧 token；通过环境变量 `SHADER_AGENT_LLM_CACHE=0` 关闭。
- **JSON 模式**：调用 DeepSeek OpenAI 兼容接口的 `response_format={"type":"json_object"}`，避免模型加 markdown 包裹。

### 运行与验收

```bash
# 单测（39 项：24 阶段三 + 15 阶段四）
pytest tests/test_agents.py tests/test_analyzer_v2.py -q

# 端到端真调 DeepSeek
python -m scripts.verify_analyzer                 # 默认 Raymarched Sphere
python -m scripts.verify_analyzer --seed seed05   # Mandelbrot
python -m scripts.verify_analyzer --no-cache      # 强制重新调用
python -m scripts.verify_analyzer --strategy single  # 对照阶段三策略
```

质量门槛（都必须满足才算 PASS）：
- `algorithm_summary ≥ 80` 字符
- `techniques` 非空且全部在受控词表内
- `section_walkthrough ≥ 1` 条
- `similar_shaders ≥ 1`（若向量库非空）
- `visual_effect` 非空
- 报告全文无 "占位 / fallback" 字样

### 产物

`data/reports/analyzer_{seed_id}_{timestamp}.md` — 8 段式完整 Markdown 报告：Overview、视觉效果、算法摘要、分段讲解、关键变量、相似样本、对照参考样本、源码。可直接贴到简历项目页或博客。

---

## 阶段五：Generator Agent 实现

### 五步流水线

仿 MetaGPT 的 PRD → Design → Code 思想：

```
user_text → ParseSpecAction      ── 规则解析（中文 prompt → GenerationSpec）
         → RetrieveExamplesAction ── 检索相似样本作为 in-context 例子
         → DraftCodeAction        ── DeepSeek coder_model（v4-flash）写 GLSL
         → ValidateCodeAction     ── 强化静态校验
         ↑↻ 失败回到 Draft（修正轮 prompt 不同于首轮）
         → SelfCritiqueAction     ── 占位文本自评（阶段六接渲染器后启用图像自评）
         → GeneratedShader
```

### 关键设计

**首轮 vs 修正轮 prompt 分离**：阶段三的修正信息只是追加到 user 末尾；阶段五专门给修正轮换一个 `_SYSTEM_PROMPT_FIX`，告诉模型"最小改动、保留算法主干、不要重写"。这对 LLM 的修正质量有显著影响——首轮强调创意，修正轮强调修复。

**强化的 ValidateCodeAction**：
- 剥注释后再做括号配对，避免注释里的 `{` 误伤
- 检测 HLSL 误用：`lerp`→`mix`、`saturate`→`clamp`、`frac`→`fract`、`Length`→`length`
- 检测 `mainImage` 签名是否完整（必须 `out vec4 ... in vec2 ...`）
- 错误信息包含具体计数（`{ count=12 vs } count=11`），方便 LLM 定位

**模型选型**：
- `chat_model` = `deepseek-v4-pro` — Analyzer 用，质量优先
- `coder_model` = `deepseek-v4-flash` — Generator 用，速度优先（修正循环可能跑 3 次）

**多模态自评（阶段六接通）**：`SelfCritiqueAction` 已经在阶段五埋好。`enable_self_critique=False` 是默认；启用且无 `renderer/critique_fn` 时走文本层弱自评（检查代码是否提及 spec 关键词）；阶段六注入 `renderer.render(code) -> bytes(PNG)` 与 `critique_fn(code, spec, image_b64) -> str(JSON)` 即可启用真正的图像自评。

### 运行与验收

```bash
# 单测（55 项：24 阶段三 + 15 阶段四 + 16 阶段五）
pytest tests/test_agents.py tests/test_analyzer_v2.py tests/test_generator_v2.py -q

# 端到端真调 DeepSeek
python -m scripts.verify_generator                           # 默认: 霓虹蓝紫万花筒
python -m scripts.verify_generator --case raymarch_blob
python -m scripts.verify_generator --case noise_water
python -m scripts.verify_generator --case "自定义中文 prompt"
python -m scripts.verify_generator --critique                # 启用文本自评
python -m scripts.verify_generator --combined                # 走 analyze_then_generate
```

质量门槛：
- `mainImage` 签名正确
- `validate_code` 报 ok=True（含修正循环后）
- `explanation` 非空
- `iterations ≥ 1`
- `--critique` 时 score ≥ 0.5
- `--combined` 时 `gen.spec.reference_report` 不为 None

### 产物

`data/reports/generator_{case_slug}_{ts}.md` — 完整 Markdown：原始 prompt + Analyzer 报告（若 combined）+ 生成代码 + 自评。

---

## 阶段六：渲染验证闭环

### 方案选择

计划提到三条路径：

- ❌ **Desktop Shadertoy CLI**：是 winit GUI 应用，无 headless CLI，不可用
- ❌ **PyO3 包装其 wgpu 管线**：跨平台编译噩梦，简历项目不值
- ✅ **moderngl headless 渲染器**：纯 Python，pip 一行装好——本阶段选这条

### 模块结构

```
shader_agent/rendering/
├── shadertoy_wrap.py    # Shadertoy fragment → GLSL 330 完整程序的包装层
│                         #  + 行号翻译（wrap_line → user_line）
├── compiler.py          # GLSLCompiler.compile() — 只编译，~5-30ms/次
├── renderer.py          # GLSLRenderer.render()  — 编译 + 全屏 quad + PNG
└── mock.py              # MockCompiler / MockRenderer — 单测与无 GL 环境
```

### Shadertoy → GLSL 包装层

用户写：
```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }
```

实际编译的是：
```glsl
#version 330 core
in vec2 v_frag_coord;
out vec4 _out_fragColor;
uniform vec3 iResolution;  // + iTime / iTimeDelta / iFrame / iMouse / iDate / iSampleRate
// ===== USER CODE BEGIN =====
{用户代码原样}
// ===== USER CODE END =====
void main() {
    vec4 _col;
    mainImage(_col, v_frag_coord);
    _out_fragColor = _col;
}
```

编译错误里的行号会自动映射回用户代码：`0(45)` → `0(45(user:32))`，方便 LLM 修正轮定位。

### 接入阶段五留好的钩子

阶段三与阶段五的接口在阶段六**完全不变**：

```python
from shader_agent.rendering import GLSLCompiler, GLSLRenderer

compiler, _ = GLSLCompiler.try_create()
renderer, _ = GLSLRenderer.try_create()

generator = ShaderGenerator(
    llm_fn=code_fn,
    compiler=compiler,     # 阶段五留的钩子，现在注入真 GL
    renderer=renderer,     # 阶段五留的钩子，现在注入真 GL
    critique_fn=make_vision_critique_fn(),
    enable_self_critique=True,
)
```

### 优雅降级

`try_create()` 失败时返回 `(None, reason)`，reason 含按平台的具体安装建议。Generator 拿到 `None` 会自动回退到阶段五的静态 `ValidateCodeAction`。**没有 moderngl 也不会让脚本崩溃**。

### 运行与验收

```bash
# 1. 装新依赖
pip install -r requirements.txt   # moderngl, Pillow, glcontext(Windows)

# 2. 单测（67 项；离线 PASS + 真 GL 自动跳过或运行）
pytest tests/test_agents.py tests/test_analyzer_v2.py \
       tests/test_generator_v2.py tests/test_rendering.py -q

# 3. 独立渲染器验证（不调 DeepSeek）
python -m scripts.verify_renderer
# 期望：3 个负样例被检测 + 8 个 seed 至少 6 个渲染成功 + PNG 头部正确

# 4. Generator 接通真编译 + 真渲染 + 真自评
python -m scripts.verify_generator --render --save-png
python -m scripts.verify_generator --vision-critique --save-png
python -m scripts.verify_generator --combined --vision-critique --save-png

# 5. 渲染单 seed
python -m scripts.verify_renderer --seed seed03 --save-png
ls data/reports/render_test_seed03.png
```

### 多模态自评

`make_vision_critique_fn()` 在 DeepSeek 当前文本模型不支持图像时**自动降级到文本评估**：
- 先尝试 OpenAI 兼容的 vision messages（`content=[{type:"text"},{type:"image_url"}]`）
- TypeError / 422 / 400 等异常 → 回退到纯文本，仅基于代码 + spec 评估
- 仍能给出合理 score，让阶段六的 pipeline 在所有 LLM 后端都能跑通

阶段七若需要更高质量的图像自评，可把 `--vision-critique` 的目标模型切到 OpenAI GPT-4V 或本地 LLaVA（修改 `make_vision_critique_fn(model=...)`）。

### 整套 5 阶段联调命令

```bash
python -m scripts.verify_deepseek                                # 阶段一
python -m scripts.verify_corpus                                  # 阶段二
python -m scripts.verify_analyzer                                # 阶段四
python -m scripts.verify_generator --vision-critique --save-png  # 阶段五+六
python -m scripts.verify_generator --combined --vision-critique --save-png  # 双 agent + 渲染闭环
```

通通 PASS 表示项目就绪——可放到简历。

---

## 阶段七：Gradio 三标签页 UI

把前 6 阶段的能力一次性暴露给非工程师用户的 web 界面：浏览器里点几下，
就能体验 Analyzer / Generator / Collaboration 三种工作流。

### 模块结构

```
shader_agent/ui/
├── app.py        # build_app() 装 3 个 Tab；launch() 起 Gradio 服务
├── runners.py    # AssemblyOptions / get_assembly() / run_analyze / run_generate / run_collaborate
├── examples.py   # 三个 Tab 的预置示例
└── styling.py    # 状态徽章 + 自定义 CSS

scripts/run_ui.py # CLI 入口（--host / --port / --share / --auth）
tests/test_ui.py  # 离线测 runners（不起 Gradio 进程）
```

**关键解耦**：
- `runners.py` 只 import `shader_agent.*`，不 import gradio →
  可以单独 pytest，也方便日后换 Streamlit / FastAPI；
- `get_assembly()` 按"运行选项"做单例缓存，三个 Tab 共享一份 Analyzer/Generator，
  避免每次回调都重建 vector_store / 真 GL context；
- LLM key / 真 GL / vector_db 任一缺失都**优雅降级**：Analyzer/Generator
  fallback 模板 + mock 渲染 + 跳过检索，UI 不崩。

### 三个标签页

**① Analyzer**：粘 GLSL → 一键解读（4 段式报告 + 检索相似样本 + 渲染缩略图）。
预置 6 个 seed 示例（Raymarched Sphere / Smooth Min Blob / Mandelbrot / Domain Warping / 万花筒 / Orbiting Camera），下拉一选即填代码。

**② Generator**：中文需求 + 调色板/复杂度/动态 控件 → 生成 GLSL + 渲染 + 自评。
预置 6 个中文 prompt（霓虹万花筒、smin 双球、fbm 水波、暖色方块、CRT、分形花朵），
点 Examples 一键填入；右侧展示编译徽章、迭代轮数、检索到的参考样本 JSON。

**③ Collaboration**：「先分析后改写」组合任务。粘一段算法 + 写一条改写指令 →
左侧出原始报告，右侧出新版代码 + 原始 vs 新版双图对照。预置 4 组 (代码, 指令) 案例。

### 运行选项（三个 Tab 共享）

顶部折叠面板：
- **渲染后端** `auto` / `real` / `mock`：auto 探测 moderngl 失败自动回退；
- **向量库** `auto` / `off`：off 时检索分支安静跳过；
- **LLM 缓存**：同 prompt 不重复请求 DeepSeek（默认 on）；
- **自评**：启用渲染图多模态评估（无视觉模型自动降级文本评估）；
- **修正循环最大轮数** 0~4：编译失败时让 Generator 重试几次；
- **检索 top_k** 1~8：影响检索的参考样本数量。

### 启动

```bash
pip install gradio>=4.40         # 阶段七唯一额外依赖

python -m scripts.run_ui                           # 默认 127.0.0.1:7860，自动开浏览器
python -m scripts.run_ui --port 8800
python -m scripts.run_ui --host 0.0.0.0            # 暴露到 LAN
python -m scripts.run_ui --share                   # Gradio 公网穿透（72h 临时链接，慎用）
python -m scripts.run_ui --auth alice:s3cret       # HTTP basic auth
python -m scripts.run_ui --auth a:1,b:2            # 多账号
```

### 产物

每次 UI 运行不主动落盘；用户在前端点"保存会话"按钮（或后续追加该按钮）
则会落到：

```
data/reports/ui_session_{ts}_{tab}/
├── payload.json        # 全部回调返回字段（不可序列化的降级为 str）
├── report.md           # AnalysisReport.to_markdown()
├── generated.glsl      # Generator 产物
├── rewritten.glsl      # Collaboration 改写版本
├── image.png           # 渲染预览
├── image_before.png    # Collaboration 原始
└── image_after.png     # Collaboration 改写后
```

### 验收

```bash
pytest tests/test_ui.py -q          # 离线 8 例
python -m scripts.run_ui            # 启动后手测 3 个 Tab
```

阶段七完整闭环：

```bash
# 1) 扩库（任选一/多路径）
python -m scripts.build_corpus --from-local-dir auto --from-id-list data/wanted_ids.txt

# 2) 起 UI
python -m scripts.run_ui

# 3) 浏览器里跑：Tab 1 粘 seed03 → 看到报告；Tab 2 输入中文需求 → 看到渲染；
#    Tab 3 加载预置 (代码, 指令) → 看到 before/after 对比。
```

通通跑通即阶段七完成。
