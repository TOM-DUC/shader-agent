"""Locust 压测脚本。

压这套接口的关键，是**分层压**而不是一股脑压生成接口：

- `/validate`、`/compile`：毫秒级纯 CPU，用来量服务框架本身的吞吐与排队行为；
- `/render`：GPU / 软件渲染，观察并发下的显存与上下文竞争；
- `/generate`：秒级、受上游速率限制，权重压到最低，主要看超时与错误率。

另外重写了失败判定：HTTP 200 但 `code != 0` 同样计为失败。默认的 Locust 只看
状态码，而这套接口把业务失败放在信封里，不改判定就会把一堆错误压测成"全绿"。

运行：
    locust -f tests_api/perf/locustfile.py --host http://127.0.0.1:8000
    # 无头模式（CI）
    locust -f tests_api/perf/locustfile.py --host http://127.0.0.1:8000 \\
           --headless -u 20 -r 5 -t 2m --csv=reports/perf/shader
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

VALID_SHADER = """\
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float d = length(uv);
    float wave = 0.5 + 0.5 * sin(6.2831 * d * 3.0 - iTime * 1.5);
    vec3 base = vec3(0.350, 0.600, 0.900);
    fragColor = vec4(base * wave, 1.0);
}
"""

PROMPTS = [
    "生成一个蓝色调的同心圆波纹动画",
    "生成一个暖色的渐变背景",
    "生成一个绿色的噪声图案",
    "生成一个紫色的分形效果",
]

QUERIES = ["raymarching sdf", "value noise 噪声", "mandelbrot 分形", "voronoi 元胞"]

# 各接口的耗时基线（毫秒），超过即在报告里标记为慢请求
SLA_MS = {"validate": 200, "compile": 500, "render": 2000, "search": 800,
          "generate": 30000}


class ShaderAgentUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        key = os.environ.get("SHADER_AGENT_API_KEY", "")
        self.client.headers.update({"Content-Type": "application/json"})
        if key:
            self.client.headers.update({"X-API-Key": key})

    # ---------- 统一的业务成功判定 ----------
    def _call(self, name: str, path: str, payload: dict) -> None:
        with self.client.post(path, json=payload, name=name,
                              catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"http {resp.status_code}")
                return
            try:
                body = resp.json()
            except Exception:
                resp.failure("响应不是合法 JSON")
                return
            if body.get("code") != 0:
                resp.failure(f"business code={body.get('code')} {body.get('message')}")
                return
            budget = SLA_MS.get(name)
            if budget and resp.elapsed.total_seconds() * 1000 > budget:
                resp.failure(f"超出耗时基线 {budget}ms")
                return
            resp.success()

    # ---------- 任务权重：贴近真实调用比例 ----------
    @task(10)
    def validate(self) -> None:
        self._call("validate", "/api/v1/shader/validate", {"code": VALID_SHADER})

    @task(6)
    def compile(self) -> None:
        self._call("compile", "/api/v1/shader/compile", {"code": VALID_SHADER})

    @task(5)
    def search(self) -> None:
        self._call("search", "/api/v1/retrieval/search",
                   {"query": random.choice(QUERIES), "top_k": 3})

    @task(4)
    def render(self) -> None:
        self._call("render", "/api/v1/shader/render",
                   {"code": VALID_SHADER, "width": 256, "height": 192,
                    "time": round(random.uniform(0, 5), 2)})

    @task(1)
    def generate(self) -> None:
        self._call("generate", "/api/v1/shader/generate",
                   {"description": random.choice(PROMPTS), "palette": "蓝色"})


@events.quitting.add_listener
def _assert_thresholds(environment, **_kw) -> None:
    """把压测变成可判定的门禁：不达标就以非 0 退出码结束，CI 直接判红。"""
    stats = environment.stats.total
    if stats.num_requests == 0:
        return
    fail_ratio = stats.fail_ratio
    p95 = stats.get_response_time_percentile(0.95)
    if fail_ratio > 0.02:
        environment.process_exit_code = 1
        print(f"[perf] 失败率 {fail_ratio:.2%} 超过 2% 阈值")
    elif p95 and p95 > 5000:
        environment.process_exit_code = 1
        print(f"[perf] P95 {p95:.0f}ms 超过 5000ms 阈值")
    else:
        environment.process_exit_code = 0
