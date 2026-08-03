"""接口自动化测试的公共装置。

两种运行形态，用例代码完全相同：

    # 1) 进程内（默认，CI 用）：直接挂 ASGI，不占端口、无网络抖动
    pytest tests_api

    # 2) 跨进程（预发/联调用）：打一个真实部署的实例
    pytest tests_api --base-url=http://10.0.0.5:8000 --api-key=xxx

环境相关的东西全部收敛在这里：起服务、造客户端、准备素材、每条用例后复位
故障配置。用例里不应出现任何 `os.environ` 或 `create_app`。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from tests_api.api_objects import FaultAPI, RetrievalAPI, ShaderAPI, SystemAPI
from tests_api.utils.yaml_loader import load_shaders

ROOT = Path(__file__).resolve().parents[1]

MARKERS = [
    ("smoke", "冒烟：每次提交必跑，2 分钟内出结果"),
    ("api", "接口功能用例"),
    ("contract", "响应契约（JSON Schema）用例"),
    ("fault", "异常与故障注入用例"),
    ("e2e", "端到端业务流用例"),
    ("image", "需要图像层校验的用例"),
    ("slow", "耗时较长，日常可用 -m 'not slow' 跳过"),
    ("gpu", "需要真实 OpenGL 环境，无 GL 时自动跳过"),
    ("live", "打真实大模型，需要 API Key，仅 nightly 跑"),
]


def pytest_addoption(parser):
    g = parser.getgroup("shader-agent")
    g.addoption("--base-url", action="store", default="",
                help="被测服务地址；留空则在进程内启动 ASGI 应用")
    g.addoption("--profile", action="store", default="test",
                choices=["test", "auto", "real"],
                help="进程内启动时使用的装配 profile")
    g.addoption("--api-key", action="store", default=os.environ.get("SHADER_AGENT_API_KEY", ""),
                help="接口鉴权用的 X-API-Key")


def pytest_configure(config):
    for name, desc in MARKERS:
        config.addinivalue_line("markers", f"{name}: {desc}")


def pytest_report_header(config):
    target = config.getoption("--base-url") or f"in-process(profile={config.getoption('--profile')})"
    return f"shader-agent target: {target}"


# ---------------- 客户端 ----------------

@pytest.fixture(scope="session")
def profile(request) -> str:
    return request.config.getoption("--profile")


@pytest.fixture(scope="session")
def api_key(request) -> str:
    return request.config.getoption("--api-key")


@pytest.fixture(scope="session")
def http_client(request, profile) -> Iterator[object]:
    base_url = request.config.getoption("--base-url")
    if base_url:
        import httpx
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=180.0) as client:
            yield client
        return

    # 进程内：先定 profile 再导入应用，保证装配层读到正确的环境变量
    os.environ.setdefault("SHADER_AGENT_PROFILE", profile)
    os.environ.setdefault("SHADER_AGENT_LLM_CACHE", "0")
    from fastapi.testclient import TestClient

    from shader_agent.api.main import create_app

    app = create_app(profile=profile)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture(scope="session")
def system_api(http_client, api_key) -> SystemAPI:
    return SystemAPI(http_client, api_key)


@pytest.fixture(scope="session")
def shader_api(http_client, api_key) -> ShaderAPI:
    return ShaderAPI(http_client, api_key)


@pytest.fixture(scope="session")
def retrieval_api(http_client, api_key) -> RetrievalAPI:
    return RetrievalAPI(http_client, api_key)


@pytest.fixture(scope="session")
def fault_api(http_client, api_key) -> FaultAPI:
    return FaultAPI(http_client, api_key)


@pytest.fixture(scope="session")
def fault_injection_available(fault_api) -> bool:
    """故障注入路由只在 test profile 下存在；不存在时相关用例整体跳过。"""
    try:
        return fault_api.current().ok
    except Exception:
        return False


@pytest.fixture
def faults(fault_api, fault_injection_available) -> Iterator[FaultAPI]:
    """要用故障注入的用例声明这个 fixture；退出时自动复位。"""
    if not fault_injection_available:
        pytest.skip("当前 profile 未启用故障注入路由（需 profile=test）")
    fault_api.reset()
    yield fault_api
    fault_api.reset()


@pytest.fixture(scope="session", autouse=True)
def _service_ready(system_api) -> None:
    """全套用例开跑前先体检一次：环境没起来就立刻失败，而不是让 50 条用例一起红。"""
    res = system_api.readyz()
    assert res.status_code == 200, f"服务未就绪：{res}"
    assert res.data["status"] in ("ok", "degraded"), f"服务不可用：{res}"


# ---------------- 素材 ----------------

@pytest.fixture(scope="session")
def shaders() -> dict[str, str]:
    return load_shaders()


@pytest.fixture(scope="session")
def valid_shader(shaders) -> str:
    return shaders["valid_plasma"]


@pytest.fixture(scope="session")
def broken_shader(shaders) -> str:
    return shaders["broken_type_mismatch"]


@pytest.fixture(scope="session")
def unsupported_shader(shaders) -> str:
    return shaders["unsupported_ichannel"]


@pytest.fixture(scope="session")
def gpu_available(system_api) -> bool:
    res = system_api.readyz()
    detail = (res.data or {}).get("components", {}).get("render", {}).get("detail", "")
    return "moderngl" in str(detail)


@pytest.fixture
def require_gpu(gpu_available):
    if not gpu_available:
        pytest.skip("当前环境无真实 OpenGL 后端")
