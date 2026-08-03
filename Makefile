# 常用命令入口。CI 与本地跑的是同一套命令，避免"本地能过、CI 挂了"。

PY ?= python
PYTEST ?= pytest

.PHONY: help install doctor api api-real smoke config test test-fast test-remote \
        fault e2e cov report perf clean

help:
	@echo "install    安装接口层与测试依赖"
	@echo "doctor     体检：打印 profile、依赖与脱敏后的凭据状态"
	@echo "api        本地启动 HTTP 服务（test profile，无需 API Key）"
	@echo "smoke      冒烟用例（约 1 分钟）"
	@echo "config     配置层守护用例（不依赖任何服务，最快）"
	@echo "test       全量接口自动化（并行）"
	@echo "test-fast  跳过 slow/gpu/live"
	@echo "fault      只跑故障注入用例"
	@echo "e2e        只跑端到端业务流"
	@echo "cov        带覆盖率"
	@echo "report     生成并打开 Allure 报告"
	@echo "perf       本地压测（需先 make api）"

install:
	$(PY) -m pip install -r requirements.txt -r requirements-api.txt -r requirements-test.txt

# 落地第一步先跑它：不起服务、不调模型，30 秒内回答"这台机器缺什么"
doctor:
	$(PY) -m shader_agent.config.doctor

api:
	SHADER_AGENT_PROFILE=test uvicorn shader_agent.api.main:app --reload --port 8000

api-real:
	SHADER_AGENT_PROFILE=auto uvicorn shader_agent.api.main:app --port 8000

smoke:
	$(PYTEST) tests_api -m smoke -q

config:
	$(PYTEST) tests -m config -q

test:
	$(PYTEST) tests_api -m "not gpu and not live" -n auto --dist loadfile \
		--alluredir=reports/allure-results --junitxml=reports/junit.xml

test-fast:
	$(PYTEST) tests_api -m "not slow and not gpu and not live" -n auto -q

fault:
	$(PYTEST) tests_api -m fault -v

e2e:
	$(PYTEST) tests_api -m e2e -v

# 打已部署实例（预发环境）
test-remote:
	$(PYTEST) tests_api --base-url=$(BASE_URL) -m "not fault" -v

cov:
	$(PYTEST) tests tests_api -m "not gpu and not live" \
		--cov=shader_agent --cov-report=html:reports/htmlcov --cov-report=term-missing

report:
	allure generate reports/allure-results -o reports/allure-report --clean
	allure open reports/allure-report

perf:
	mkdir -p reports/perf
	locust -f tests_api/perf/locustfile.py --host http://127.0.0.1:8000 \
		--headless -u 20 -r 5 -t 2m --csv=reports/perf/shader

clean:
	rm -rf reports .pytest_cache .coverage htmlcov
	find . -name __pycache__ -type d -exec rm -rf {} +
