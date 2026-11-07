.PHONY: install deps test load trace smoke llm check dataset show clean

deps:               # 只補相依套件（例如遇到 No module named 'yaml'）
	python -m pip install pyyaml pytest

install:            # 用 python -m pip：保證裝進當前啟用的 conda 環境
	python -m pip install --no-cache-dir -e .

test:               # 沙盒外的程式行為測試（改程式時跑）
	python -m pytest checks/tests/ -q

load:               # 把全部資料載入，印出規模
	python -m rsandbox load

trace:              # 用預設腳本跑一輪，把過程印出來（不判對錯）
	python -m rsandbox trace $(TASK)

smoke:              # 六個變體的每一條腳本都跑一遍，只看會不會出錯
	python -m pytest checks/tests/test_loop.py -q

llm:                # 跑真模型之前的前置檢查（需要 ollama serve）
	python -m rsandbox check-llm --model $(MODEL)

check: load test                   # 改完程式就跑這個

dataset:            # 資料集檢查（資料集定稿後跑一次）
	python checks/agent_visible.py

show:               # 看載入結果
	python -m rsandbox show tools
	python -m rsandbox show rules
	python -m rsandbox show tasks

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf src/*.egg-info build dist .pytest_cache
