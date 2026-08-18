.DEFAULT_GOAL := qwen-help

QWEN_CLI ?= ./dwarfstar
QWEN_SETUP_ARGS ?=
QWEN_DOWNLOAD_ARGS ?=
QWEN_DOCTOR_ARGS ?=
QWEN_CHAT_ARGS ?=
QWEN_SERVER_ARGS ?= --host 127.0.0.1 --port 8080 --max-sequences 1
QWEN_BENCH_ARGS ?= --mode serial
QWEN_TEST_PYTHON ?= ./.venv/bin/python

.PHONY: qwen-help qwen-setup qwen-install-command qwen-download qwen-doctor qwen-chat qwen-server qwen-bench qwen-test clean

qwen-help:
	@echo "Qwen 3.8 27B on Apple Silicon (16 GB):"
	@echo "  make qwen-setup                 Create .venv and install pinned MLX dependencies"
	@echo "  make qwen-install-command       Install the 'dwarfstar' user command"
	@echo "  make qwen-download              Download the pinned 3-bit target (MTP is optional)"
	@echo "  make qwen-doctor                Check Apple Silicon, MLX and the local model cache"
	@echo "  make qwen-chat                  Start a cached xhigh reasoning chat"
	@echo "  make qwen-server                Start the local OpenAI-compatible server on 127.0.0.1:8080"
	@echo "  make qwen-bench                 Benchmark safe serial decoding on complex prompts"
	@echo "  make qwen-test                  Run Qwen wrapper unit tests (no model inference)"
	@echo ""
	@echo "Pass extra options with QWEN_*_ARGS, for example:"
	@echo "  make qwen-chat QWEN_CHAT_ARGS='--profile balanced'"
	@echo "  make qwen-server QWEN_SERVER_ARGS='--host 127.0.0.1 --port 9000 --no-mtp'"
	@echo "  make qwen-bench QWEN_BENCH_ARGS='--mode both --max-tokens 64 --repeats 2'"

qwen-setup:
	./scripts/setup-qwen-macos.sh $(QWEN_SETUP_ARGS)

qwen-install-command:
	./scripts/install-qwen-command.sh

qwen-download:
	$(QWEN_CLI) download $(QWEN_DOWNLOAD_ARGS)

qwen-doctor:
	$(QWEN_CLI) doctor $(QWEN_DOCTOR_ARGS)

qwen-chat:
	$(QWEN_CLI) chat $(QWEN_CHAT_ARGS)

qwen-server:
	$(QWEN_CLI) serve $(QWEN_SERVER_ARGS)

qwen-bench:
	$(QWEN_CLI) benchmark $(QWEN_BENCH_ARGS)

qwen-test:
	$(QWEN_TEST_PYTHON) -m unittest discover -s tests -p 'test_qwen*.py'

clean:
	rm -rf .venv .dwarfstar
