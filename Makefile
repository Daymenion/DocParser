# Simple Makefile for dots.ocr app and vLLM stack

PROJECT_NAME := doc-parser
DOCKER_DIR := docker
COMPOSE := docker-compose -f $(DOCKER_DIR)/docker-compose.yml
PY := python
WEIGHTS := weights/DotsOCR

.PHONY: ensure-model
ensure-model:
	@if [ ! -d "$(WEIGHTS)" ] || [ -z "`ls -A $(WEIGHTS) 2>/dev/null`" ]; then \
		echo "[ensure-model] Weights not found in $(WEIGHTS). Downloading..."; \
		$(PY) tools/download_model.py --type huggingface --name rednote-hilab/dots.ocr; \
	else \
		echo "[ensure-model] Weights already present in $(WEIGHTS)."; \
	fi

.PHONY: all build up down logs clean restart

all: build up

build: ensure-model
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

restart:
	$(COMPOSE) down; $(COMPOSE) up -d

logs:
	$(COMPOSE) logs -f --tail=200

ps:
	$(COMPOSE) ps

stop:
	$(COMPOSE) stop

	
start:
	$(COMPOSE) start

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v --remove-orphans

# Convenience targets
app-shell:
	docker exec -it doc-parser-app /bin/bash

vllm-shell:
	docker exec -it doc-parser-vllm /bin/bash
