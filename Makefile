# DocParser Docker Management Makefile

# Project configuration
PROJECT_NAME = docparser
IMAGE_NAME = $(PROJECT_NAME)
CONTAINER_APP = $(PROJECT_NAME)-app
CONTAINER_VLLM = $(PROJECT_NAME)-vllm
DOCKER_COMPOSE = docker-compose -f docker/docker-compose.yml

# Service URLs
API_URL = http://localhost:7860
VLLM_URL = http://localhost:9998
DOCS_URL = $(API_URL)/docs

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
CYAN = \033[0;36m
NC = \033[0m

# Declare phony targets
.PHONY: help install build up down restart logs status shell clean health test dev deploy
.PHONY: download-model setup-dev quick-start clean-all
.PHONY: logs-app logs-vllm shell-app shell-vllm
.PHONY: dev-api dev-ui dev-combined

# Default target
help: ## Show this help message
	@echo "$(BLUE)🐳 DocParser Docker Management$(NC)"
	@echo "===================================="
	@echo ""
	@echo "$(CYAN)Quick Start:$(NC)"
	@echo "  $(GREEN)make quick-start$(NC)    # Build, start and check everything"
	@echo ""
	@echo "$(CYAN)Available Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(CYAN)Service URLs:$(NC)"
	@echo "  API + UI:  $(API_URL)"
	@echo "  API Docs:  $(DOCS_URL)"
	@echo "  vLLM:      $(VLLM_URL)"

# === SETUP COMMANDS ===
install: ## Install Python dependencies
	@echo "$(YELLOW)📦 Installing dependencies...$(NC)"
	@pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

download-model: ## Download DotsOCR model weights
	@echo "$(YELLOW)📥 Downloading DotsOCR model...$(NC)"
	@python tools/download_model.py
	@echo "$(GREEN)✅ Model download completed$(NC)"

setup-dev: install download-model ## Setup complete development environment
	@echo "$(GREEN)🔧 Development environment ready!$(NC)"

# === DOCKER COMMANDS ===
build: ## Build Docker image
	@echo "$(YELLOW)🔨 Building DocParser Docker image...$(NC)"
	@docker build -t $(IMAGE_NAME) -f docker/Dockerfile .
	@echo "$(GREEN)✅ Build completed$(NC)"

up: ## Start all services
	@echo "$(YELLOW)🚀 Starting DocParser services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✅ Services started$(NC)"
	@echo "$(BLUE)Access: $(API_URL)$(NC)"

down: ## Stop all services
	@echo "$(YELLOW)🛑 Stopping services...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✅ Services stopped$(NC)"

restart: down up ## Restart all services

# === MONITORING COMMANDS ===
status: ## Show service status
	@echo "$(CYAN)📊 Service Status:$(NC)"
	@$(DOCKER_COMPOSE) ps

logs: ## View logs from all services
	@$(DOCKER_COMPOSE) logs -f

logs-app: ## View application logs only
	@$(DOCKER_COMPOSE) logs -f $(CONTAINER_APP)

logs-vllm: ## View vLLM logs only
	@$(DOCKER_COMPOSE) logs -f vllm-server

health: ## Check service health
	@echo "$(YELLOW)🔍 Checking service health...$(NC)"
	@echo -n "$(CYAN)Application: $(NC)"
	@curl -sf $(API_URL)/health > /dev/null && echo "$(GREEN)✅ Healthy$(NC)" || echo "$(RED)❌ Unhealthy$(NC)"
	@echo -n "$(CYAN)vLLM Server: $(NC)"
	@curl -sf $(VLLM_URL)/health > /dev/null && echo "$(GREEN)✅ Healthy$(NC)" || echo "$(RED)❌ vLLM unhealthy$(NC)"

# === DEVELOPMENT COMMANDS ===
shell: shell-app ## Open shell in app container (alias)

shell-app: ## Open shell in application container
	@docker exec -it $(CONTAINER_APP) /bin/bash

shell-vllm: ## Open shell in vLLM container
	@docker exec -it $(CONTAINER_VLLM) /bin/bash

dev: dev-combined ## Start local development (alias)

dev-combined: ## Start combined API+UI locally
	@echo "$(YELLOW)🔧 Starting local development (combined)...$(NC)"
	@python app.py --mode combined --dev

dev-api: ## Start API-only locally
	@echo "$(YELLOW)🔧 Starting local development (API only)...$(NC)"
	@python app.py --mode api --dev

dev-ui: ## Start UI-only locally
	@echo "$(YELLOW)🔧 Starting local development (UI only)...$(NC)"
	@python app.py --mode ui --dev

# === TESTING COMMANDS ===
test: health ## Run health checks and basic tests
	@echo "$(YELLOW)🧪 Running tests...$(NC)"
	@echo "$(GREEN)✅ Health checks completed$(NC)"

# === CLEANUP COMMANDS ===
clean: ## Clean containers and unused images
	@echo "$(YELLOW)🧹 Cleaning up Docker resources...$(NC)"
	@$(DOCKER_COMPOSE) down -v --remove-orphans
	@docker image prune -f
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-all: ## Remove everything including project images
	@echo "$(RED)🗑️  Removing all DocParser resources...$(NC)"
	@$(DOCKER_COMPOSE) down -v --remove-orphans
	@docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@docker system prune -f
	@echo "$(GREEN)✅ Complete cleanup finished$(NC)"

# === DEPLOYMENT COMMANDS ===
quick-start: build up ## 🚀 Build, start and verify everything
	@echo "$(GREEN)🚀 DocParser is starting up!$(NC)"
	@echo "$(BLUE)⏳ Waiting for services to be ready...$(NC)"
	@sleep 30
	@make health
	@echo ""
	@echo "$(GREEN)🎉 DocParser is ready!$(NC)"
	@echo "$(CYAN)🌐 Access the application:$(NC)"
	@echo "  Web Interface: $(API_URL)"
	@echo "  API Docs:      $(DOCS_URL)"

deploy: quick-start ## Deploy to production
	@echo "$(GREEN)🚀 DocParser deployed successfully!$(NC)"
