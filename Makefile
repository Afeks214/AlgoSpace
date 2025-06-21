.PHONY: help build up down logs clean test lint format install dev prod

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean up generated files"
	@echo "  make dev        - Start development environment"
	@echo "  make prod       - Start production environment"

# Install dependencies
install:
	pip install -r requirements.txt
	pre-commit install

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

# Development
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Database operations
db-init:
	docker-compose exec postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB -f /docker-entrypoint-initdb.d/init.sql

db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# Monitoring
monitor-start:
	docker-compose up -d prometheus grafana

monitor-stop:
	docker-compose stop prometheus grafana

# Ray cluster
ray-start:
	docker-compose up -d ray-head

ray-stop:
	docker-compose stop ray-head

ray-dashboard:
	@echo "Ray dashboard available at http://localhost:8265"

# Jupyter
jupyter-start:
	docker-compose up -d jupyter
	@echo "Waiting for Jupyter to start..."
	@sleep 5
	@docker-compose exec jupyter jupyter notebook list

jupyter-stop:
	docker-compose stop jupyter

# Production deployment
deploy:
	@echo "Deploying to production..."
	./scripts/deploy.sh

# Backup
backup:
	@echo "Creating backup..."
	./scripts/backup.sh

restore:
	@echo "Restoring from backup..."
	./scripts/restore.sh