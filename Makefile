.PHONY: test test-unit test-integration install-test-deps clean-test

# Install test dependencies
install-test-deps:
	uv add --optional test pytest pytest-cov pytest-mock

# Run all tests
test: install-test-deps
	uv run python -m pytest tests/ -v

# Run only unit tests
test-unit: install-test-deps
	uv run python -m pytest tests/unit/ -v -m unit

# Run only integration tests  
test-integration: install-test-deps
	uv run python -m pytest tests/integration/ -v -m integration

# Run tests with coverage
test-cov: install-test-deps
	uv run python -m pytest tests/ -v --cov=src/cml_wd_pytorch --cov-report=html --cov-report=term

# Clean test artifacts
clean-test:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
