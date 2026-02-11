.PHONY: test test-unit test-integration test-system lint clean install

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov flake8 mypy

test: test-unit test-integration test-system

test-unit:
	pytest python/gps/tests -m unit -v

test-integration:
	pytest python/gps/tests/integration -m integration -v

test-system:
	pytest python/gps/tests/system -m system -v

test-all:
	pytest python/gps/tests -v

coverage:
	pytest python/gps/tests -m "unit or integration" --cov=python/gps --cov-report=html --cov-report=term

lint:
	flake8 python/gps --count --select=E9,F63,F7,F82 --show-source --statistics
	mypy python/gps --ignore-missing-imports || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

check-syntax:
	python3 -m compileall python/gps

format:
	black python/gps || true
	isort python/gps || true
