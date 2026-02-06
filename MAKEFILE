.PHONY: install test lint format clean data docker-build docker-run

# Installation
install:
	pip install -e .

# Tests
test:
	pytest tests/

# Qualité de code
lint:
	flake8 mt5d/ examples/ scripts/

format:
	black mt5d/ examples/ scripts/

# Nettoyage
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/

# Génération de données
data:
	python scripts/download_example_data.py --dataset medical --size 500

# Docker
docker-build:
	docker build -t mt5d-api -f docker/Dockerfile .

docker-run:
	docker run -p 8000:8000 mt5d-api

# Démonstration complète
demo: install data
	python examples/medical_example.py
