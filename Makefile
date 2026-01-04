.PHONY: help data train evaluate clean

PYTHON = python3

help:
	@echo "Available commands:"
	@echo "  make data       Download and process data"
	@echo "  make train      Train the model"
	@echo "  make evaluate   Evaluate the model"
	@echo "  make clean      Remove generated files"

data:
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/preprocess.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

clean:
	rm -rf data/processed
	rm -rf artifacts/models artifacts/metrics artifacts/plots