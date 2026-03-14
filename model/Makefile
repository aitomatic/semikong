PYTHON ?= python
PIP ?= pip
TRAIN_CONFIG ?= configs/training-config.yaml
INFER_CONFIG ?= configs/inference-config.yaml

.PHONY: help install train infer check

help:
	@echo "Available targets:"
	@echo "  install  Install model dependencies"
	@echo "  train    Run model training"
	@echo "  infer    Run raw model inference"
	@echo "  check    Validate Python entrypoints compile"

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) training/train.py --config $(TRAIN_CONFIG)

infer:
	$(PYTHON) inference/raw_inference.py --config $(INFER_CONFIG)

check:
	$(PYTHON) -m py_compile training/train.py inference/raw_inference.py inference/openai_client.py inference/vllm_client.py common/logging_setup.py
