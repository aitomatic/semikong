# Using the SEMIKONG Model

This guide covers the intended command paths for training, local inference, and vLLM serving.

## Quick Start

From the repository root:

```bash
make -C model install
make -C model train
make -C model infer
```

Before running those commands, update:

- [configs/training-config.yaml](/Users/ctn/src/aitomatic/semikong/model/configs/training-config.yaml)
- [configs/inference-config.yaml](/Users/ctn/src/aitomatic/semikong/model/configs/inference-config.yaml)

At minimum, set the paths for the base model, adapter output, and local dataset.

## Training

The supported training entrypoint is:

```bash
make -C model train
```

That target runs:

```bash
python model/training/train.py --config model/configs/training-config.yaml
```

Expected inputs:

- a local base model checkpoint
- a local instruction-style dataset
- a writable output directory for checkpoints and adapters

## Local Inference

The supported inference entrypoint is:

```bash
make -C model infer
```

That target runs:

```bash
python model/inference/raw_inference.py --config model/configs/inference-config.yaml
```

Use this path for direct local generation against a configured base model and adapter.

## vLLM Serving

For an OpenAI-compatible API server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <path_to_model_or_hf_model> \
  --dtype auto \
  --max-lora-rank 32 \
  --api-key token-abc123
```

For the vLLM API server endpoint:

```bash
python -m vllm.entrypoints.api_server \
  --model <path_to_model_or_hf_model> \
  --device cuda \
  --max-lora-rank 32 \
  --dtype auto \
  --port 8080
```

## Models, Datasets, and Paper

- Public model weights:
  - Base 70B: <https://huggingface.co/pentagoniac/SEMIKONG-70B>
  - Quantized 8B GPTQ: <https://huggingface.co/pentagoniac/SEMIKONG-8b-GPTQ>
  - Quantized 8B instruct GPTQ: <https://huggingface.co/sitloboi2012/SEMIKONG-8B-Instruct-GPTQ>
- Dataset and benchmark resources: <https://drive.google.com/drive/u/0/folders/1IjuVyP35-xBEe_i_KkG9MnE-4o7Eb7tq>
- Tech report / paper: <https://arxiv.org/abs/2411.13802>
