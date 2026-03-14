# Overview

## What is SEMIKONG?

SEMIKONG is an open-source, semiconductor-focused language model intended for manufacturing-process and domain-knowledge use cases.

Key points:

- domain-specific focus on semiconductor manufacturing and related technical knowledge
- model variants including `SEMIKONG-8B-Instruct` and `SEMIKONG-70B-Instruct`
- compatibility with the broader Llama tooling ecosystem

## Model Links

| Model | Link |
|---|---|
| SEMIKONG-70B-Instruct | <https://huggingface.co/pentagoniac/SEMIKONG-70B> |
| SEMIKONG-8B-Instruct | <https://huggingface.co/pentagoniac/SEMIKONG-8b-GPTQ> |

## Quick Start

1. Clone the repository.
2. Install dependencies with `pip install -r model/requirements.txt`.
3. Configure model and dataset paths in:
   - `model/configs/training-config.yaml`
   - `model/configs/inference-config.yaml`
4. Run training with `python model/training/train.py`.
5. Run inference with `python model/inference/raw_inference.py`.

For detailed environment setup, see [INSTALL.md](/Users/ctn/src/aitomatic/semikong/model/INSTALL.md).

## Hardware Notes

- `SEMIKONG-8B` is oriented toward single-GPU setups with at least 16 GB VRAM.
- `SEMIKONG-70B` requires substantially more GPU and CPU memory.

See [ecosystem.md](/Users/ctn/src/aitomatic/semikong/model/docs/ecosystem.md) for deployment context.
