# Setup SEMIKONG

This guide covers the current contents of the `model/` subtree.

This documentation dedicated to instruct on how to setup the environment for training, evaluation and inference SEMIKONG model.

For runnable commands after installation, see [USAGE.md](USAGE.md).

## Requirement Hardware

- CUDA Version: use a current NVIDIA driver/runtime compatible with the PyTorch version pinned in `model/requirements.txt`
~~~
1. SEMIKONG 8B Chat
    - CPU: Expected to be around 4 cores
    - GPU: Any NVIDIA GPU model with at least 16GB VRAM (A100, A30, RTX 3090, etc.)
    - Disk Memory: At least 10GB disk space
    - RAM Memory: At least 16GB

2. SEMIKONG 70B Chat
    - CPU: Expected to be around 8 cores
    - GPU: Any NVIDIA GPU model with at least 150GB VRAM. Recommend high-end GPU such as A100 or H100 or > RTX 3000
    - Disk Memory: At least 20GB disk space
    - RAM Memory: At least 64GB
~~~

## Environment Setup

- Using `conda` or `poetry` or `venv` to setup the virtual environment
~~~
conda create --name semikong-env python=3.11
conda activate semikong-env
make -C model install
~~~

## Supported Package Versions

The current dependency baseline in [requirements.txt](requirements.txt) includes:

- `torch==2.8.0`
- `torchaudio==2.8.0`
- `torchvision==0.23.0`
- `datasets`
- `transformers`
- `flash_attn`
- `vllm`
- `vllm-flash-attn`

If you need a reproducible environment for debugging install issues, start from that file rather than mixing older PyTorch and torchaudio pins.

## Training
~~~
1. Download the Meta-LLaMA/Meta-LLaMA base model first on HuggingFace Hub
2. Download the dataset for training semiconductor manufacturing process (should follow the alpaca style)
3. Create a foler `model` which contains the base model
4. Create a folder `data` which contains the dataset
5. Replace the path to the model and dataset folder in `model/configs/training-config.yaml` and `model/configs/inference-config.yaml`
6. Run `make -C model train`
7. Run `make -C model infer`
~~~

## Inference
~~~
1. Using OpenAI Client: 
python -m vllm.entrypoints.openai.api_server --model <path_to_model_or_HF_model_card_name> --dtype auto --max-lora-rank 32 --api-key token-abc123

2. Using vLLM Server: 
python -m vllm.entrypoints.api_server --model <path_to_model_or_HF_model_card_name> --device cuda --max-lora-rank 32 --dtype auto --port 8080
~~~

## References

- Usage guide: [USAGE.md](USAGE.md)
- Dataset and benchmark resources: <https://drive.google.com/drive/u/0/folders/1IjuVyP35-xBEe_i_KkG9MnE-4o7Eb7tq>
- Tech report / paper: <https://arxiv.org/abs/2411.13802>
