# Setup SEMIKONG

This documentation dedicated to instruct on how to setup the environment for training, evaluation and inference SEMIKONG model.

## Requirement Hardware
- CUDA Version: >= 10.x (ideally 11.x)
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
pip install -r requirements.txt
~~~

## Training
~~~
1. Download the Meta-LLaMA/Meta-LLaMA base model first on HuggingFace Hub
2. Download the dataset for training semiconductor manufacturing process (should follow the alpaca style)
3. Create a foler `model` which contains the base model
4. Create a folder `data` which contains the dataset
5. Replace the path to the model and dataset foldr in `training.py` and `raw_inference.py`
6. Run `python training.py`
7. Run `raw_inference.py`
~~~

## Inference
~~~
1. Using OpenAI Client: 
python -m vllm.entrypoints.openai.api_server --model <path_to_model_or_HF_model_card_name> --dtype auto --max-lora-rank 32 --api-key token-abc123

2. Using vLLM Server: 
python -m vllm.entrypoints.api_server --model <path_to_model_or_HF_model_card_name> --device cuda --max-lora-rank 32 --dtype auto --port 8080
~~~