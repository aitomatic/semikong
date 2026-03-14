# Ecosystem

## Upstream

SEMIKONG follows the Llama model architecture closely enough to benefit from existing libraries, serving tools, and inference workflows in that ecosystem.

Example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("pentagoniac/SEMIKONG-8b-GPTQ", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("pentagoniac/SEMIKONG-8b-GPTQ", device_map="auto")
```

## Deployment Notes

Before deploying quantized models, verify the surrounding toolchain for the specific quantization format you plan to use.

| Model Type | Software |
|---|---|
| 4-bit quantized | AWQ and CUDA |
| 8-bit quantized | GPTQ and CUDA |

## Hardware Guidance

| Model | Minimum VRAM | Example |
|---|---:|---|
| SEMIKONG-70B-Instruct | 170 GB | 3 x A100 80GB or 5 x A100 40GB |
| SEMIKONG-8B-Instruct | 16 GB | RTX 3060 / RTX 4060 class |

## Downstream Uses

Potential downstream uses include:

- interactive chat assistants for semiconductor engineering tasks
- internal copilots for manufacturing documentation and workflows
- domain-adapted model fine-tuning
- serving through Hugging Face, vLLM, or other Llama-compatible infrastructure

## Paper and Citation

Paper: <https://arxiv.org/abs/2411.13802>

```bibtex
@article{semikong2024,
  title={SemiKong: Curating, Training, and Evaluating A Semiconductor Industry-Specific Large Language Model},
  author={Christopher Nguyen, William Nguyen, Atsushi Suzuki, Daisuke Oku, Hong An Phan, Sang Dinh, Zooey Nguyen, Anh Ha, Shruti Raghavan, Huy Vo, Thang Nguyen, Lan Nguyen, Yoshikuni Hirayama},
  journal={arXiv preprint arXiv:2411.13802},
  year={2024}
}
```
