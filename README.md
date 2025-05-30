<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/teaser.png" width="200px">
  <source media="(prefers-color-scheme: light)" srcset="figures/teaser.png" width="200px"> 
  <img alt="specify theme context for images" src="figures/teaser.png" width="200px">
</picture>

</br>
</br>

</div>

<div id="top"></div>  

<div align="center">
  <h3 align="center">SEMIKONG - The Open Source Foundation Model for Semiconductor Manufacturing Process</h3>
</div>
<p align="center">
 <a href="https://drive.google.com/drive/u/0/folders/1IjuVyP35-xBEe_i_KkG9MnE-4o7Eb7tq" target="_blank">Dataset and Benchmarks</a> • 🤖 <a href="https://huggingface.co/pentagoniac/SEMIKONG-70B" target="_blank">Hugging Face Model 70B-Instruct</a>• 🤖 <a href="https://huggingface.co/pentagoniac/SEMIKONG-8b-GPTQ" target="_blank">Hugging Face Model 8B-Instruct</a>
</p> 

<p align="center">
    👩‍🚀 Ask questions or discuss ideas on <a href="#" target="_blank"> GitHub </a>
</p> 

<p align="center">
    📝 Check out  <a href="https://arxiv.org/abs/2411.13802"> SemiKong Paper </a>
</p> 

<!-- DO NOT REMOVE ME -->

<hr>

<details open>
<summary></b>📕 Table of Contents</b></summary>

- [What is SEMIKONG?](#what-is-semikong)
  - [Introduction](#introduction)
  - [Key Features](#features)
  - [Models](#models)
    - [Chat models](#chat-models)
- [How to use SEMIKONG?](#how-to-use-semikong)
  - [Quick start](#quick-start)
    - [Choose your path](#choose-your-path)
    - [pip](#quick-start---pip)
    - [docker](#quick-start---docker)
    - [Web demo](#web-demo)
  - [Fine-tuning](#fine-tuning)
  - [Quantization](#quantization)
  - [Deployment](#deployment)
  - [FAQ](#faq)
  - [Learning hub](#learning-hub)
- [Why SEMIKONG?](#why-semikong)
  - [Ecosystem](#ecosystem)
    - [Upstream](#upstream)
    - [Downstream](#downstream)
      - [Serving](#serving)
      - [Quantization](#quantization-1)
      - [Fine-tuning](#fine-tuning-1)
      - [API](#api)
  - [Benchmarks](#benchmarks)
    - [Chat model performance](#chat-model-performance)
  - [Tech report](#tech-report)
    - [Citation](#citation)
- [Who can use SEMIKONG?](#who-can-use-semikong)
- [Misc.](#misc)
  - [Acknowledgements](#acknowledgments)
  - [Disclaimer](#disclaimer)
  - [License](#license)

</details>

<hr>

# What is SEMIKONG?

## Introduction 

- 🤖 SEMIKONG is an open-source, industry-specific large language model (LLM) tailored to the semiconductor domain. It aims to address the unique challenges faced by the semiconductor industry, such as the physics and chemistry of semiconductor devices and processes, by incorporating domain-specific knowledge into the model.

- 🙌 Targeted as a bilingual language model and trained on 3T multilingual corpus, the SEMIKONG series models become one of the strongest LLM worldwide, showing promise in language understanding, commonsense reasoning, reading comprehension, and more. For example,
  
  - SEMIKONG-8B / 70B-Instruct model .
  
  - 🙏 (Credits to Llama) Thanks to the Transformer and Llama open-source communities, as they reduce the efforts required to build from scratch and enable the utilization of the same tools within the AI ecosystem.  
</ul>
</details>

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

## News 

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

## Key Features

- First industry-specific LLM for the semiconductor domain
- Trained on a comprehensive semiconductor-related text corpus
- Novel pre-training approach leveraging domain-specific knowledge
- Superior performance compared to general-purpose LLMs on industry-relevant benchmarks
- Serves as a valuable foundation for companies to build proprietary models tailored to their needs

## Models

SEMIKONG models come in multiple sizes and cater to different use cases. You can also fine-tune SEMIKONG models to meet your specific requirements. 

If you want to deploy SEMIKONG models, make sure you meet the [software and hardware requirements](#deployment).

### Instruct models

| Model | Download  |
|---|---|
|SEMIKONG-70B-Instruct	| • [🤗 Hugging Face](https://huggingface.co/pentagoniac/SEMIKONG-70B) |
|SEMIKONG-8B-Instruct | • [🤗 Hugging Face](https://huggingface.co/pentagoniac/SEMIKONG-8b-GPTQ) |


# How to use SEMIKONG?

- [Quick start](#quick-start)
  - [Choose your path](#choose-your-path)
  - [pip](#quick-start---pip)
  - [docker](#quick-start---docker)
  - [Web demo](#web-demo)
- [Fine-tuning](#fine-tuning)
- [Quantization](#quantization)
- [Deployment](#deployment)
- [FAQ](#faq)
- [Learning hub](#learning-hub)

## Quick start

Getting up and running with SEMIKONG models is simple with multiple choices available. 

### Choose your path

Select one of the following paths to begin your journey with SEMIKONG!

#### 🎯 Deploy SEMIKONG locally

If you prefer to deploy SEMIKONG models locally, 

  - 🙋‍♀️ and you have **sufficient** resources (for example, NVIDIA A100 40GB), you can choose one of the following methods:
    - [pip](#quick-start---pip)
    - [Docker](#quick-start---docker)

#### 🎯 Not to deploy SEMIKONG locally

If you prefer not to deploy SEMIKONG models locally, you can explore SEMIKONG's capabilities using any of the following options.

##### 🙋‍♀️ Chat with SEMIKONG

 If you want to chat with SEMIKONG, you can use one of these online services, which offer a similar user experience:

- [SEMIKONG-70B-Instruct](#) (SEMIKONG official on Hugging Face)

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Quick start - pip 

This tutorial guides you through every step of running **SEMIKONG-8B-Instruct locally on an A100 (40G)** and then performing inference.

#### Step 0: Prerequisites

- Make sure Python 3.10 or a later version is installed.

- If you want to run other SEMIKONG models, see [software and hardware requirements](#deployment).

#### Step 1: Prepare your environment 

To set up the environment and install the required packages, execute the following command.

```bash
git clone https://github.com/aitomatic/semikong.git
cd semikong
pip install -r requirements.txt
```

#### Step 2: Download the SEMIKONG model

You can download the weights and tokenizer of SEMIKONG models from the following sources:

- [Hugging Face](https://huggingface.co/pentagoniac)

#### Step 3: Perform inference

You can perform inference with SEMIKONG chat models as below.

##### Perform inference with SEMIKONG chat model

1. Create a file named  `quick_start.py` and copy the following content to it.

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = '<your-model-path>'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype='auto'
    ).eval()

    # Prompt content: "hi"
    messages = [
        {"role": "user", "content": "hi"}
    ]

    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'))
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Model response: "Hello! How can I assist you today?"
    print(response)
    ```

2. Run `quick_start.py`.

    ```bash
    python quick_start.py
    ```

    Then you can see an output similar to the one below. 🥳

    ```bash
    Hello! How can I assist you today?
    ```

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Quick start - Docker 
TBA

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Web demo

You can build a web UI demo for SEMIKONG **chat** models.

[Step 1: Prepare your environment](#step-1-prepare-your-environment). 

[Step 2: Download the SEMIKONG model](#step-2-download-the-semikong-model).

Step 3. To start a web service locally, run the following command.

```bash
python demo/web_demo.py -c <your-model-path>
```

You can access the web UI by entering the address provided in the console into your browser. 

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Fine-tuning

### Finetune code for SEMIKONG 8B and 70B

#### Hardware Setup

For the SEMIKONG-8B model, a node with 1 GPUs, each with GPU memory larger than 16GB, is recommended.

For the SEMIKONG-70B model, because the usage of the zero-offload technique consumes a lot of CPU memory, please be careful to limit the number of GPUs in the 70B finetune training. Please use CUDA_VISIBLE_DEVICES to limit the number of GPUs (as shown in scripts/run_sft_SEMIKONG_70b.sh).

A typical hardware setup for finetuning the 70B model is a node with 8 GPUs (limited to 4 in running by CUDA_VISIBLE_DEVICES=0,1,2,3), each with GPU memory larger than 80GB, and total CPU memory larger than 900GB.

#### Quick Start

### Deployment

If you want to deploy SEMIKONG models, make sure you meet the software and hardware requirements. 

#### Software requirements

Before using SEMIKONG quantized models, make sure you've installed the correct software listed below.

| Model | Software
|---|---
SEMIKONG 4-bit quantized models | [AWQ and CUDA](https://github.com/casper-hansen/AutoAWQ?tab=readme-ov-file#install-from-pypi)
SEMIKONG 8-bit quantized models |  [GPTQ and CUDA](https://github.com/PanQiWei/AutoGPTQ?tab=readme-ov-file#quick-installation)

#### Hardware requirements

Before deploying SEMIKONG in your environment, make sure your hardware meets the following requirements.

##### Instruction models

| Model                | Minimum VRAM |        Recommended GPU Example       |
|:----------------------|:--------------|:-------------------------------------:|
| SEMIKONG-70B-Instruct           | 170 GB         | 3 x A100 80GB <br> 5 x A100 40GB             |
| SEMIKONG-8B-Instruct     | 16 GB          | 1 x RTX 3060 (12 GB)<br> 1 x RTX 4060 (8 GB)                   |

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>


# Why SEMIKONG? 

  - [Ecosystem](#ecosystem)
    - [Upstream](#upstream)
    - [Downstream](#downstream)
      - [Serving](#serving)
      - [Quantization](#quantization-1)
      - [Fine-tuning](#fine-tuning-1)
      - [API](#api)
  - [Benchmarks](#benchmarks)
    - [Chat model performance](#chat-model-performance)

## Ecosystem

SEMIKONG has a comprehensive ecosystem, offering a range of tools, services, and models to enrich your experiences and maximize productivity.

- [Upstream](#upstream)
- [Downstream](#downstream)
  - [Serving](#serving)
  - [Quantization](#quantization-1)
  - [Fine-tuning](#fine-tuning-1)
  - [API](#api)

### Upstream

The SEMIKONG series models follow the same model architecture as Llama. By choosing SEMIKONG, you can leverage existing tools, libraries, and resources within the Llama ecosystem, eliminating the need to create new tools and enhancing development efficiency.

For example, the SEMIKONG series models are saved in the format of the Llama model. You can directly use `LlamaForCausalLM` and `LlamaTokenizer` to load the model. For more information, see [Use the chat model](#31-use-the-chat-model).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("pentagoniac/SEMIKONG-8b-GPTQ", use_fast=False)

model = AutoModelForCausalLM.from_pretrained("pentagoniac/SEMIKONG-8b-GPTQ", device_map="auto")
```

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Downstream

> 💡 Tip
> 
> - Feel free to create a PR and share the fantastic work you've built using the SEMIKONG series models.
>
> - To help others quickly understand your work, it is recommended to use the format of `<model-name>: <model-intro> + <model-highlights>`.

#### Serving 

If you want to get up with SEMIKONG in a few minutes, you can use the following services built upon SEMIKONG.

- SEMIKONG-70B-Instruct: you can chat with SEMIKONG using one of the following platforms:
  - [SEMIKONG-70B-Instruct | Hugging Face](https://huggingface.co/pentagoniac/SEMIKONG-70B)
  - [SEMIKONG-70B-Instruct | SEMIKONG Platform](#): 

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

## Tech report

For detailed capabilities of the SEMIKONG series model, see [SemiKong Paper](#).

### Citation

```bibtex
@article{semikong2024,
  title={SemiKong: Curating, Training, and Evaluating A Semiconductor Industry-Specific Large Language Model},
  author={Christopher Nguyen, William Nguyen, Atsushi Suzuki, Daisuke Oku, Hong An Phan, Sang Dinh, Zooey Nguyen, Anh Ha, Shruti Raghavan, Huy Vo, Thang Nguyen, Lan Nguyen, Yoshikuni Hirayama},
  journal={arXiv preprint arXiv:2411.13802},
  year={2024}
}
```

## Benchmarks 

- [Chat model performance](#chat-model-performance)

### Chat model performance

SEMIKONG-70B-Chat model demonstrates exceptional performance, ranking first among all existing open-source models in the benchmarks including MMLU, CMMLU, BBH, GSM8k, and more.

![Chat model performance]() 

<details>
<summary> Evaluation methods and challenges. ⬇️ </summary>
</details>

# Who can use SEMIKONG?

Everyone! 🙌 ✅

The code and weights of the SEMIKONG series models are distributed under the [Apache 2.0 license](https://github.com/01-ai/SEMIKONG/blob/main/LICENSE), which means the SEMIKONG series models are free for personal usage, academic purposes, and commercial use. 

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

# Misc.

### Contributions

This project is the result of a collaborative effort involving multiple companies and individuals:

- Tokyo Electron: Atsushi Suzuki, Daisuke Oku
- FPT Software AIC: [Huy Vo](https://github.com/sitloboi2012), Thang Nguyen, [Lan Nguyen](https://www.linkedin.com/in/lan-nguyen-b7bb2517/)
- Aitomatic: [William Nguyen](https://github.com/nguyennm1024), [Vinh Luong](https://github.com/LuongTheVinh), [Christopher Nguyen](https://github.com/ctn).
- AI Alliance members and researchers

We would like to express our gratitude to the AI Alliance (https://thealliance.ai) for providing the impetus, resources, and platform for this work, and for collaboration in open science. We also extend our thanks to the member organizations of the AI Alliance, their researchers and engineers for their valuable contributions to this study, including:

- Noritaka Yokomori (Tokyo Electron)
- Anthony Annunziata (IBM Research)
- Sean Hughes (ServiceNow)
- Phong Nguyen (FPT Software, AI Center)

Their expertise, insights, and collaborative spirit have been instrumental in advancing our research.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### Disclaimer

We use data compliance checking algorithms during the training process, to
ensure the compliance of the trained model to the best of our ability. Due to
complex data and the diversity of language model usage scenarios, we cannot
guarantee that the model will generate correct, and reasonable output in all
scenarios. Please be aware that there is still a risk of the model producing
problematic outputs. We will not be responsible for any risks and issues
resulting from misuse, misguidance, illegal usage, and related misinformation,
as well as any associated data security concerns.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>

### License

The code and weights of the SEMIKONG series models are distributed under the [Apache 2.0 license](https://github.com/01-ai/SEMIKONG/blob/main/LICENSE).

If you create derivative works based on this model, please include the following attribution in your derivative works:

    This work is a derivative of SEMIKONG by AI Alliance, used under the Apache 2.0 License.

<p align="right"> [
  <a href="#top">Back to top ⬆️ </a>  ] 
</p>
