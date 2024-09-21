# import os
# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     HfArgumentParser,
#     TrainingArguments,
#     pipeline,
#     logging,
# )
# from peft import LoraConfig, PeftModel
# from trl import SFTTrainer

# # Used for multi-gpu
# local_rank = -1
# per_device_train_batch_size = 4
# per_device_eval_batch_size = 4
# gradient_accumulation_steps = 1
# learning_rate = 2e-4
# max_grad_norm = 0.3
# weight_decay = 0.001
# lora_alpha = 16
# lora_dropout = 0.1
# lora_r = 64
# max_seq_length = None

# # The model that you want to train from the Hugging Face hub
# model_name = "model_path_folder or model_name_hf"

# # Fine-tuned model name
# new_model = "semikong-8b"

# # The instruction dataset to use
# dataset_name = "dataset_path_folder or dataset_name_hf"

# # Activate 4-bit precision base model loading
# use_4bit = True

# # Activate nested quantization for 4-bit base models
# use_nested_quant = False

# # Compute dtype for 4-bit base models
# bnb_4bit_compute_dtype = "float16"

# # Quantization type (fp4 or nf4)
# bnb_4bit_quant_type = "nf4"

# # Number of training epochs
# num_train_epochs = 2

# # Enable fp16 training, (bf16 to True with an A100)
# fp16 = False

# # Enable bf16 training
# bf16 = False

# # Use packing dataset creating
# packing = False

# # Enable gradient checkpointing
# gradient_checkpointing = True

# # Optimizer to use, original is paged_adamw_32bit
# optim = "paged_adamw_32bit"

# # Learning rate schedule (constant a bit better than cosine, and has advantage for analysis)
# lr_scheduler_type = "cosine"

# # Number of optimizer update steps, 10K original, 20 for demo purposes
# max_steps = -1

# # Fraction of steps to do a warmup for
# warmup_ratio = 0.03

# # Group sequences into batches with same length (saves memory and speeds up training considerably)
# group_by_length = True

# # Save checkpoint every X updates steps
# save_steps = 10

# # Log every X updates steps
# logging_steps = 1

# # The output directory where the model predictions and checkpoints will be written
# output_dir = "./results"

# # Load the entire model on the GPU 0
# device_map = {"": 0}

# def load_model(model_name):
#     # Load tokenizer and model with QLoRA configuration
#     compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=use_4bit,
#         bnb_4bit_quant_type=bnb_4bit_quant_type,
#         bnb_4bit_compute_dtype=compute_dtype,
#         bnb_4bit_use_double_quant=use_nested_quant,
#     )

#     if compute_dtype == torch.float16 and use_4bit:
#         major, _ = torch.cuda.get_device_capability()
#         if major >= 8:
#             print("=" * 80)
#             print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
#             print("=" * 80)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map=device_map,
#         quantization_config=bnb_config
#     )

#     model.config.use_cache = False
#     model.config.pretraining_tp = 1

#     # Load LoRA configuration
#     peft_config = LoraConfig(
#         lora_alpha=lora_alpha,
#         lora_dropout=lora_dropout,
#         r=lora_r,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )

#     # Load Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "right"

#     return model, tokenizer, peft_config

# def format_dolly(sample):
#     instruction = f"<s>[INST] {sample['instruction']}"
#     context = f"Here's some context: {sample['input']}" if len(sample["input"]) > 0 else None
#     response = f" [/INST] {sample['output']}"
#     # join all the parts together
#     prompt = "".join([i for i in [instruction, context, response] if i is not None])
#     return prompt

# # template dataset to add prompt to each sample
# def template_dataset(sample):
#     sample["text"] = f"{format_dolly(sample)}{TOKENIZER.eos_token}"
#     return sample

# MODEL, TOKENIZER, PEFT_CONFIG = load_model(model_name)

# dataset = load_dataset("json", data_files=dataset_name, split="train")
# dataset_shuffled = dataset.shuffle(seed=42)
# # Select the first 50 rows from the shuffled dataset, comment if you want 15k
# dataset = dataset_shuffled.select(range(50))
# dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))

# training_arguments = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=per_device_train_batch_size,
#     gradient_accumulation_steps=gradient_accumulation_steps,
#     optim=optim,
#     save_steps=save_steps,
#     logging_steps=logging_steps,
#     learning_rate=learning_rate,
#     fp16=fp16,
#     bf16=bf16,
#     max_grad_norm=max_grad_norm,
#     max_steps=max_steps,
#     warmup_ratio=warmup_ratio,
#     group_by_length=group_by_length,
#     lr_scheduler_type=lr_scheduler_type,
# )

# trainer = SFTTrainer(
#     model=MODEL,
#     train_dataset=dataset,
#     peft_config=PEFT_CONFIG,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     tokenizer=TOKENIZER,
#     args=training_arguments,
#     packing=packing,
# )

# trainer.train()
# trainer.model.save_pretrained(output_dir)

import os
import torch
import yaml
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Function to load the configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Load model and tokenizer based on configuration
def load_model(config):
    compute_dtype = getattr(torch, config["model"]["bnb_4bit_compute_dtype"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["use_4bit"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["model"]["use_nested_quant"],
    )

    model_name = config["model"]["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=config["model"]["device_map"],
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# Configure LoRA settings
def configure_lora(config):
    lora_params = config["lora"]
    return LoraConfig(
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
        r=lora_params["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )

# Format data into the instruction template
def format_dolly(sample):
    instruction = f"<s>[INST] {sample['instruction']}"
    context = f"Here's some context: {sample['input']}" if len(sample["input"]) > 0 else None
    response = f" [/INST] {sample['output']}"
    return "".join([i for i in [instruction, context, response] if i is not None])

# Template dataset mapping
def template_dataset(sample, tokenizer):
    sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}"
    return sample

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA and 4-bit precision.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load model and tokenizer
    model, tokenizer = load_model(config)

    # Configure LoRA
    peft_config = configure_lora(config)

    # Load and process the dataset
    dataset_name = config["model"]["dataset_name"]
    dataset = load_dataset("json", data_files=dataset_name, split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(50))  # Optional: select first 50 rows for demo
    dataset = dataset.map(lambda sample: template_dataset(sample, tokenizer), remove_columns=list(dataset.features))

    # Set up training arguments
    training_arguments = TrainingArguments(
        output_dir=config["model"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        optim=config["training"]["optim"],
        save_steps=config["training"]["save_steps"],
        logging_steps=config["training"]["logging_steps"],
        learning_rate=config["training"]["learning_rate"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        max_grad_norm=config["training"]["max_grad_norm"],
        max_steps=config["training"]["max_steps"],
        warmup_ratio=config["training"]["warmup_ratio"],
        group_by_length=config["training"]["group_by_length"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
    )

    # Initialize and start training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["training"]["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config["training"]["packing"],
    )

    trainer.train()
    trainer.model.save_pretrained(config["model"]["output_dir"])

if __name__ == "__main__":
    main()
