import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
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
    # This function formats a given sample into a specific template for instructions.
    # Parameters:
    # - sample (dict): A dictionary containing 'instruction', 'input', and 'output' keys.
    # Returns:
    # - str: A formatted string combining the instruction, optional context, and the response.

    instruction = f"<s>[INST] {sample['instruction']}"
    # If there is input context, include it; otherwise, set it to None.
    context = f"Here's some context: {sample['input']}" if len(sample["input"]) > 0 else None
    response = f" [/INST] {sample['output']}"
    
    # Join the instruction, context (if present), and response into a single string.
    return "".join([i for i in [instruction, context, response] if i is not None])

# Template dataset mapping
def template_dataset(sample, tokenizer):
    # This function maps a dataset sample to a text template using a tokenizer.
    # Parameters:
    # - sample (dict): A dictionary containing 'instruction', 'input', and 'output' keys.
    # - tokenizer: A tokenizer object that provides an end-of-sequence (eos) token.
    # Returns:
    # - dict: The input sample updated with a "text" key containing the formatted text.

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
