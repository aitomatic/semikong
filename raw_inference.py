import argparse

import torch
import transformers
import yaml
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)


def load_config(config_file="./configs/inference-config.yaml"):
    """
    Load the configuration from a YAML file.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def configure_bnb_4bit(config):
    """
    Configures the 4-bit quantization settings for the model.
    """
    compute_dtype = getattr(torch, config["model"]["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=config["model"]["use_4bit"],
        bnb_4bit_quant_type=config["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["model"]["use_nested_quant"]
    )


def load_model_and_tokenizer(config, bnb_config):
    """
    Loads the model and tokenizer with the given configuration.
    """
    model_name = config["model"]["model_name"]
    device_map = config["model"]["device_map"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def configure_lora(config):
    """
    Configures the LoRA parameters for model fine-tuning.
    """
    lora_params = config["lora"]
    return LoraConfig(
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
        r=lora_params["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def text_gen_eval_wrapper(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """
    A wrapper function for inferencing, generating text based on a prompt.
    """
    # Suppress logging
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)

    # Initialize text generation pipeline
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        do_sample=True,
        temperature=temperature
    )

    # Generate text from prompt
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    generated_text = result[0]['generated_text']

    # Extract text after the prompt
    index = generated_text.find("[/INST] ")
    return generated_text[index + len("[/INST] "):].strip() if index != -1 else generated_text.strip()


def main():
    # Setup argparse to receive the config file path
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument("--config", type=str, default="./configs/inference-config.yaml", help="Path to the config file")

    # Parse the arguments
    args = parser.parse_args()

    # Load configuration from the file specified by the --config argument
    config = load_config(args.config)

    # Configure 4-bit settings
    bnb_config = configure_bnb_4bit(config)

    # Load the base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, bnb_config)

    # Load LoRA configuration and merge it with the model
    configure_lora(config)
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["model_name"],
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=config["model"]["device_map"],
    )

    model = PeftModel.from_pretrained(base_model, config["model"]["output_dir"])
    model = model.merge_and_unload()

    # Perform text generation
    prompt = "tell me about different type of etching in semiconductor"
    generated_text = text_gen_eval_wrapper(model, tokenizer, prompt, max_length=200, temperature=0.7)
    print(generated_text)


if __name__ == "__main__":
    main()
