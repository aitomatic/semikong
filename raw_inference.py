import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Used for multi-gpu
local_rank = -1
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
weight_decay = 0.001
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64
max_seq_length = None

# The model that you want to train from the Hugging Face hub
model_name = "model_path_folder or model_name_hf"

# Fine-tuned model name
new_model = "semikong-8b"

# The instruction dataset to use
dataset_name = "dataset_path_folder or dataset_name_hf"

# Activate 4-bit precision base model loading
use_4bit = True

# Activate nested quantization for 4-bit base models
use_nested_quant = False

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Number of training epochs
num_train_epochs = 2

# Enable fp16 training, (bf16 to True with an A100)
fp16 = False

# Enable bf16 training
bf16 = False

# Use packing dataset creating
packing = False

# Enable gradient checkpointing
gradient_checkpointing = True

# Optimizer to use, original is paged_adamw_32bit
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine, and has advantage for analysis)
lr_scheduler_type = "cosine"

# Number of optimizer update steps, 10K original, 20 for demo purposes
max_steps = -1

# Fraction of steps to do a warmup for
warmup_ratio = 0.03

# Group sequences into batches with same length (saves memory and speeds up training considerably)
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 10

# Log every X updates steps
logging_steps = 1

# The output directory where the model predictions and checkpoints will be written
output_dir = "./results"

# Load the entire model on the GPU 0
device_map = {"": 0}

def load_model(model_name):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config

def text_gen_eval_wrapper(model, tokenizer, prompt, model_id=1, show_metrics=True, temp=0.7, max_length=200):
    """
    A wrapper function for inferencing, evaluating, and logging text generation pipeline.

    Parameters:
        model (str or object): The model name or the initialized text generation model.
        tokenizer (str or object): The tokenizer name or the initialized tokenizer for the model.
        prompt (str): The input prompt text for text generation.
        model_id (int, optional): An identifier for the model. Defaults to 1.
        show_metrics (bool, optional): Whether to calculate and show evaluation metrics.
                                       Defaults to True.
        max_length (int, optional): The maximum length of the generated text sequence.
                                    Defaults to 200.

    Returns:
        generated_text (str): The generated text by the model.
        metrics (dict): Evaluation metrics for the generated text (if show_metrics is True).
    """
    # Suppress Hugging Face pipeline logging
    logging.set_verbosity(logging.CRITICAL)

    # Initialize the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temp)

    # Generate text using the pipeline
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    generated_text = result[0]['generated_text']

    # Find the index of "### Assistant" in the generated text
    index = generated_text.find("[/INST] ")
    if index != -1:
        # Extract the substring after "### Assistant"
        substring_after_assistant = generated_text[index + len("[/INST] "):].strip()
    else:
        # If "### Assistant" is not found, use the entire generated text
        substring_after_assistant = generated_text.strip()

    return substring_after_assistant


# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

model = PeftModel.from_pretrained(base_model, output_dir)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

prompt="tell me about different type of etching in semiconductor"
print(text_gen_eval_wrapper(model, tokenizer, prompt, show_metrics=False))