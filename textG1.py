# %%
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from tqdm.notebook import tqdm  # Nice Jupyter progress bars

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math

import time
import pynvml
import matplotlib.pyplot as plt


# %%
# To log the gpu usage
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(1)
total_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2  # in MB


timestamps, gpu_util, mem_used, gpu_temp = [], [], [], []

def log_gpu():
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    timestamps.append(time.time())
    gpu_util.append(util.gpu)
    mem_used.append(mem.used / 1024**2)
    gpu_temp.append(temp)





# %%

# Load the dataset
dataset = load_dataset("roneneldan/TinyStories",split = "train")
print(dataset[6])

# %%

# Load GPT-2 model and tokenizer
model_name =   "distilgpt2" #or "EleutherAI/gpt-neo-125M" "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
# %%

# Use first 1000 samples for pseudo-evaluation
eval_dataset = dataset

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=192)

eval_tokenized = eval_dataset.map(tokenize, batched=True, remove_columns=["text"])
eval_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

from torch.utils.data import DataLoader
eval_loader = DataLoader(
    eval_tokenized,
    batch_size=256,
    shuffle=False,
    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    num_workers=2
)

# %%
# Ensure model is in evaluation mode and on the correct device

if torch.cuda.device_count() > 1:
    print(f"✨ Using {torch.cuda.device_count()} GPUs (DataParallel)")
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# %%

def evaluate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity
# %%

# Get a random story beginning
prompt = dataset[6]["text"].split(".")[0] + "."

print(f"Prompt: {prompt}")

# %%

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

log_gpu()  # Before generation
# %%

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id  # prevent warning
    )

log_gpu()  # After generation
# %%

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Story:\n")
print(generated_text)

base_ppl = evaluate_perplexity(model, eval_loader)
print(f"📦 Base Model Perplexity: {base_ppl:.2f}")
