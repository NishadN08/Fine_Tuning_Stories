# %%
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
import evaluate


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import pynvml
import matplotlib.pyplot as plt
import seaborn as sns

# %%

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
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

dataset = load_dataset("roneneldan/TinyStories",split = "train[:15000]")
dataset1 = load_dataset("roneneldan/TinyStories",split = "validation")
print(dataset1[1])

# %%

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # or "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure model is in evaluation mode and on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# %%

# Get a random story beginning
prompt = dataset1[1]["text"].split(".")[0] + "."

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

# %%

log_gpu()  # Before LoRA

# Prepare LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention and projection layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# %%

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

log_gpu()  # After LoRA
# %%

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=340)

tokenized_dataset = dataset.map(tokenize, batched=True,remove_columns=["text"])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Tokenize validation set
tokenized_val_dataset = dataset1.map(tokenize, batched=True, remove_columns=["text"])
tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# %%


# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    eval_strategy="steps",        # enable validation
    eval_steps=100,                      # evaluate every 50 steps
    save_total_limit=1,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_val_dataset,  # add this
    tokenizer=tokenizer,
    data_collator=data_collator
)

# %%
import threading

stop_flag = False

def monitor_gpu(interval=1):
    while not stop_flag:
        log_gpu()
        time.sleep(interval)

thread = threading.Thread(target=monitor_gpu)
thread.start()


log_gpu()  # Before training

# Train
trainer.train()

# %%
stop_flag = True
thread.join()

log_gpu()  # After training

# %%


# Ensure model is in evaluation mode and on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

log_gpu()  # Before final generation
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

log_gpu()  # After final generation

# %%

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Story After Fine-Tuning:\n")
print(generated_text)
# %%


pynvml.nvmlShutdown()
start_time = timestamps[0]
times = [t - start_time for t in timestamps]

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(times, gpu_util, label="GPU Util (%)", color='blue', marker='o')
plt.ylabel("Utilization (%)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(times, mem_used, label="Memory Used (MB)", color='green', marker='x')
plt.axhline(y=total_mem, label=f"Total Memory ({total_mem:.0f} MB)", color='gray', linestyle='--')
plt.ylabel("Memory (MB)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(times, gpu_temp, label="GPU Temp (°C)", color='red', marker='s')
plt.ylabel("Temp (°C)")
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend()

plt.suptitle("GPU Performance During LoRA Fine-Tuning", fontsize=14)
plt.tight_layout()
plt.show()


# %%

# Get logged metrics
logs = trainer.state.log_history

# Extract step, train loss, and eval loss
steps = []
train_losses = []
eval_losses = []

for log in logs:
    if "loss" in log and "step" in log:
        steps.append(log["step"])
        train_losses.append(log["loss"])
    elif "eval_loss" in log:
        eval_losses.append((log["step"], log["eval_loss"]))

# Unpack eval loss steps and values
eval_steps, eval_values = zip(*eval_losses)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label="Training Loss", marker='o')
plt.plot(eval_steps, eval_values, label="Validation Loss", marker='s')
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use 0 for first GPU

info = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(f"Total GPU memory: {info.total / 1024**2:.2f} MB")
print(f"Used GPU memory:  {info.used / 1024**2:.2f} MB")
print(f"Free GPU memory:  {info.free / 1024**2:.2f} MB")

pynvml.nvmlShutdown()


# %%
