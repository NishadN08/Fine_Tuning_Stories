# %%
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.notebook import tqdm  # Nice Jupyter progress bars

from torch.cuda.amp import autocast, GradScaler

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time, threading
import pynvml
import matplotlib.pyplot as plt

# %%
# Logging GPU
pynvml.nvmlInit()
gpu_handles  = [pynvml.nvmlDeviceGetHandleByIndex(i)
                 for i in range(torch.cuda.device_count())]
timestamps, util_all, mem_all, temp_all = [], [], [], []

timestamps, gpu_util, mem_used, gpu_temp = [], [], [], []

def log_gpu():
    now = time.time()
    util_all.append([pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in gpu_handles])
    mem_all .append([pynvml.nvmlDeviceGetMemoryInfo(h).used/1024**2 for h in gpu_handles])
    temp_all.append([pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                     for h in gpu_handles])
    timestamps.append(now)

# background logger thread
stop_flag = False
def monitor(interval=1):
    while not stop_flag:
        log_gpu()
        time.sleep(interval)
threading.Thread(target=monitor, daemon=True).start()



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
#Check for devices(GPUs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


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
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=192)

tokenized_dataset = dataset.map(tokenize, batched=True,remove_columns=["text"])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# %%

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#Trainer, Dataloader and Optimizer
train_loader = torch.utils.data.DataLoader(
    tokenized_dataset,
    batch_size=96,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=2
)

# %%
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scaler    = GradScaler()                 # fp16 helper

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

# %%
# Train
##trainer.train()
stepwise_train_loss = []

gradient_accum_steps = 2
num_epochs = 1
step_counter = 0

stop_flag = False
thread = threading.Thread(target=monitor_gpu)
thread.start()

log_gpu()  # Before training

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    train_batches = 0
    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    optimizer.zero_grad(set_to_none=True)

    for step, batch in progress:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with autocast():                       # fp16 forward & loss calc
            outputs = model(**batch)
            loss    = outputs.loss.mean() / gradient_accum_steps

        scaler.scale(loss).backward()

        total_train_loss += loss.item() * gradient_accum_steps
        train_batches += 1
        step_counter += 1

        if (step + 1) % gradient_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Log training loss every batch
        avg_loss = total_train_loss / train_batches
        stepwise_train_loss.append((step_counter, avg_loss))
        progress.set_postfix(train_loss=loss.item() * gradient_accum_steps)


# %%
stop_flag = True
thread.join()

log_gpu()  # After training

# %%
# Ensure model is in evaluation mode and on the correct device
model.eval()

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)

log_gpu()  # Before final generation
# %%

# Generate text
with torch.no_grad(), autocast():
    outputs = model.module.generate(
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

elapsed = [t - timestamps[0] for t in timestamps]
g0_util = [u[0] for u in util_all]
g1_util = [u[1] for u in util_all]
g0_mem  = [m[0] for m in mem_all]
g1_mem  = [m[1] for m in mem_all]
g0_temp = [t[0] for t in temp_all]
g1_temp = [t[1] for t in temp_all]

fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

# Utilization %
ax[0].plot(elapsed, g0_util, label="GPU-0 util %")
ax[0].plot(elapsed, g1_util, label="GPU-1 util %")
ax[0].set_ylabel("Utilisation (%)"); ax[0].grid(); ax[0].legend()

# Memory usage
ax[1].plot(elapsed, g0_mem, label="GPU-0 MB")
ax[1].plot(elapsed, g1_mem, label="GPU-1 MB")
ax[1].set_ylabel("Memory (MB)"); ax[1].set_xlabel("Seconds"); ax[1].grid(); ax[1].legend()

# Temperature
ax[2].plot(elapsed, g0_temp, label="GPU-0 temp (°C)", color="red")
ax[2].plot(elapsed, g1_temp, label="GPU-1 temp (°C)", color="orange")
ax[2].set_ylabel("Temp (°C)")
ax[2].set_xlabel("Time (s)")
ax[2].grid()
ax[2].legend()
plt.suptitle("GPU usage during LoRA fine-tune", fontsize=14); plt.tight_layout(); plt.show()


# %%
# Unpack steps and losses
train_steps, train_vals = zip(*stepwise_train_loss)

# Optional: downsample every n steps if plotting is slow
# downsample_every = 10
# train_steps = train_steps[::downsample_every]
# train_vals = train_vals[::downsample_every]

plt.figure(figsize=(8,4))
plt.plot(train_steps, train_vals, marker='.', linewidth=1)
plt.title("Training loss per batch"); plt.xlabel("Global step"); plt.ylabel("Loss")
plt.grid(); plt.tight_layout(); plt.show()


# %%

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(1)

info = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(f"Total GPU memory: {info.total / 1024**2:.2f} MB")
print(f"Used GPU memory:  {info.used / 1024**2:.2f} MB")
print(f"Free GPU memory:  {info.free / 1024**2:.2f} MB")

pynvml.nvmlShutdown()

# %%
perplexity = torch.exp(torch.tensor(avg_loss))

print(f"Training loss: {avg_loss:.4f}")
print(f"Training perplexity: {perplexity:.4f}")