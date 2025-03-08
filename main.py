# Code written by Asadullah Hamzah
# Licensed under MIT
# If you use, modify or whatever this code, please feel free to also promote this repo ♥
# Script tested on a device running Windows 10 with 64GB RAM + RTX4060 8GB; VRAM 8GB native + 22GB Unified CUDA

# constants for you to set
epochs = 1 # more takes more
batchsize = 1 # per device
gradientaccum = 2 # if your context length is already too low, prefer lowering this
maxctxlen = 4 * 1024

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

print("Imported libraries.\n")

# Load model & tokenizer
modelname = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
print(f"Loaded {modelname} successfully.\n")

# Expectes a [ { "instruction": "That's the system prompt", "input": "user input", "output": "expected output" }, repeat ] / Alpaca format
rawdataset = load_dataset("json", data_files="data.json", streaming=False) # or data_files={"train": "data.json"}, streaming false to avoid crashing halfway through training

# ChatML
def merge_fields(example):
    return {"text": f"<|im_start|>system\n{example['instruction']}\n<|im_end|>\n<|im_start|>user\n{example['input']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"}
merged_dataset = rawdataset.map(merge_fields)

def tokenize(examples):
    encodings = tokenizer(examples["text"], truncation=True, padding="longest", max_length=maxctxlen) # or whatever context length you can afford (this will take 30GB VRAM, unified is supported)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings
dataset = merged_dataset["train"].map(tokenize, batched=True, remove_columns=["instruction", "input", "output"])
samples = len(dataset)
maxsteps = (samples // (batchsize * gradientaccum)) * epochs
print(f"Found {samples} samples / rows, logically results to {maxsteps} steps.")

config = TrainingArguments(
    output_dir=".PROSXIMA_OUTPUT",
    per_device_train_batch_size=batchsize,
    gradient_accumulation_steps=gradientaccum,
    num_train_epochs=epochs,
    logging_steps=1,
    save_steps=100,
    eval_strategy="no",
    save_total_limit=4,
    fp16=False,
    bf16=True,
    lr_scheduler_type="linear",
    report_to="none",
    max_steps=maxsteps,
)

# Simple SGD Trainer
class SGDTrainer(Trainer):
    def create_optimizer(self):
        super().create_optimizer()
        return torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.95)

trainer = SGDTrainer(model=model, args=config, train_dataset=dataset)
print("Starting training...\n")
trainer.train()
