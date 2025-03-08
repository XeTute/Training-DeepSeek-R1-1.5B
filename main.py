# Code written by Asadullah Hamzah
# Licensed under MIT
# If you use, modify or whatever this code, please feel free to also promote this repo ♥

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
rawdataset = load_dataset("json", data_files="data.json", streaming=True) # or data_files={"train": "data.json"}

# ChatML
def merge_fields(example):
    return {"text": f"<|im_start|>system\n{example['instruction']}\n<|im_end|>\n<|im_start|>user\n{example['input']}\n<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"}

merged_dataset = rawdataset.map(merge_fields)

def tokenize(examples):
    encodings = tokenizer(examples["text"], truncation=True, padding="longest", max_length=4096) # or whatever context length you can afford (this will take 30GB VRAM, unified is supported)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

dataset = merged_dataset["train"].map(tokenize, batched=True, remove_columns=["instruction", "input", "output"])

config = TrainingArguments(
    output_dir=".PROSXIMA_OUTPUT",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4, # If your input + output pairs are shorter than max_length at def tokenize & you run out of memory, decrease this number
    num_train_epochs=1,
    logging_steps=1,
    save_steps=100,
    evaluation_strategy="no",
    eval_strategy="no",
    save_total_limit=4,
    fp16=False,
    bf16=True,
    lr_scheduler_type="linear",
    report_to="none",
)

# Simple SGD Trainer
class SGDTrainer(Trainer):
    def create_optimizer(self):
        super().create_optimizer()
        return torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.95)

trainer = SGDTrainer(model=model, args=config, train_dataset=dataset)
print("Starting training...\n")
trainer.train()