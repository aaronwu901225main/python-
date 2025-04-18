import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# 你可以在這裡更換模型（例如：meta-llama/Llama-2-7b-chat-hf）
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# 1. 載入資料集
def load_data():
    with open("correct_college_biology_2025-04-18_03-16-49.json") as f1, open("wrong_college_biology_2025-04-18_03-16-49.json") as f2:
        data = json.load(f1) + json.load(f2)
    for sample in data:
        input_text = sample["question"] + "\n" + sample["CoT"]
        label = sample["label"]
        sample["input"] = input_text
        sample["output"] = "approve" if label == 1 else "reject"
    return Dataset.from_list(data)

# 2. 建 tokenizer 和 model，使用 4-bit 量化
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# 3. 設定 LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 資料前處理
def tokenize(sample):
    prompt = f"[INST] {sample['input']} [/INST]"
    target = sample["output"]
    full = prompt + " " + target
    tokenized = tokenizer(full, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

dataset = load_data().map(tokenize)

# 5. 訓練參數
training_args = TrainingArguments(
    output_dir="./lora-bio",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    report_to="none"
)

# 6. 訓練
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
