from unsloth import FastLanguageModel
import torch

model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

max_seq_length = 15000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset
import os
import json

def load_json_files_from_directory(directory):
    # 用來儲存所有載入的資料，所有資料將會放在同一個列表中
    all_data = []
    amount = 0

    # 遞迴處理指定目錄
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".json"):  # 檢查檔案是否為 json 
                # 去除.ipynb_checkpoints資料夾
                if ".ipynb_checkpoints" in root:
                    continue
                amount += 1
                file_path = os.path.join(root, file_name)
                print(f"正在處理 {file_path}...")

                # 載入 json 檔案
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # 假設每個 JSON 檔案的內容本來就是一個列表
                        if isinstance(data, list):
                            all_data.extend(data)  # 將每個檔案的資料合併進 all_data 列表
                        else:
                            all_data.append(data)  # 如果不是列表，直接將資料放入
                except Exception as e:
                    print(f"處理 {file_path} 時發生錯誤: {e}")
    print(f"總共載入了 {amount} 筆資料。")
    return all_data

# 指定目錄
directory = "./LLM_COT"

# 載入所有 json 檔案並合併
merged_data = load_json_files_from_directory(directory)

# 顯示載入的數據量
print(f"總共載入了 {len(merged_data)} 筆資料。")

# 如果需要，您可以將這些資料進一步處理或儲存成一個新的檔案
# 例如，將所有資料合併成一個新的 json 檔案
output_file = "merged_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

dataset = load_dataset("json", data_files="merged_data.json", split="train")

chat_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Please read the following question and its reasoning process, and determine whether the reasoning is valid.
Only output 'Accept' or 'Reject'.

### Question:
{}

### Reasoning:
{}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}
<|eot_id|>"""

def formatting_func(example):
    return chat_template.format(
        example["question"],
        example["CoT"],
        "Accept" if example["label"] == 1 else "Reject",
    )

dataset = dataset.map(lambda x: {"text": formatting_func(x)})

# print(dataset[0]["text"])

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
    data_collator=DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|start_header_id|>assistant<|end_header_id|>\n",
    ),
)

trainer_stats = trainer.train()

# 儲存 LoRA 模型
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, "outputs")
model = model.merge_and_unload()  # 合併 LoRA
model.save_pretrained("./LLm_accessor_merged_model", safe_serialization=True)