from datasets import Dataset
import json

# 讀取 JSON 格式資料
with open("train.json", "r") as f:
    data = json.load(f)

# 轉換為 Hugging Face Dataset
dataset = Dataset.from_dict({"content": [item["content"] for item in data]})

# 保存為 Arrow 格式
dataset.save_to_disk("./your_huggingface_dataset")