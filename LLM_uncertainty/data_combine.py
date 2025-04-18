import os
import json
import argparse
from glob import glob

def load_all_json_from_dir(data_dir):
    merged_data = []
    for file_path in glob(os.path.join(data_dir, "*.json")):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                items = json.load(f)
                for item in items:
                    question = item.get("question", "").strip()
                    cot = item.get("CoT", "").strip()
                    label = item.get("label")
                    if label not in [0, 1]:
                        continue
                    prompt = f"### Question:\n{question}\n\n### Reasoning:\n{cot}"
                    response = "accept" if label == 1 else "reject"
                    merged_data.append({
                        "input": prompt,
                        "output": f"### Response: {response}"
                    })
            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")
    return merged_data

def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="資料夾路徑，裡面放一堆json")
    parser.add_argument("--output_path", type=str, default="merged_dataset.jsonl", help="輸出的jsonl路徑")
    args = parser.parse_args()

    all_data = load_all_json_from_dir(args.data_dir)
    save_to_jsonl(all_data, args.output_path)
    print(f"✅ 轉換完成，共 {len(all_data)} 筆資料，已儲存至 {args.output_path}")
