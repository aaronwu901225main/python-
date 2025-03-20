import json

def convert_jsonl_to_json(input_file, output_file, repeat=10):
    transformed_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            # 重新構造 JSON
            transformed_entry = {
                "question": data["problem"],
                "cot_answer": f'{data["solution"]}\n\nanswer : <{data["answer"]}>'
            }
            
            # 重複 10 次
            transformed_data.extend([transformed_entry] * repeat)

    # 將轉換後的資料寫入 JSON 檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)

# 使用範例
input_file = "test.jsonl"  # 替換成你的 JSONL 檔案名稱
output_file = "math500_6.json"  # 轉換後的 JSON 檔案名稱

convert_jsonl_to_json(input_file, output_file, repeat=5)
