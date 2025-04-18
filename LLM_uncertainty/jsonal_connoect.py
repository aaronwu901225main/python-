import os
import json

def load_json_files_from_directory(directory):
    # 用來儲存所有載入的資料
    all_data = []

    # 遞迴處理指定目錄
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".json"):  # 檢查檔案是否為 json 檔
                file_path = os.path.join(root, file_name)
                print(f"正在處理 {file_path}...")

                # 載入 json 檔案
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        all_data.append(data)
                except Exception as e:
                    print(f"處理 {file_path} 時發生錯誤: {e}")

    return all_data

# 指定目錄
directory = "./LLM_COT"

# 載入所有 json 檔案
merged_data = load_json_files_from_directory(directory)

# 顯示載入的數據量
print(f"總共載入了 {len(merged_data)} 筆資料。")

# 如果需要，您可以將這些資料進一步處理或儲存成一個新的檔案
# 例如，將所有資料合併成一個新的 json 檔案
output_file = "merged_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)
print(f"合併後的資料已儲存至 {output_file}。")