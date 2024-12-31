import os
import json

def convert_to_json(input_folder, output_file):
    data = []
    # 遍歷資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                # 將每篇文章的內容分段加入 JSON
                data.append({"content": content})
    
    # 將結果儲存為 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"轉換完成！JSON 檔案儲存於 {output_file}")

# 使用方法
input_folder = "papers/"  # 您的論文存放資料夾
output_file = "dataset/train.json"
convert_to_json(input_folder, output_file)