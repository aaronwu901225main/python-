import os
import json
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """從PDF文件中提取文本內容"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip()

def convert_pdfs_to_json(input_folder, output_file):
    """將PDF文件資料夾轉換為JSON格式"""
    data = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(input_folder, filename)
            print(f"正在處理文件: {filename}")
            pdf_text = extract_text_from_pdf(file_path)
            data.append({"content": pdf_text})
    
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print(f"轉換完成！JSON 檔案儲存於 {output_file}")

# 使用方法
input_folder = "papers/"  # PDF文件存放資料夾
output_file = "dataset/train.json"
convert_pdfs_to_json(input_folder, output_file)