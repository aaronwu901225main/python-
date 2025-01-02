import os
import json
from PyPDF2 import PdfReader
import re

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    content = []
    for page in reader.pages:
        content.append(page.extract_text())
    return " ".join(content)  # 合併所有頁面的文字

def split_into_sentences(text):
    """Split text into sentences using regular expressions."""
    # 使用正則表達式分割句子
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def generate_json(sentences):
    """Convert sentences into JSON format."""
    json_data = [{"content": sentence} for sentence in sentences]
    return json_data

def process_pdfs(pdf_folder, output_json):
    """Process all PDFs in a folder and save extracted sentences to a JSON file."""
    all_data = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}...")
            pdf_path = os.path.join(pdf_folder, filename)
            content = extract_text_from_pdf(pdf_path)
            sentences = split_into_sentences(content)
            json_data = generate_json(sentences)
            all_data.extend(json_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
    print(f"JSON saved to {output_json}")

# 使用方式
pdf_folder = "papers/"  # 替換為PDF所在資料夾路徑
output_json = "dataset/train.json"  # 輸出JSON檔案名稱
process_pdfs(pdf_folder, output_json)
