from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
import torch
import re
from datetime import datetime
import json
import os

TEST_SIZE = 'all'

MODEL_ID = "./LLm_accessor_merged_model"

REPEAT = 1
CONTEXT_LENGTH = 32000
MAX_TOKENS = 30000
TEMPERATURE = 0.6
TOP_P = 0.95

start_time = datetime.now()
filename = start_time.strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("./accessed", exist_ok=True)

llm = LLM(
    model=MODEL_ID,
    dtype=torch.float16,
    trust_remote_code=True,
    quantization='bitsandbytes',
    load_format='bitsandbytes',
    max_model_len=CONTEXT_LENGTH,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

root_dir = "./test_cot"
# 設為資料夾中所有資料
subjects = [name for name in os.listdir(root_dir)]
print(subjects)

for subject in subjects:
    print(f"\n🧪 Evaluating subject: {subject}")
    data = load_from_disk(os.path.join(root_dir, subject))
    
    if TEST_SIZE != 'all':
        data = data.select(range(TEST_SIZE))
    
    def extract_boxed_answer(text: str) -> str:
        matches = re.findall(r'\\boxed\{(.*?)\}', text)
        return matches[-1].strip() if matches else None
    
    params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    
    results = {"Correctly_Accept": 0, "Correctly_Reject": 0, "Leakage": 0, "Overkill": 0}
    choices = ['Accept', 'Reject']
    correct_accessor = []
    wrong_accessor = []
    
    for i, d in enumerate(data):
        user_prompt = (
            f"Read the following multiple choices question:\n{d['question']}\n"
            f"And then Check the reasoning process:\n{d['CoT']}\n"
            f"Is the reasoning process valid?\n"
            f"Please answer with 'Accept' or 'Reject'.\n"
            "And put your final answer within \\boxed{}.For example if the reasoning process is valid ,then output \\boxed{Accept}\n"
        )
    
        prompt =f"<｜begin▁of▁sentence｜><｜User｜>{user_prompt}\n<｜Assistant｜><think>"

        output1 = llm.generate(prompt, params)
        
        response_1 = output1[0].outputs[0].text +'\nFinal answer: \\boxed{'

        output2 = llm.generate(prompt + response_1, params)

        response_2 = output2[0].outputs[0].text
        
        response = response_1 + response_2

        extracted_answer = extract_boxed_answer(response)
    
        print(prompt + response)
        print(f'\n{subject} Question {i + 1}/{len(data)}')
        print(f'res toks 1: {len(output1[0].outputs[0].token_ids)}')
        print(f'res toks 2: {len(output2[0].outputs[0].token_ids)}')
        print(f'Extracted answer: {extracted_answer}')
        # 如果 label 是 1，則答案是 Accept，否則是 Reject
        if d["label"] == 1:
            print(f'True answer: {choices[0]}')
        else:
            print(f'True answer: {choices[1]}')

        true_answer = choices[d["label"]]  # label: 1 for Accept, 0 for Reject
        cot = {
            'question': user_prompt,
            'CoT': '<think>' + response,
            'extracted_answer': extracted_answer,
            'true_answer': true_answer,
        }

        if extracted_answer == true_answer:
            if extracted_answer == 'Accept':
                labels = "Correctly Accept"
                results["Correctly_Accept"] += 1
            else:
                labels = "Correctly Reject"
                results["Correctly_Reject"] += 1
            correct_accessor.append({**cot, "label": labels})
        else:
            if extracted_answer == 'Accept':
                labels = "Leakage"
                results["Leakage"] += 1
            else:
                labels = "Overkill"
                results["Overkill"] += 1
            wrong_accessor.append({**cot, "label": labels})

    
        print(f"Current results: {results}")
        print('Spend time:', datetime.now() - start_time)
        print('=' * 20)
        
    # 儲存各科目結果（這是新增的）
    with open(f"./report/correct_{subject}_{filename}.json", "w") as f:
        json.dump(correct_accessor, f, indent=4)

    with open(f"./report/wrong_{subject}_{filename}.json", "w") as f:
        json.dump(wrong_accessor, f, indent=4)

    # 重設資料以進行下一個 subject 的處理
    correct_accessor = []
    wrong_accessor = []
    results = {"Correctly_Accept": 0, "Correctly_Reject": 0, "Leakage": 0, "Overkill": 0}
