from unsloth import FastLanguageModel
import re
import os
os.makedirs("accessed",exist_ok=True)
model_id = "./outputs/checkpoint-717"
#model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

CONTEXT_LENGTH = 15000

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = CONTEXT_LENGTH,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

prompt_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Please read the following question and its reasoning process, and determine whether the reasoning is valid.
The question is between <question> and  </question> tags, and the reasoning is between <reasoning> and </reasoning> tags.

<question>
{}
</question>

<reasoning>
{}
</reasoning>

If the reasoning is logically sound and correctly supports the answer to the question, output 'Accept'.
If the reasoning is flawed, inconsistent, or does not properly support the answer, output 'Reject'.
Only output 'Accept' or 'Reject'.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def extract_response(response: str) -> str:
    match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>', response, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_question(question: str) -> str:
    # 移除前綴與後綴
    prefix = "Solve the following multiple choices question:\n"
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}. For example, if the answer is A, please write \\boxed{A}."
    # 去掉前後指定內容
    return question[len(prefix): -len(suffix)]


import glob
import json

data = []
json_files = glob.glob("test_cot/*.json")
for file_path in json_files:
    with open(file_path, "r") as f:
        file_data = json.load(f)
        if isinstance(file_data, list):
            data.extend(file_data)
        else:
            data.append(file_data)

print(f"Loaded {len(json_files)} JSON files:")
for path in json_files:
    print(" -", path)


correctly_accepted = 0
correctly_rejected = 0
leakage = 0
overkill = 0
for i, d in enumerate(data):
    question = extract_question(d["question"])
    cot = d["CoT"]

    prompt = prompt_template.format(question, cot)

    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(response)
    response = extract_response(response)
    print("\n\nExtracted response:", response)
    print("=" * 50)
    if response == "Accept" or response == "Accept.":
        if d["label"] == "1":
            correctly_accepted += 1
        else:
            leakage += 1
    elif response == "Reject" or response == "Reject.":
        if d["label"] == "0":
            correctly_rejected += 1
        else:
            overkill += 1

    data[i]["assessor_response"] = response

print("correctly_accepted:", correctly_accepted,"\n")
print("correctly_rejected:", correctly_rejected,"\n")
print("leakage:", leakage,"\n")
print("overkill:", overkill,"\n")
print("Total:", correctly_accepted + correctly_rejected + leakage + overkill, len(data))
name=model_id.split("/")[-1]
with open(f"accessed/{name}_test_assessor_response.json", "w") as f:
    json.dump(data, f, indent=4)