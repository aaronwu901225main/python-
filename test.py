from datasets import load_dataset

# 加载 CSV 文件到 Hugging Face Dataset
data_files = {
    "train": "C:\\Users\\AaronWu\\Desktop\\xcat.zip",
    "test": "C:\\Users\\AaronWu\\Desktop\\xcat test result.zip"
}

# 加载数据集
dataset = load_dataset('zip', data_files=data_files)

# 将数据集推送到 Hugging Face Hub
dataset.push_to_hub("test")