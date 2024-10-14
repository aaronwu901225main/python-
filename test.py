from datasets import load_dataset

# 加载 CSV 文件到 Hugging Face Dataset
data_files = {
    "train": "C:\\Users\\AaronWu\\Desktop\\folder\\暫存\\titanic\\train.csv",
    "test": "C:\\Users\\AaronWu\\Desktop\\folder\\暫存\\titanic\\test.csv"
}

# 加载数据集
dataset = load_dataset('csv', data_files=data_files)

# 将数据集推送到 Hugging Face Hub
dataset.push_to_hub("dataset")

