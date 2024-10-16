import torch
print(torch.cuda.is_available())  # 應該輸出 True
print(torch.cuda.current_device())  # 應該顯示當前使用的 GPU 設備 ID
