# %% [markdown]
# # 使用 yolo11

# %%
import yaml

data_yaml = {
    'train': './dataset/images/train',
    'val': './dataset/images/val',  
    'test': './dataset/images/test',
    'nc': 1,
    'names': ['pineapple']
}

with open('./pineapple.yaml', 'w') as f:
    yaml.dump(data_yaml, f)

print("✅ pineapple.yaml created!")

# %%
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="pineapple.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="gpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# test the model by test dataset
test_results = model.val(data="pineapple.yaml", split="test")

# evaluate the model accuracy on the test dataset
print(test_results)

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model


