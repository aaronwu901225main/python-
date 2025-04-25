import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T
import pandas as pd
import os
from PIL import Image

def load_model(model_path, device):
    backbone = resnet_fpn_backbone('resnet50', weights="IMAGENET1K_V1")
    model = FasterRCNN(backbone, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, img_tensor, device, threshold=0.5):
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])
    boxes = prediction[0]['boxes'].cpu()
    scores = prediction[0]['scores'].cpu()
    selected_boxes = boxes[scores >= threshold]
    return selected_boxes

def get_ground_truth(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return len(lines)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = "fasterrcnn_pineapple.pth"
    test_images_dir = "project/images/test"
    test_labels_dir = "project/labels/test"
    output_csv = "inference_results.csv"

    model = load_model(model_path, device)

    transform = T.Compose([
        T.ToTensor()
    ])

    results = []

    image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
    image_files.sort()

    for img_file in image_files:
        img_path = os.path.join(test_images_dir, img_file)
        label_path = os.path.join(test_labels_dir, img_file.replace('.jpg', '.txt'))

        # 預測
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image)
        predicted_boxes = predict(model, img_tensor, device)
        pred_count = len(predicted_boxes)

        # Ground Truth
        gt_count = get_ground_truth(label_path)

        # 紀錄結果
        results.append({
            'filename': img_file,
            'gt_count': gt_count,
            'pred_count': pred_count
        })

        print(f"{img_file} - GT: {gt_count}, Predicted: {pred_count}")

    # 存成 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"推論完成！結果已儲存到 {output_csv}")

if __name__ == "__main__":
    main()
