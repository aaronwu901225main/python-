import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class PineappleDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))

        # 讀圖片
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # 讀標註
        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id, cx, cy, w, h = map(float, parts)
                xmin = (cx - w / 2) * width
                ymin = (cy - h / 2) * height
                xmax = (cx + w / 2) * width
                ymax = (cy + h / 2) * height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id))  # 鳳梨類別，應該都為0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_transform():
    return T.Compose([
        T.ToTensor()
    ])

# 使用範例
# dataset = PineappleDataset('project/images/train', 'project/labels/train', transforms=get_transform())
# img, target = dataset[0]
