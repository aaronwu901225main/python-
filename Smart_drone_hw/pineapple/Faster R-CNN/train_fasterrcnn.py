import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms as T
from tqdm import tqdm
from pineapple_dataset import PineappleDataset, get_transform

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 讀取資料集
    train_dataset = PineappleDataset(
        images_dir='project/images/train',
        labels_dir='project/labels/train',
        transforms=get_transform(train=True)
    )
    test_dataset = PineappleDataset(
        images_dir='project/images/test',
        labels_dir='project/labels/test',
        transforms=get_transform(train=False)
    )


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 建 Faster R-CNN 模型
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = FasterRCNN(backbone, num_classes=2)  # 1 類鳳梨 + 1 個背景

    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 100

    # 訓練
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

    print("訓練完成！")

    # 儲存模型
    torch.save(model.state_dict(), "fasterrcnn_pineapple.pth")
    print("模型已儲存為 fasterrcnn_pineapple.pth")

if __name__ == "__main__":
    main()
