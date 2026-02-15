import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import glob

#============================
#1. データセット定義（HSV変換）
#============================

# RGB → HSV 変換
def rgb_to_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32) / 255.0   # 0〜1 に正規化
    return hsv

class ColorDataset(Dataset):
    def __init__(self, img_paths, labels):
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]

        # 画像読み込み
        img = cv2.imread(path)

        # --- 色モデルの前処理（HSV） ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   # (H,W,3) uint8
        hsv = hsv.astype(np.float32) / 255.0         # float32 に統一（重要）

        # OpenCV resize のバグ対策
        hsv = cv2.resize(hsv, (128, 128), interpolation=cv2.INTER_LINEAR)
        hsv = np.ascontiguousarray(hsv, dtype=np.float32)

        # (H,W,C) → (C,H,W)
        hsv = hsv.transpose(2, 0, 1)

        hsv_tensor = torch.from_numpy(hsv)

        return hsv_tensor, label

#============================
#2. CNNモデル定義（色特徴を学習）
#============================

class ColorCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 16x16
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#============================
#3. 学習ループ
#============================
def train_color_model():
    img_paths = glob.glob("dataset/*.jpg")
    labels = [0 if "target" in p else 1 for p in img_paths]

    dataset = ColorDataset(img_paths, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ColorCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for imgs, labels in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "color_model.pth")
train_color_model()

#============================
#4. ONNX 変換
#============================

def export_color_onnx(model_path="color_model.pth", onnx_path="color_model.onnx"):
    model = ColorCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )

    print("ONNX 変換完了:", onnx_path)


if __name__ == "__main__":
    export_color_onnx()


