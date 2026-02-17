import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import torch


# ============================
# 1. データセット定義
# ============================
class FaceDataset(Dataset):
    def __init__(self, img_paths, labels):
        self.img_paths = img_paths
        self.labels = labels

        # 軽度の回転に対応させるためのデータ拡張
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=15),  # ← 回転耐性
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]

        # 画像読み込み
        img = cv2.imread(path)

        # --- 形モデルの前処理 ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        equalized = cv2.resize(equalized, (128, 128))

        # CNNに渡すためにチャンネル次元を追加
        equalized = np.expand_dims(equalized, axis=2)

        img_tensor = self.transform(equalized)

        return img_tensor, label


# ============================
# 2. CNNモデル定義（線・形を認識）
# ============================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
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


# ============================
# 3. 学習ループ
# ============================
def train_model():
    img_paths = glob.glob("dataset/taget/*.jpg")
    labels = [0 if "target" in p else 1 for p in img_paths]

    dataset = FaceDataset(img_paths, labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleCNN(num_classes=2)
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

    torch.save(model.state_dict(), "shape_model.pth")

train_model()

# ============================
# 4.ONNX変換
# ============================

def export_onnx(model_path="shape_model.pth", onnx_path="shape_model.onnx"):
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 128, 128)

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
    export_onnx()



