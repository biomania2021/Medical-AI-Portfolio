import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from datasets import BraTSDataset2D

def dice_coefficient(predicted, target, thresh=0.5):
    smooth = 1e-5
    preds = (torch.sigmoid(predicted) > thresh).float()
    intersection = (preds * target).sum()
    return (2.0 * intersection + smooth) / (preds.sum() + target.sum() + smooth)

def train_unet(data_dir, epochs=1, batch_size=8, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用中のデバイス: {device}")

    # BraTS用 Dataset & DataLoader の構築
    dataset = BraTSDataset2D(data_dir=data_dir, slice_range=(50, 130))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    print(f"総サンプル（スライス）数: {len(dataset)} (バッチサイズ: {batch_size}, ステップ数/Epoch: {len(dataloader)})")

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🚀 本番データ（BraTS2020）による学習を開始します...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_dice += dice_coefficient(outputs, masks).item() * images.size(0)
            total_samples += images.size(0)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / total_samples
        epoch_dice = (running_dice / total_samples) * 100
        print(
            f"✨ Epoch [{epoch+1}/{epochs}] 終了 - Loss: {epoch_loss:.4f},"
            f" Dice Accuracy: {epoch_dice:.2f}%"
        )

    # 💡 モデルの重み（パラメータ）を保存するコードを追加！
    torch.save(model.state_dict(), "unet_brats2020.pth")
    print("💾 モデルの重みを 'unet_brats2020.pth' に保存しました。")

    print("🎉 本番データでの学習テストが正常に完了しました！")

if __name__ == "__main__":
    # 実データの親ディレクトリパスを指定
    BRATS_DIR = "/home/akiram/Documents/AI_Workspace/datasets/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    
    train_unet(data_dir=BRATS_DIR, epochs=1, batch_size=8, lr=0.0001)