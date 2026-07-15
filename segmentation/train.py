import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# 先ほど作成した model.py から Medical3DCNN をインポート！
from segmentation.model import Medical3DCNN

# ==========================================
# 1. 医療画像用のカスタムDatasetクラス
# ==========================================
class LUNA16PatchDataset(Dataset):
    """
    datasets.py で作成した .npy パッチデータを PyTorch で読み込むためのクラス。
   
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 陽性と陰性のファイルをすべてリストアップ
        self.pos_dir = os.path.join(data_dir, "processed_fixed", "positives")
        self.neg_dir = os.path.join(data_dir, "processed_fixed", "negatives")
        
        self.pos_files = [os.path.join(self.pos_dir, f) for f in os.listdir(self.pos_dir) if f.endswith('.npy')] if os.path.exists(self.pos_dir) else []
        self.neg_files = [os.path.join(self.neg_dir, f) for f in os.listdir(self.neg_dir) if f.endswith('.npy')] if os.path.exists(self.neg_dir) else []
        
        # 全データと、対応するラベル（陽性=1, 陰性=0）を結合
        self.all_files = self.pos_files + self.neg_files
        self.labels = [1] * len(self.pos_files) + [0] * len(self.neg_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        # 3Dパッチデータのロード
        patch = np.load(self.all_files[idx])
        # PyTorchが扱えるように (Channel, Z, Y, X) の形状に変形
        patch = np.expand_dims(patch, axis=0) # (1, 64, 64, 64)
        
        tensor_data = torch.tensor(patch, dtype=torch.float32)
        tensor_label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return tensor_data, tensor_label

# ==========================================
# 2. メインの学習ループ
# ==========================================
def train_model(data_dir=".data", epochs=10, batch_size=4, lr=0.001):
    # ① 計算デバイスの自動判定（GPUがあれば使い、なければCPUで安全に動かすお作法）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用中のデバイス: {device}")

    # ② データローダーの準備
    dataset = LUNA16PatchDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"総サンプル数: {len(dataset)} (バッチサイズ: {batch_size}, ステップ数/Epoch: {len(dataloader)})")

    # ③ モデル、損失関数、最適化アルゴリズムの定義
    model = Medical3DCNN().to(device)
    criterion = nn.CrossEntropyLoss() # 2クラス分類（結節あり/なし）
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🚀 学習を開始します...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # tqdm で進捗バーを表示（実務での進捗監視に必須）
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            # 統計情報の計算
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 進捗バーに現在のLossを表示
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f"✨ Epoch [{epoch+1}/{epochs}] 終了 - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    print("🎉 全Epochの学習が完了しました！")

# ==========================================
# 3. 直接実行ブロック
# ==========================================
if __name__ == "__main__":
    # ここにダミーデータ環境での実行確認など、ハイパーパラメータをまとめて記述します
    train_model(data_dir=".data", epochs=2, batch_size=2, lr=0.001)