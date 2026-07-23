import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 自分たちで作った model.py と datasets.py から必要な部品をインポート！
from model import UNet
from datasets import BrainSegmentationDataset

def dice_coefficient(predicted, target, thresh=0.5):
    """
    セグメンテーションの評価基準：Dice係数を計算する（実務・面接の必須知識）
    1.0に近いほど、正解の輪郭とAIの予測の輪郭がピタッと重なっていることを意味します。
    """
    smooth = 1e-5
    # 確率に変換してしきい値で0か1のマスクにする
    preds = (torch.sigmoid(predicted) > thresh).float()
    
    # 重なっている部分（積集合）の面積
    intersection = (preds * target).sum()
    
    # 2 * (重なり) / (予測の面積 + 正解の面積)
    return (2.0 * intersection + smooth) / (preds.sum() + target.sum() + smooth)

def train_unet(data_dir=".dummy_brain_data", epochs=10, batch_size=2, lr=0.0001):
    # ① 計算デバイスの自動判定（GPUがあれば優先、なければCPUで安全稼働）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用中のデバイス: {device}")

    # ② 脳用データローダーの準備
    dataset = BrainSegmentationDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"総サンプル数: {len(dataset)} (バッチサイズ: {batch_size}, ステップ数/Epoch: {len(dataloader)})")

    # ③ モデル、損失関数、最適化アルゴリズムの定義
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    # 🌟分類のCrossEntropyではなく、ピクセル単位の分類用Lossを採用！
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🚀 脳腫瘍セグメンテーション（U-Net）の学習を開始します...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            # 勾配のリセット
            optimizer.zero_grad()
            
            # 順伝播（U-Netに脳画像を流し込む）
            outputs = model(images)
            loss = criterion(outputs, masks) # 出力マスクと正解マスクのズレを計算
            
            # 逆伝播と最適化（左半分のエンコーダーまで過去に遡って重みを修正！）
            loss.backward()
            optimizer.step()

            # 統計情報の計算
            running_loss += loss.item() * images.size(0) # バッチサイズ分の損失を加算 image.size(0)はバッチ内のサンプル数
            # imagesは（Batch_Size, Channel, Height, Width）なので、images.size(0)(images.shape[0])はBatch_Sizeつまりはバッチに入っている画像の枚数を返す
            running_dice += dice_coefficient(outputs, masks).item() * images.size(0) # .item()でテンソルからPythonの数値に変換してGPUメモリを消費しない
            total_samples += images.size(0)
            
            # 進捗バーに現在のLossを表示
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total_samples # 1エポック全体の「画像1枚あたりの平均Loss」を算出（累計Loss ÷ 全画像数）
        epoch_dice = (running_dice / total_samples) * 100 # Dice係数をパーセンテージ表示に変換 Dice制度の合計値を全画面枚数で割り、一枚あたりの平均Dice係数(0.0から1.01)を算出している
        print(f"✨ Epoch [{epoch+1}/{epochs}] 終了 - Loss: {epoch_loss:.4f}, Dice Accuracy: {epoch_dice:.2f}%")
        
    print("🎉 脳用 U-Net の全学習プロセスが正常に完了しました！")

# ==========================================
# 実行ブロック
# ==========================================
if __name__ == "__main__":
    # 2エポックだけ回して、全体のバケツリレーがエラーなく繋がっているかテスト
    train_unet(data_dir=".dummy_brain_data", epochs=2, batch_size=2, lr=0.0001)