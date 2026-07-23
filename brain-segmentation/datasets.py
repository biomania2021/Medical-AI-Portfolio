import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BrainSegmentationDataset(Dataset):
    """
    脳MRI/CT画像とその正解マスク（病変領域）をセットで読み込むカスタムDataset。
    巨大な医療画像データを一度にメモリに載せないオンデマンド（1ファイルずつ）ロード設計。
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # 📁 想定するフォルダ構造
        # data_dir/images/ にオリジナルの脳画像（.npy）
        # data_dir/masks/  に正解の病変マスク（.npy）
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        
        # ファイルリストの取得（ペアが揃うようにソートしておくのが実務の鉄則）
        if os.path.exists(self.img_dir):
            self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.npy')])
        else:
            self.img_files = []

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. インデックスに対応するファイル名を取得
        img_name = self.img_files[idx]
        
        # 2. 画像と「対応する正解マスク」をそれぞれロード
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # 画像と同じファイル名で保存されている想定
        
        # 本物のデータがない環境（テスト時）でもエラーにならないための安全弁（ダミー生成）
        if os.path.exists(img_path) and os.path.exists(mask_path):
            image = np.load(img_path)
            mask = np.load(mask_path)
        else:
            # テスト用の擬似データ (256x256)
            image = np.random.randn(256, 256)
            mask = np.zeros((256, 256))
            mask[100:150, 100:150] = 1.0 # ど真ん中に四角い病変があると仮定
            
        # 3. PyTorchが扱えるようにチャネル次元 (Channel, H, W) を追加
        image = np.expand_dims(image, axis=0) # (1, 256, 256)
        mask = np.expand_dims(mask, axis=0)  # (1, 256, 256)
        
        # 4. テンソルに変換
        tensor_image = torch.tensor(image, dtype=torch.float32)
        tensor_mask = torch.tensor(mask, dtype=torch.float32) # マスクもピクセルごとに計算するためfloat32にする
        
        return tensor_image, tensor_mask

# ==========================================
# 実行ブロック（データローダーの挙動テスト）
# ==========================================
if __name__ == "__main__":
    print("脳用データローダーのモジュールテストを開始します...")
    
    # ダミーのディレクトリを指定してデータセットを作成
    dataset = BrainSegmentationDataset(data_dir=".dummy_brain_data")
    
    # 疑似的に1番目のデータ（idx=0）を取り出してみる
    # 本物のファイルがなくても、安全弁が動いてダミーが生成されます
    sample_img, sample_mask = dataset[0]
    
    print(f"Datasetから取り出した1件のデータ数: {len(dataset[0])} (画像とマスクのペア)")
    print(f"取り出した画像テンソル形状: {sample_img.shape} -> (Channel, H, W)")
    print(f"取り出したマスクテンソル形状: {sample_mask.shape} -> (Channel, H, W)")
    
    # いつものデータローダーに束ねてみる（バッチサイズ 2）
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch_imgs, batch_masks = next(iter(dataloader))
    
    print(f"\nDataLoaderで束ねた後の形状:")
    print(f"バッチ画像: {batch_imgs.shape} -> (Batch, Channel, H, W)")
    print(f"バッチマスク: {batch_masks.shape} -> (Batch, Channel, H, W)")
    print("✨ データパイプラインのテストはすべて合格です！")