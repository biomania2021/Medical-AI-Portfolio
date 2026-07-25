import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

class BraTSDataset2D(Dataset):
    """
    BraTS2020 (.nii) データセットから2Dスライスを抽出するデータローダー
    """
    def __init__(self, data_dir, slice_range=(50, 130)):
        """
        data_dir: MICCAI_BraTS2020_TrainingData などの親フォルダパス
        slice_range: 脳がしっかり映っているスライス範囲 (例: 50〜130枚目)
        """
        self.data_dir = data_dir
        self.slice_range = slice_range
        self.samples = []

        # 患者フォルダ (BraTS20_Training_*) を全検索
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))

        # 各患者フォルダから FLAIR画像 と Seg(正解マスク) のパスをペアで登録
        for p_dir in patient_dirs:
            p_id = os.path.basename(p_dir)
            flair_path = os.path.join(p_dir, f"{p_id}_flair.nii")
            seg_path = os.path.join(p_dir, f"{p_id}_seg.nii")

            if os.path.exists(flair_path) and os.path.exists(seg_path):
                # 脳が映っている有効なスライス範囲を1枚ずつサンプルとして登録
                for s in range(slice_range[0], slice_range[1]):
                    self.samples.append((flair_path, seg_path, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flair_path, seg_path, slice_idx = self.samples[idx]

        # NIfTI データのロード
        flair_obj = nib.load(flair_path)
        seg_obj = nib.load(seg_path)

        # numpy 配列 (240, 240, 155) として取得
        flair_data = flair_obj.get_fdata()
        seg_data = seg_obj.get_fdata()

        # 指定したスライスの2D画像を抽出 -> (240, 240)
        img_slice = flair_data[:, :, slice_idx]
        mask_slice = seg_data[:, :, slice_idx]

        # 0〜1に正規化 (最大値で割る)
        if img_slice.max() > 0:
            img_slice = img_slice / img_slice.max()

        # マスクの二値化 (腫瘍領域があれば 1.0、無ければ 0.0)
        mask_slice = (mask_slice > 0).astype(np.float32)

        # PyTorch Tensor 形式に変換 (Channel, Height, Width) -> (1, 240, 240)
        img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)

        return img_tensor, mask_tensor