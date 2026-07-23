import os
import glob
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, Spacingd, EnsureChannelFirstd
from utils import extract_patch

# ==========================================
# 1. 設定
# ==========================================
BASE_PATH = ".data"
SAVE_DIR_POS = os.path.join(BASE_PATH, "processed_fixed", "positives")
SAVE_DIR_NEG = os.path.join(BASE_PATH, "processed_fixed", "negatives")
os.makedirs(SAVE_DIR_POS, exist_ok=True)
os.makedirs(SAVE_DIR_NEG, exist_ok=True)

CROP_SIZE = 64 

# 【CT画像のハウンズフィールド単位(HU)の正規化】
# CT画像は水が0、空気が-1000という物理的な基準（HU値）を持っています。
# ここでは、肺の解析に不要な「空気(-1000以下)」と「硬い骨(400以上)」をクリッピング(切り捨て)し、
# AIが計算しやすい 0.0 〜 1.0 の範囲にスケーリングしています。

TARGET_SPACING = (0.8, 0.8, 0.8)

# ==========================================
# 2. SimpleITK 
# ==========================================
# annotations.csv と candidates.csv の読み込み
df_annotations = pd.read_csv(os.path.join(BASE_PATH, "annotations.csv"))
df_candidates = pd.read_csv(os.path.join(BASE_PATH, "candidates.csv"))
# candidates の中から class=0 (陰性) だけを抽出
df_negatives = df_candidates[df_candidates['class'] == 0]

# mhdファイルのリスト取得
mhd_files = glob.glob(os.path.join(BASE_PATH, "*.mhd"))
mhd_dict = {os.path.basename(f).replace(".mhd", ""): f for f in mhd_files}

print(f"見つかったCT画像: {len(mhd_dict)}件")

# ==========================================
# 3. 切り出しループ
# ==========================================
d = CROP_SIZE // 2

for uid, mhd_path in tqdm(mhd_dict.items(), desc="Processing CT scans"):
    
    # 1. SimpleITK で画像を読み込む
    itk_img = sitk.ReadImage(mhd_path)
    
    # SimpleITK は (X, Y, Z) の順番でピクセルを扱う
    # Numpy に変換すると (Z, Y, X) にひっくり返る
    img_np = sitk.GetArrayFromImage(itk_img) # 形状は (Z, Y, X)
    
    # --------------------------------
    # 【陽性】の切り出し (annotations)
    # --------------------------------
    pos_rows = df_annotations[df_annotations['seriesuid'] == uid]
    for i, row in pos_rows.iterrows():
        # ワールド座標（ミリ）
        world_coords = (row['coordX'], row['coordY'], row['coordZ'])
        voxel_coords = itk_img.TransformPhysicalPointToIndex(world_coords)
        
        v_x, v_y, v_z = voxel_coords
        
        # はみ出しチェック
        if (v_z - d >= 0 and v_z + d < img_np.shape[0] and
            v_y - d >= 0 and v_y + d < img_np.shape[1] and
            v_x - d >= 0 and v_x + d < img_np.shape[2]):
            
            # (Z, Y, X) の順番で切り出し
            patch = img_np[v_z-d:v_z+d, v_y-d:v_y+d, v_x-d:v_x+d]
            
            # まだ 0.8mm にリサイズしていないので、SimpleITK の画像オブジェクトに戻してリサイズ
            # （※本当はここでリサイズしますが、まずは正しく真ん中が切り出せているか確認するため、
            # 今回はリサイズ処理を一旦省いてそのまま正規化して保存します）
            
            patch_norm = extract_patch(patch)
            
            save_path = os.path.join(SAVE_DIR_POS, f"{uid}_pos_{i}.npy")
            np.save(save_path, patch_norm)

    # --------------------------------
    # 【陰性】の切り出し (candidates) 
    # --------------------------------
    neg_rows = df_negatives[df_negatives['seriesuid'] == uid]
    # 例：1つのCTから最大3個までにする（バランスをとるため）
    neg_rows = neg_rows.head(3) 
    
    for i, row in neg_rows.iterrows():
        world_coords = (row['coordX'], row['coordY'], row['coordZ'])
        voxel_coords = itk_img.TransformPhysicalPointToIndex(world_coords)
        v_x, v_y, v_z = voxel_coords
        
        if (v_z - d >= 0 and v_z + d < img_np.shape[0] and
            v_y - d >= 0 and v_y + d < img_np.shape[1] and
            v_x - d >= 0 and v_x + d < img_np.shape[2]):
            
            patch = img_np[v_z-d:v_z+d, v_y-d:v_y+d, v_x-d:v_x+d]
            patch_norm = extract_patch(patch)
            
            save_path = os.path.join(SAVE_DIR_NEG, f"{uid}_neg_{i}.npy")
            np.save(save_path, patch_norm)

print("✨ 完璧な座標での切り出しが完了しました！")