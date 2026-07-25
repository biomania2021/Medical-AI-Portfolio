import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from model import UNet

def visualize_multiple_samples(model_path, data_dir, output_image_path="multi_slice_results.png", targets=None):
    """
    targets: 検証したい (患者インデックス, スライス番号, ラベル名) のリスト
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用中のデバイス: {device}")

    # 1. モデルのセットアップ
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))

    # デフォルトのターゲット設定 (症例やスライス位置を変えて比較)
    if targets is None:
        targets = [
            (0, 90, "Patient 001 - Large Tumor"),
            (0, 65, "Patient 001 - Small Tumor Edge"),
            (1, 85, "Patient 002 - Medium Tumor"),
            (2, 95, "Patient 003 - Another Case"),
        ]

    num_samples = len(targets)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for i, (p_idx, slice_idx, title_label) in enumerate(targets):
        test_patient = patient_dirs[p_idx]
        p_id = os.path.basename(test_patient)

        flair_path = os.path.join(test_patient, f"{p_id}_flair.nii")
        seg_path = os.path.join(test_patient, f"{p_id}_seg.nii")

        # NIfTI の読み込み
        flair_data = nib.load(flair_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()

        img_slice = flair_data[:, :, slice_idx]
        mask_slice = seg_data[:, :, slice_idx]

        # 前処理
        norm_img = img_slice / img_slice.max() if img_slice.max() > 0 else img_slice
        gt_mask = (mask_slice > 0).astype(np.float32)

        # 推論
        input_tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_prob = torch.sigmoid(output).cpu().squeeze().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.float32)

        # --- 描画処理 ---
        # 行が1つだけの場合のレイアウト調整
        ax_row = axes[i] if num_samples > 1 else axes

        # ① 入力画像
        ax_row[0].imshow(img_slice.T, cmap="gray", origin="lower")
        ax_row[0].set_title(f"{title_label}\nInput (FLAIR)")
        ax_row[0].axis("off")

        # ② 正解マスク
        ax_row[1].imshow(img_slice.T, cmap="gray", origin="lower")
        ax_row[1].imshow(gt_mask.T, cmap="Reds", alpha=0.5, origin="lower")
        ax_row[1].set_title("Ground Truth")
        ax_row[1].axis("off")

        # ③ 予測結果
        ax_row[2].imshow(img_slice.T, cmap="gray", origin="lower")
        ax_row[2].imshow(pred_mask.T, cmap="jet", alpha=0.5, origin="lower")
        ax_row[2].set_title("Model Prediction")
        ax_row[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"🖼️ 複数症例の比較可視化結果を '{output_image_path}' に保存しました！")

if __name__ == "__main__":
    BRATS_DIR = "/home/akiram/Documents/AI_Workspace/datasets/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    MODEL_PATH = "unet_brats2020.pth"

    # 検証したい患者のインデックス(0, 1, 2...)とスライス位置を指定
    custom_targets = [
        (0, 95, "Case 001 (Slice 95: Large)"),
        (0, 60, "Case 001 (Slice 60: Small)"),
        (1, 85, "Case 002 (Slice 85)"),
        (2, 90, "Case 003 (Slice 90)"),
    ]

    visualize_multiple_samples(
        model_path=MODEL_PATH,
        data_dir=BRATS_DIR,
        output_image_path="multi_slice_results.png",
        targets=custom_targets
    )