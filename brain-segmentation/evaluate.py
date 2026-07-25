import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import UNet
from datasets import BraTSDataset2D

def calculate_metrics(pred_mask, gt_mask, smooth=1e-5):
    """
    1枚のスライスに対する Dice 係数 と IoU を計算する関数
    pred_mask: Binary numpy array (0 or 1)
    gt_mask: Binary numpy array (0 or 1)
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    # Dice Coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + gt_sum + smooth)
    
    # IoU (Intersection over Union)
    iou = (intersection + smooth) / (union + smooth)

    return dice, iou

def evaluate_model(model_path, data_dir, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用中のデバイス: {device}")

    # 1. モデルのロード
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ モデル '{model_path}' の読み込み完了")

    # 2. データセットの準備
    dataset = BraTSDataset2D(data_dir=data_dir, slice_range=(50, 130))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    dice_scores = []
    iou_scores = []
    tumor_present_dice = []  # 実際に腫瘍が存在するスライスのみのDiceスコア

    print("📊 データセット全体に対する定量評価を開始します...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            # Sigmoidを通して二値化 (閾値 0.5)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.uint8)
            gts = masks.numpy().astype(np.uint8)

            # バッチ内の各スライスごとにスコアを算出
            for pred, gt in zip(preds, gts):
                p_slice = pred[0]
                g_slice = gt[0]

                dice, iou = calculate_metrics(p_slice, g_slice)
                dice_scores.append(dice)
                iou_scores.append(iou)

                # 腫瘍が実際に存在するスライス（正解ピクセル数 > 0）のみ抽出して集計
                if g_slice.sum() > 0:
                    tumor_present_dice.append(dice)

    # 3. 統計値の算出
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    mean_tumor_dice = np.mean(tumor_present_dice)

    print("\n" + "="*40)
    print("📈 【評価レポート (Evaluation Metrics)】")
    print("="*40)
    print(f"・総スライス数: {len(dice_scores):,} 枚")
    print(f"・平均 Dice 係数 (全体): {mean_dice * 100:.2f}%")
    print(f"・平均 IoU スコア (全体):  {mean_iou * 100:.2f}%")
    print(f"・平均 Dice 係数 (腫瘍有りスライスのみ): {mean_tumor_dice * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    BRATS_DIR = "/home/akiram/Documents/AI_Workspace/datasets/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    MODEL_PATH = "unet_brats2020.pth"

    evaluate_model(model_path=MODEL_PATH, data_dir=BRATS_DIR)