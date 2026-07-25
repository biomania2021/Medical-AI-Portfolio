# 🧠 Brain Tumor Segmentation using 2D U-Net (BraTS2020)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://www.python.org/)

本プロジェクトは、医療画像解析コンペティション **BraTS2020 (Brain Tumor Segmentation Challenge 2020)** の 3D 脳 MRI データ（.nii）から 2D スライスを切り出し、**U-Net** アーキテクチャを用いて脳腫瘍領域（Whole Tumor）をセグメンテーションするエンドツーエンドのディープラーニング・パイプラインです。

---

## 📸 識別・推論結果 (Qualitative Results)

未知の患者や、様々な形状・大きさの腫瘍領域に対するモデルの推論結果です。  
正解領域（Ground Truth: 赤）とモデル予測領域（Model Prediction: 赤/オーバーレイ）の精度の高い一致が確認できます。

![Multi Slice Results](multi_slice_results.jpg)

---

## 📈 定量評価指標 (Quantitative Evaluation)

データセット全体（29,440枚の2Dスライス）に対する評価結果です。

| 評価指標 (Metrics) | スコア | 備考 |
| :--- | :--- | :--- |
| **平均 Dice 係数 (全体)** | **80.11%** | 正常部位を含むデータセット全体 |
| **平均 Dice 係数 (腫瘍有りスライスのみ)** | **75.20%** | 純粋な病変検出性能 |
| **平均 IoU スコア (Jaccard Index)** | **74.47%** | 重なる領域の比率評価 |

---

## 🚀 使い方 (Quick Start)

### 1. リポジトリのクローンと依存パッケージの導入
` ` `bash
git clone [https://github.com/your-username/brain-segmentation.git](https://github.com/your-username/brain-segmentation.git)
cd brain-segmentation
pip install torch torchvision nibabel numpy matplotlib tqdm
` ` `

### 2. データセットの設定
BraTS2020 データセットをダウンロードし、ローカル環境に配置します。  
`train.py`, `evaluate.py`, `visualize.py` 内の `BRATS_DIR` をご自身の環境のデータセットパスに変更してください。

` ` `python
# 各スクリプト内のパス指定例
BRATS_DIR = "/path/to/your/MICCAI_BraTS2020_TrainingData"
` ` `

### 3. 学習の実行
` ` `bash
python train.py
` ` `

### 4. 定量評価の実行
` ` `bash
python evaluate.py
` ` `

### 5. 推論結果の可視化
` ` `bash
python visualize.py
` ` `

---

## 🛠️ プロジェクト構造

` ` `text
brain-segmentation/
├── datasets.py        # NIfTI (.nii) 3Dデータから2Dスライスへの抽出・正規化Dataset
├── model.py           # U-Net アーキテクチャの実装 (PyTorch)
├── train.py           # 学習・逆伝播処理およびモデル重みの保存
├── evaluate.py        # Dice 係数 / IoU の定量評価用スクリプト
├── visualize.py       # 推論結果と正解データの比較可視化スクリプト
├── unet_brats2020.pth # 学習済みモデルのパラメータ (重み)
└── README.md          # プロジェクト概要ドキュメント
` ` `