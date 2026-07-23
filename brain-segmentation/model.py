import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm or GroupNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # 医療画像ではバッチサイズが小さくなりがちなため、ここでもGroupNormを採用して安定化
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """脳腫瘍・病変抽出用 2D U-Net"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # --- エンコーダー（縮める側） ---
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- デコーダー（戻す側 ＆ スキップ接続） ---
        # 拡大（Up）しながら、エンコーダー側の特徴量と結合（Concat）するため、入力チャネルが2倍になります
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256) # 256 (Up) + 256 (Skip) = 512

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128) # 128 (Up) + 128 (Skip) = 256

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)   # 64 (Up) + 64 (Skip) = 128

        # 最終出力層（ピクセルごとに「病変か背景か」を1チャネルで出力）
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # エンコーダーの各階層での出力を、スキップ接続用に記憶（バケツリレーの準備）
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # デコーダー側：拡大して、同じ高さのエンコーダーの記憶（x3, x2, x1）をドッキング（torch.cat）
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        return logits

# ==========================================
# 実行ブロック（モデルの形状テスト）
# ==========================================
if __name__ == "__main__":
    print("脳用 U-Net モデルのテンソルサイズテストを開始します...")
    # 2枚のバッチ、1チャネル（白黒の脳MRI/CT）、サイズ 256x256 のダミーデータ
    dummy_brain = torch.randn(2, 1, 256, 256)
    
    model = UNet(in_channels=1, out_channels=1)
    output = model(dummy_brain)
    
    print(f"入力画像サイズ: {dummy_brain.shape} -> (Batch, Channel, H, W)")
    print(f"出力画像サイズ: {output.shape} -> 入力と同じサイズ (2, 1, 256, 256) であればセグメンテーション成功！")