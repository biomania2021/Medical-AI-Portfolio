import torch
import torch.nn as nn
import torch.nn.functional as F

class Medical3DCNN(nn.Module):
    """
    肺結節検出用 3D CNNモデル
    小さな結節の特徴を逃さないよう、Global Max Pooling (GMP) を採用。
    3D画像でバッチサイズが小さくなることを考慮し、BatchNormではなくGroupNormを使用。
    """
    def __init__(self):
        super(Medical3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool3d(2) 

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool3d(2) 

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(8, 128)
        self.pool3 = nn.MaxPool3d(2) 

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(16, 256)
        self.pool4 = nn.MaxPool3d(2) 
        
        # GAP(Global Average Pooling)ではなく、GMP(Global Max Pooling)を採用
        # 理由: 背景(健康な肺)の中にぽつんと存在する「がんのシグナル」を薄めずに全結合層へ伝えるため
        self.gmp = nn.AdaptiveMaxPool3d(1)
        
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.2) 
        self.fc2 = nn.Linear(64, 2) # 2クラス分類（陽性/陰性）
        
    def forward(self, x):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))
        x = self.pool4(F.relu(self.gn4(self.conv4(x))))
        
        x = self.gmp(x) # 平均ではなく「一番怪しいところ」をダイレクトに抽出
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 実行ブロック（モデルの形状テスト）
# ==========================================
if __name__ == "__main__":
    # バッチサイズ2、1チャネル、64x64x64のダミーデータ(偽物のCT画像)を作成
    print("モデルのコンパイルとテンソルサイズのテストを開始します...")
    dummy_input = torch.randn(2, 1, 64, 64, 64)
    model = Medical3DCNN()
    
    # 実際にダミーデータをモデルに通してみる
    output = model(dummy_input)
    
    print(f"入力テンソル: {dummy_input.shape}")
    print(f"出力テンソル: {output.shape} -> (バッチサイズ, クラス数) になっていれば成功！")