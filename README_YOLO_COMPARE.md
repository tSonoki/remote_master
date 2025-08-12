# YOLO11人検知精度比較ツール

## 🚀 使い方（1コマンド）

```bash
python yolo_compare.py
```

## 📊 出力される結果

### コンソール出力例
```
YOLO11 HUMAN DETECTION COMPARISON RESULTS
============================================================
Dataset: 3 images
Confidence: 0.3
------------------------------------------------------------
Model           Precision  Recall   F1-Score TP   FP   FN  
------------------------------------------------------------
yolo11n         0.667      0.667    0.667    2    1    1   
yolo11s         0.750      0.750    0.750    3    1    1   
------------------------------------------------------------

Best performing model:
  yolo11s: F1-Score = 0.750
```

### 生成ファイル
- `confusion_matrix_yolo11n.png` - コンフュージョンマトリクス画像
- `test_dataset.json` - テストデータセット設定
- `test_images/` - テスト画像フォルダ

## 🔧 動作内容

1. **自動セットアップ**: YOLOモデルが無い場合は自動ダウンロード
2. **テスト画像作成**: 人が写ったテスト画像を自動生成
3. **精度評価**: Precision、Recall、F1-Scoreを計算
4. **コンフュージョンマトリクス**: 視覚的な性能比較グラフを生成

## 📋 必要な環境

```bash
pip install ultralytics matplotlib seaborn numpy opencv-python
```

## 🎯 カスタマイズ

### 信頼度閾値を変更
```python
# yolo_compare.py の main() 関数内
confidence = 0.5  # デフォルト: 0.3
```

### 独自データセットで評価
```python
# test_dataset.json を以下の形式で作成
{
  "images": ["your_image1.jpg", "your_image2.jpg"],
  "annotations": [
    [{"bbox": [x1, y1, x2, y2], "class": 0}],
    []
  ]
}
```

## ❓ トラブルシューティング

**エラー: "No module named 'ultralytics'"**
```bash
pip install ultralytics
```

**GPU使用を無効化（CPU強制）**
```python
# yolo_compare.py の最初に追加
import torch
torch.cuda.is_available = lambda: False
```

**メモリ不足の場合**
- 信頼度閾値を上げる（0.5以上）
- 画像解像度を下げる

## 📈 精度指標の意味

- **Precision（適合率）**: 予測した人のうち正解の割合
- **Recall（再現率）**: 実際の人のうち検出できた割合  
- **F1-Score**: PrecisionとRecallの調和平均（総合指標）