# YOLO11人検知モデル評価ツール

YOLOモデルの精度比較、コンフュージョンマトリクス出力、各種精度指標の計算を行うPythonスクリプトです。

## 必要なライブラリ

```bash
pip install ultralytics opencv-python matplotlib seaborn scikit-learn pandas numpy
```

## ファイル構成

- `yolo_evaluation.py` - メインの評価スクリプト
- `create_evaluation_dataset.py` - データセット作成ヘルパー
- `YOLO_EVALUATION_README.md` - このファイル

## 使用方法

### 1. サンプルデータセットでテスト実行

```bash
# サンプルデータセット作成
python create_evaluation_dataset.py --mode sample --output sample_dataset.json

# 単一モデル評価
python yolo_evaluation.py --models yolo11n.pt --dataset sample_dataset.json

# 複数モデル比較（例：旧モデルと新モデル）
python yolo_evaluation.py --models yolo11n.pt yolo11s.pt --dataset sample_dataset.json
```

### 2. 実際のデータセットで評価

#### YOLO形式のデータセットから変換
```bash
# YOLO形式データセット（images/, labels/）を評価用JSONに変換
python create_evaluation_dataset.py --mode yolo \
    --images-dir /path/to/images \
    --labels-dir /path/to/labels \
    --output my_dataset.json

# 評価実行
python yolo_evaluation.py --models yolo11n.pt old_model.pt \
    --dataset my_dataset.json \
    --conf-threshold 0.5 \
    --iou-threshold 0.5 \
    --output-dir evaluation_results
```

#### 手動でデータセットJSON作成
```json
{
  "images": [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg",
    "/path/to/image3.jpg"
  ],
  "annotations": [
    [
      {"bbox": [100, 100, 200, 300], "class": 0},
      {"bbox": [300, 150, 400, 350], "class": 0}
    ],
    [
      {"bbox": [150, 80, 250, 280], "class": 0}
    ],
    []
  ]
}
```

## 出力結果

### 1. コンソール出力例
```
==================================================
Evaluating: yolo11n.pt
==================================================
Progress: 1/100
Progress: 11/100
...

Results for yolo11n:
Precision: 0.847
Recall: 0.892
F1-Score: 0.869
AP: 0.834
TP: 156, FP: 28, FN: 19

==================================================
MODEL COMPARISON
==================================================
    Model  Precision    Recall  F1-Score        AP   TP  FP  FN  Total Images  Total Persons
   yolo11n      0.847     0.892     0.869     0.834  156  28  19           100            175
   yolo11s      0.901     0.863     0.881     0.856  151  17  24           100            175
```

### 2. 生成されるファイル

- `confusion_matrix_[model_name].png` - 各モデルのコンフュージョンマトリクス
- `precision_recall_curves.png` - PR曲線比較
- `model_comparison_YYYYMMDD_HHMMSS.csv` - 比較結果CSV
- `detailed_results_YYYYMMDD_HHMMSS.json` - 詳細結果JSON

### 3. 精度指標の説明

- **Precision（適合率）**: TP / (TP + FP) - 予測した人のうち、実際に人だった割合
- **Recall（再現率）**: TP / (TP + FN) - 実際の人のうち、正しく検出できた割合  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - PrecisionとRecallの調和平均
- **AP（Average Precision）**: Precision-Recall曲線下の面積

### 4. コンフュージョンマトリクス

```
                 Predicted
                 No Person  Person
Actual No Person    TN        FP
Actual Person       FN        TP
```

## パラメータ説明

- `--models`: 評価するモデルファイルパス（複数指定可能）
- `--dataset`: データセット設定JSONファイル
- `--conf-threshold`: 信頼度閾値（デフォルト: 0.5）
- `--iou-threshold`: IoU閾値（デフォルト: 0.5）
- `--output-dir`: 結果出力ディレクトリ

## 実用例

### モデル改善前後の比較
```bash
python yolo_evaluation.py \
    --models models/yolo11_before.pt models/yolo11_after.pt \
    --dataset validation_dataset.json \
    --conf-threshold 0.3 \
    --output-dir comparison_results
```

### 信頼度閾値の最適化
```bash
# 複数の信頼度で評価
for conf in 0.3 0.4 0.5 0.6 0.7; do
    python yolo_evaluation.py \
        --models yolo11n.pt \
        --dataset test_dataset.json \
        --conf-threshold $conf \
        --output-dir results_conf_$conf
done
```

## 注意事項

- 人検知（クラス0）のみを評価対象とします
- バウンディングボックスの形式は [x1, y1, x2, y2] です
- IoUマッチングは最良マッチング方式を採用しています
- メモリ使用量を考慮し、大量画像の場合は分割実行を推奨します

## トラブルシューティング

### エラー: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### エラー: "Could not load image"
- 画像ファイルパスが正しいか確認
- 画像ファイルが破損していないか確認

### メモリ不足
- データセットを小分けにして実行
- 画像解像度を下げて実行