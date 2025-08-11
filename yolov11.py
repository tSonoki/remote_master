from ultralytics import YOLO

# 1. YOLOv11モデルの読み込み
# 'yolo11n.pt' (nano), 'yolo11s.pt' (small), 'yolo11m.pt' (medium) などから選択できます。
# nanoモデルが最も軽量で高速ですが、精度は少し劣ります。
model = YOLO('yolo11n.pt') 

# 2. ONNX形式でモデルをエクスポート
# imgsz: 入力画像のサイズ（YOLOv8と同じ640で問題ない場合が多いです）
# dynamic=False: ONNXモデルの入力サイズを固定します。Webブラウザでの実行にはこちらが推奨されることが多いです。
model.export(format='onnx', imgsz=640, dynamic=False)

print("ONNXモデルが 'yolo11n.onnx' という名前で保存されました。")