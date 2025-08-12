#!/usr/bin/env python3
"""
YOLO11人検知精度比較ツール
使い方: python yolo_compare.py
"""

import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class YOLOCompare:
    def __init__(self):
        self.results = {}
    
    def create_test_data(self):
        """テスト画像とデータセットを作成"""
        print("Creating test images...")
        
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        images = []
        annotations = []
        
        for i in range(3):
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            img[:] = (64, 64, 64)
            
            image_path = test_dir / f"test_{i}.jpg"
            
            if i < 2:  # 人を描画
                num_persons = 1 if i == 0 else 2
                person_annotations = []
                
                for j in range(num_persons):
                    x, y, w, h = 100 + j * 250, 150, 80, 200
                    
                    # 人の描画
                    cv2.circle(img, (x + w//2, y + 25), 25, (200, 180, 160), -1)
                    cv2.rectangle(img, (x + 10, y + 50), (x + w - 10, y + h - 50), (100, 100, 200), -1)
                    cv2.rectangle(img, (x + 15, y + h - 50), (x + 35, y + h), (80, 80, 120), -1)
                    cv2.rectangle(img, (x + w - 35, y + h - 50), (x + w - 15, y + h), (80, 80, 120), -1)
                    
                    person_annotations.append({
                        "bbox": [x, y, x + w, y + h],
                        "class": 0
                    })
                
                annotations.append(person_annotations)
            else:
                annotations.append([])
            
            cv2.imwrite(str(image_path), img)
            images.append(str(image_path))
            print(f"  Created {image_path.name} with {len(annotations[-1])} person(s)")
        
        # データセット保存
        dataset = {"images": images, "annotations": annotations}
        with open("test_dataset.json", 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print("Test dataset created: test_dataset.json")
        return dataset
    
    def calculate_iou(self, box1, box2):
        """IoU計算"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_model(self, model_path, dataset, confidence=0.5):
        """単一モデルの評価"""
        print(f"\nEvaluating {model_path}...")
        
        model = YOLO(model_path)
        tp, fp, fn = 0, 0, 0
        
        for i, (image_path, gt_annotations) in enumerate(zip(dataset['images'], dataset['annotations'])):
            print(f"  Processing {Path(image_path).name}...")
            
            # 推論
            results = model(image_path, conf=confidence, verbose=False)
            
            # 人検出結果を抽出
            predictions = []
            for r in results:
                if r.boxes is not None:
                    for j in range(len(r.boxes)):
                        if int(r.boxes.cls[j]) == 0:  # 人のみ
                            bbox = r.boxes.xyxy[j].cpu().numpy()
                            conf = float(r.boxes.conf[j])
                            predictions.append({'bbox': bbox.tolist(), 'confidence': conf})
            
            # 正解データ（人のみ）
            gt_persons = [gt for gt in gt_annotations if gt['class'] == 0]
            
            # マッチング
            matched_pred = [False] * len(predictions)
            matched_gt = [False] * len(gt_persons)
            
            for pred_idx, pred in enumerate(predictions):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_persons):
                    if matched_gt[gt_idx]:
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou > 0.5:
                    matched_pred[pred_idx] = True
                    matched_gt[best_gt_idx] = True
            
            # カウント
            tp += sum(matched_pred)
            fp += len(predictions) - sum(matched_pred)
            fn += len(gt_persons) - sum(matched_gt)
        
        # 精度計算
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'model': Path(model_path).stem,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1_score': f1_score
        }
    
    def create_confusion_matrix(self, results):
        """コンフュージョンマトリクス作成"""
        for result in results:
            tp, fp, fn = result['tp'], result['fp'], result['fn']
            tn = max(0, 3 - tp - fp - fn)  # 3画像から推定
            
            cm = np.array([[tn, fp], [fn, tp]])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Person', 'Person'],
                       yticklabels=['No Person', 'Person'])
            plt.title(f'Confusion Matrix - {result["model"]}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            filename = f"confusion_matrix_{result['model']}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
    
    def compare_models(self, model_paths, confidence=0.5):
        """複数モデル比較"""
        # テストデータ作成
        if not Path("test_dataset.json").exists():
            dataset = self.create_test_data()
        else:
            with open("test_dataset.json", 'r') as f:
                dataset = json.load(f)
        
        # 各モデルを評価
        results = []
        for model_path in model_paths:
            if Path(model_path).exists():
                result = self.evaluate_model(model_path, dataset, confidence)
                results.append(result)
            else:
                print(f"Model not found: {model_path}")
        
        if not results:
            print("No models to evaluate!")
            return
        
        # 結果表示
        print("\n" + "="*60)
        print("YOLO11 HUMAN DETECTION COMPARISON RESULTS")
        print("="*60)
        print(f"Dataset: {len(dataset['images'])} images")
        print(f"Confidence: {confidence}")
        print("-" * 60)
        print(f"{'Model':<15} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['model']:<15} {result['precision']:<10.3f} {result['recall']:<8.3f} "
                  f"{result['f1_score']:<8.3f} {result['tp']:<4} {result['fp']:<4} {result['fn']:<4}")
        
        # コンフュージョンマトリクス作成
        print("\nCreating confusion matrices...")
        self.create_confusion_matrix(results)
        
        print(f"\nEvaluation completed! Check confusion_matrix_*.png files.")
        return results

def main():
    """メイン関数"""
    print("YOLO11 Human Detection Comparison Tool")
    print("="*50)
    
    comparator = YOLOCompare()
    
    # 利用可能なモデルを確認
    available_models = []
    candidate_models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
    
    for model in candidate_models:
        if Path(model).exists():
            available_models.append(model)
    
    if not available_models:
        print("No YOLO models found. Downloading yolo11n.pt...")
        try:
            model = YOLO('yolo11n.pt')  # 自動ダウンロード
            available_models.append('yolo11n.pt')
        except Exception as e:
            print(f"Failed to download model: {e}")
            return
    
    print(f"Found models: {', '.join(available_models)}")
    
    # 比較実行
    confidence = 0.3  # 低めの閾値で検出しやすく
    results = comparator.compare_models(available_models, confidence)
    
    if len(results) > 1:
        print("\nBest performing model:")
        best = max(results, key=lambda x: x['f1_score'])
        print(f"  {best['model']}: F1-Score = {best['f1_score']:.3f}")

if __name__ == "__main__":
    main()