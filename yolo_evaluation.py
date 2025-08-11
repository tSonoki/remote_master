#!/usr/bin/env python3
"""
YOLO11人検知モデル評価スクリプト
- 精度比較（Precision, Recall, F1-Score, mAP）
- コンフュージョンマトリクス出力
- 複数モデルの比較機能
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import pandas as pd
from ultralytics import YOLO
import argparse
from datetime import datetime

class YOLOEvaluator:
    def __init__(self, confidence_threshold: float = 0.5, iou_threshold: float = 0.5):
        """
        YOLO評価クラス
        
        Args:
            confidence_threshold: 信頼度閾値
            iou_threshold: IoU閾値（Non-Maximum Suppression用）
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.results = {}
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        2つのバウンディングボックスのIoU（Intersection over Union）を計算
        
        Args:
            box1, box2: [x1, y1, x2, y2] 形式のバウンディングボックス
            
        Returns:
            IoU値 (0-1)
        """
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
    
    def match_detections(self, predictions: List[Dict], ground_truths: List[Dict], 
                        iou_threshold: float = 0.5) -> Tuple[List[bool], List[bool]]:
        """
        予測結果と正解データをマッチングし、TP/FP/FNを判定
        
        Args:
            predictions: 予測結果リスト [{"bbox": [x1,y1,x2,y2], "confidence": float, "class": int}]
            ground_truths: 正解データリスト [{"bbox": [x1,y1,x2,y2], "class": int}]
            iou_threshold: マッチング用IoU閾値
            
        Returns:
            (pred_matches, gt_matches): 各予測・正解のマッチング状況
        """
        pred_matches = [False] * len(predictions)
        gt_matches = [False] * len(ground_truths)
        
        # 信頼度順にソート
        pred_indices = sorted(range(len(predictions)), 
                            key=lambda i: predictions[i]['confidence'], reverse=True)
        
        for pred_idx in pred_indices:
            pred = predictions[pred_idx]
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matches[gt_idx] or pred['class'] != gt['class']:
                    continue
                    
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                pred_matches[pred_idx] = True
                gt_matches[best_gt_idx] = True
        
        return pred_matches, gt_matches
    
    def evaluate_single_image(self, model_path: str, image_path: str, 
                            ground_truth: List[Dict]) -> Dict:
        """
        単一画像でのモデル評価
        
        Args:
            model_path: YOLOモデルファイルパス
            image_path: 評価画像パス
            ground_truth: 正解データ
            
        Returns:
            評価結果辞書
        """
        model = YOLO(model_path)
        
        # 推論実行
        results = model(image_path, conf=self.confidence_threshold, verbose=False)
        
        # 人（クラス0）の検出結果のみ抽出
        predictions = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    if int(boxes.cls[i]) == 0:  # 人のみ
                        bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        conf = float(boxes.conf[i])
                        predictions.append({
                            'bbox': bbox.tolist(),
                            'confidence': conf,
                            'class': 0
                        })
        
        # マッチング
        pred_matches, gt_matches = self.match_detections(predictions, ground_truth)
        
        tp = sum(pred_matches)
        fp = len(predictions) - tp
        fn = len(ground_truth) - sum(gt_matches)
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'predictions': predictions,
            'pred_matches': pred_matches,
            'gt_matches': gt_matches,
            'num_detections': len(predictions),
            'num_ground_truths': len(ground_truth)
        }
    
    def evaluate_dataset(self, model_path: str, dataset_config: Dict) -> Dict:
        """
        データセット全体での評価
        
        Args:
            model_path: YOLOモデルファイルパス
            dataset_config: データセット設定 {"images": [...], "annotations": [...]}
            
        Returns:
            評価結果
        """
        total_tp, total_fp, total_fn = 0, 0, 0
        all_predictions = []
        all_ground_truths = []
        image_results = []
        
        print(f"Evaluating model: {model_path}")
        print(f"Dataset size: {len(dataset_config['images'])} images")
        
        for i, (image_path, annotations) in enumerate(zip(dataset_config['images'], 
                                                         dataset_config['annotations'])):
            if i % 10 == 0:
                print(f"Progress: {i+1}/{len(dataset_config['images'])}")
                
            # 人のアノテーションのみ抽出
            person_annotations = [ann for ann in annotations if ann['class'] == 0]
            
            result = self.evaluate_single_image(model_path, image_path, person_annotations)
            
            total_tp += result['tp']
            total_fp += result['fp']
            total_fn += result['fn']
            
            # AP計算用のデータを保存
            for j, pred in enumerate(result['predictions']):
                all_predictions.append({
                    'confidence': pred['confidence'],
                    'correct': result['pred_matches'][j]
                })
            
            all_ground_truths.extend([1] * len(person_annotations))
            image_results.append(result)
        
        # 精度指標計算
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AP計算
        if len(all_predictions) > 0:
            confidences = [p['confidence'] for p in all_predictions]
            correct_flags = [p['correct'] for p in all_predictions]
            
            # 信頼度順にソート
            sorted_indices = np.argsort(confidences)[::-1]
            correct_sorted = np.array(correct_flags)[sorted_indices]
            
            # Precision-Recall曲線用のデータ
            tp_cumsum = np.cumsum(correct_sorted)
            fp_cumsum = np.cumsum(~correct_sorted)
            
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            recalls = tp_cumsum / len(all_ground_truths) if len(all_ground_truths) > 0 else 0
            
            # AP計算（11点補間）
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                p_interp = precisions[recalls >= t]
                p_max = np.max(p_interp) if len(p_interp) > 0 else 0
                ap += p_max / 11
        else:
            ap = 0
            precisions = np.array([])
            recalls = np.array([])
        
        return {
            'model_path': model_path,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ap': ap,
            'precisions': precisions,
            'recalls': recalls,
            'image_results': image_results,
            'num_images': len(dataset_config['images']),
            'num_total_persons': len(all_ground_truths)
        }
    
    def create_confusion_matrix(self, results: Dict) -> np.ndarray:
        """
        コンフュージョンマトリクス作成
        
        Args:
            results: evaluate_datasetの結果
            
        Returns:
            2x2のコンフュージョンマトリクス [[TN, FP], [FN, TP]]
        """
        tp = results['total_tp']
        fp = results['total_fp']
        fn = results['total_fn']
        
        # 人検知の場合、TNは定義が困難（背景領域の数が不明確）
        # 簡易的にTN=0として扱うか、画像数から推定
        tn = results['num_images'] - tp - fp - fn  # 簡易推定
        if tn < 0:
            tn = 0
            
        return np.array([[tn, fp], [fn, tp]])
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                            save_path: Optional[str] = None):
        """
        コンフュージョンマトリクスの可視化
        
        Args:
            cm: コンフュージョンマトリクス
            model_name: モデル名
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted: No Person', 'Predicted: Person'],
                   yticklabels=['Actual: No Person', 'Actual: Person'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        plt.show()
    
    def plot_precision_recall_curve(self, results: List[Dict], save_path: Optional[str] = None):
        """
        複数モデルのPrecision-Recall曲線を描画
        
        Args:
            results: 複数の評価結果リスト
            save_path: 保存パス
        """
        plt.figure(figsize=(10, 8))
        
        for result in results:
            model_name = Path(result['model_path']).stem
            precisions = result['precisions']
            recalls = result['recalls']
            ap = result['ap']
            
            if len(precisions) > 0 and len(recalls) > 0:
                plt.plot(recalls, precisions, 
                        label=f'{model_name} (AP={ap:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Person Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved: {save_path}")
        plt.show()
    
    def compare_models(self, results: List[Dict]) -> pd.DataFrame:
        """
        複数モデルの比較結果をデータフレームで出力
        
        Args:
            results: 複数の評価結果リスト
            
        Returns:
            比較結果DataFrame
        """
        comparison_data = []
        
        for result in results:
            model_name = Path(result['model_path']).stem
            comparison_data.append({
                'Model': model_name,
                'Precision': f"{result['precision']:.3f}",
                'Recall': f"{result['recall']:.3f}",
                'F1-Score': f"{result['f1_score']:.3f}",
                'AP': f"{result['ap']:.3f}",
                'TP': result['total_tp'],
                'FP': result['total_fp'],
                'FN': result['total_fn'],
                'Total Images': result['num_images'],
                'Total Persons': result['num_total_persons']
            })
        
        df = pd.DataFrame(comparison_data)
        return df

def create_sample_dataset() -> Dict:
    """
    サンプルデータセットの作成（テスト用）
    
    Returns:
        データセット設定
    """
    # 実際の使用時は、実際の画像パスとアノテーションデータに置き換える
    sample_images = [
        "sample_image_1.jpg",
        "sample_image_2.jpg", 
        "sample_image_3.jpg"
    ]
    
    sample_annotations = [
        # 画像1のアノテーション（人が2人）
        [
            {"bbox": [100, 100, 200, 300], "class": 0},  # 人1
            {"bbox": [300, 150, 400, 350], "class": 0}   # 人2
        ],
        # 画像2のアノテーション（人が1人）
        [
            {"bbox": [150, 80, 250, 280], "class": 0}    # 人1
        ],
        # 画像3のアノテーション（人なし）
        []
    ]
    
    return {
        "images": sample_images,
        "annotations": sample_annotations
    }

def main():
    parser = argparse.ArgumentParser(description='YOLO Model Evaluation')
    parser.add_argument('--models', nargs='+', required=True,
                       help='YOLO model paths to evaluate')
    parser.add_argument('--dataset', type=str, 
                       help='Dataset config JSON file path')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # データセット読み込み
    if args.dataset:
        with open(args.dataset, 'r') as f:
            dataset_config = json.load(f)
    else:
        print("No dataset provided, using sample dataset")
        dataset_config = create_sample_dataset()
    
    # 評価器初期化
    evaluator = YOLOEvaluator(args.conf_threshold, args.iou_threshold)
    
    # 各モデルを評価
    all_results = []
    
    for model_path in args.models:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*50}")
        
        result = evaluator.evaluate_dataset(model_path, dataset_config)
        all_results.append(result)
        
        # 個別結果出力
        print(f"\nResults for {Path(model_path).stem}:")
        print(f"Precision: {result['precision']:.3f}")
        print(f"Recall: {result['recall']:.3f}")
        print(f"F1-Score: {result['f1_score']:.3f}")
        print(f"AP: {result['ap']:.3f}")
        print(f"TP: {result['total_tp']}, FP: {result['total_fp']}, FN: {result['total_fn']}")
        
        # コンフュージョンマトリクス作成・可視化
        cm = evaluator.create_confusion_matrix(result)
        cm_path = output_dir / f"confusion_matrix_{Path(model_path).stem}.png"
        evaluator.plot_confusion_matrix(cm, Path(model_path).stem, str(cm_path))
    
    # 比較結果出力
    if len(all_results) > 1:
        print(f"\n{'='*50}")
        print("MODEL COMPARISON")
        print(f"{'='*50}")
        
        comparison_df = evaluator.compare_models(all_results)
        print(comparison_df.to_string(index=False))
        
        # CSV保存
        csv_path = output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nComparison results saved: {csv_path}")
        
        # PR曲線描画
        pr_curve_path = output_dir / "precision_recall_curves.png"
        evaluator.plot_precision_recall_curve(all_results, str(pr_curve_path))
        
        # 結果JSON保存
        results_json = {
            'evaluation_config': {
                'confidence_threshold': args.conf_threshold,
                'iou_threshold': args.iou_threshold,
                'evaluation_date': datetime.now().isoformat()
            },
            'results': all_results
        }
        
        json_path = output_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            # NumPy配列をリストに変換してJSON化
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            json.dump(results_json, f, indent=2, default=convert_numpy, ensure_ascii=False)
        
        print(f"Detailed results saved: {json_path}")

if __name__ == "__main__":
    main()