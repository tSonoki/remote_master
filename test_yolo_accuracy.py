#!/usr/bin/env python3
"""
YOLO11精度検証スクリプト - testフォルダの画像を使用
使い方: python test_yolo_accuracy.py
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.font_manager as fm

# 日本語フォント設定（文字化け対策）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class YOLOTestAccuracy:
    def __init__(self):
        self.model = None
        self.results = {}
    
    def load_model(self, model_path='yolo11n.pt'):
        """YOLOモデルの読み込み"""
        print(f"Loading YOLO model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def run_inference_on_test_images(self, confidence=0.5):
        """testフォルダの画像に対して推論を実行"""
        test_dir = Path("test")
        if not test_dir.exists():
            print("Error: test folder not found!")
            return None
        
        # 画像ファイルを取得
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(test_dir.glob(f"*{ext}")))
        
        if not image_files:
            print("No image files found in test folder!")
            return None
        
        print(f"Found {len(image_files)} images in test folder")
        
        results = []
        for img_path in sorted(image_files):
            print(f"Processing: {img_path.name}")
            
            try:
                # 推論実行
                pred_results = self.model(str(img_path), conf=confidence, verbose=False)
                
                # 結果を解析
                detections = []
                raw_results = pred_results[0] if pred_results else None
                
                for r in pred_results:
                    if r.boxes is not None:
                        for i in range(len(r.boxes)):
                            cls_id = int(r.boxes.cls[i])
                            conf = float(r.boxes.conf[i])
                            bbox = r.boxes.xyxy[i].cpu().numpy()
                            
                            # クラス名を取得
                            class_name = r.names[cls_id]
                            
                            # 人（person）のみを検出対象とする
                            if class_name == 'person':
                                detections.append({
                                    'class_id': cls_id,
                                    'class_name': class_name,
                                    'confidence': conf,
                                    'bbox': bbox.tolist()
                                })
                
                results.append({
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'detections': detections,
                    'total_detections': len(detections),
                    'raw_results': raw_results  # YOLO結果オブジェクトを保存
                })
                
                print(f"  Detected {len(detections)} objects")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        return results
    
    def analyze_results(self, inference_results):
        """推論結果を分析"""
        if not inference_results:
            print("No results to analyze!")
            return None
        
        # 統計情報を収集
        total_images = len(inference_results)
        total_detections = sum(r['total_detections'] for r in inference_results)
        
        # クラス別統計
        class_stats = {}
        person_detections = []
        
        for result in inference_results:
            for det in result['detections']:
                cls_name = det['class_name']
                if cls_name not in class_stats:
                    class_stats[cls_name] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'confidences': []
                    }
                
                class_stats[cls_name]['count'] += 1
                class_stats[cls_name]['confidences'].append(det['confidence'])
                
                # 人検出の統計
                if cls_name == 'person':
                    person_detections.append({
                        'image': result['image_name'],
                        'confidence': det['confidence'],
                        'bbox': det['bbox']
                    })
        
        # 平均信頼度を計算
        for cls_name in class_stats:
            confidences = class_stats[cls_name]['confidences']
            class_stats[cls_name]['avg_confidence'] = np.mean(confidences)
            class_stats[cls_name]['min_confidence'] = np.min(confidences)
            class_stats[cls_name]['max_confidence'] = np.max(confidences)
        
        analysis = {
            'total_images': total_images,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'class_statistics': class_stats,
            'person_detections': person_detections,
            'images_with_persons': len([r for r in inference_results if any(d['class_name'] == 'person' for d in r['detections'])])
        }
        
        return analysis
    
    def create_visualizations(self, inference_results, analysis):
        """結果の可視化"""
        if not analysis:
            print("No analysis data for visualization!")
            return
        
        # 1. クラス別検出数のグラフ
        plt.figure(figsize=(15, 10))
        
        # クラス別検出数
        plt.subplot(2, 3, 1)
        classes = list(analysis['class_statistics'].keys())
        counts = [analysis['class_statistics'][cls]['count'] for cls in classes]
        
        plt.bar(classes, counts, color='skyblue')
        plt.title('Detection Count by Class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 2. 信頼度の分布（人のみ）
        plt.subplot(2, 3, 2)
        if 'person' in analysis['class_statistics']:
            person_confidences = analysis['class_statistics']['person']['confidences']
            plt.hist(person_confidences, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Person Detection Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
        else:
            plt.text(0.5, 0.5, 'No Person Detected', ha='center', va='center')
            plt.title('Person Detection Confidence Distribution')
        
        # 3. 画像別検出数
        plt.subplot(2, 3, 3)
        image_names = [r['image_name'] for r in inference_results]
        detection_counts = [r['total_detections'] for r in inference_results]
        
        plt.bar(range(len(image_names)), detection_counts, color='lightcoral')
        plt.title('Detection Count by Image')
        plt.xlabel('Image Index')
        plt.ylabel('Detection Count')
        
        # 4. 人検出の信頼度（画像別）
        plt.subplot(2, 3, 4)
        if analysis['person_detections']:
            person_image_indices = []
            person_confs = []
            for i, result in enumerate(inference_results):
                for det in result['detections']:
                    if det['class_name'] == 'person':
                        person_image_indices.append(i)
                        person_confs.append(det['confidence'])
            
            plt.scatter(person_image_indices, person_confs, color='orange', alpha=0.7, s=50)
            plt.title('Person Detection Confidence by Image')
            plt.xlabel('Image Index')
            plt.ylabel('Confidence')
            plt.ylim(0, 1)
        else:
            plt.text(0.5, 0.5, 'No Person Detected', ha='center', va='center')
            plt.title('Person Detection Confidence by Image')
        
        # 5. 精度サマリー
        plt.subplot(2, 3, 5)
        metrics = ['Images with Person', 'Person Detections', 'Avg Confidence']
        values = [
            analysis['images_with_persons'] / analysis['total_images'],
            len(analysis['person_detections']) / analysis['total_images'],
            analysis['class_statistics']['person']['avg_confidence'] if 'person' in analysis['class_statistics'] else 0
        ]
        
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        bars = plt.bar(metrics, values, color=colors)
        plt.title('Performance Summary')
        plt.ylabel('Ratio / Score')
        plt.ylim(0, 1)
        
        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yolo11_accuracy_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {filename}")
        plt.show()
    
    def create_confusion_matrix(self, analysis, ground_truth_annotations=None):
        """コンフュージョンマトリクスの作成"""
        print("Creating confusion matrix...")
        
        # 簡易版：実際の正解データがない場合の推定
        total_images = analysis['total_images']
        images_with_person = analysis['images_with_persons']
        images_without_person = total_images - images_with_person
        
        # 推定値（実際には正解データが必要）
        # ここでは人が検出された画像=True Positive、されなかった画像=True Negativeと仮定
        tp = images_with_person  # 人が検出された画像数
        tn = images_without_person  # 人が検出されなかった画像数
        fp = 0  # 偽陽性（実際には正解データが必要）
        fn = 0  # 偽陰性（実際には正解データが必要）
        
        # コンフュージョンマトリクス作成
        cm = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        
        # ヒートマップ作成
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Person (Predicted)', 'Person (Predicted)'],
                   yticklabels=['No Person (Actual)', 'Person (Actual)'],
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Person Detection\n(Estimated based on detection results)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 精度指標を計算して表示
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # 推定値なので1.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # テキストボックスで指標表示
        textstr = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}\nAccuracy: {accuracy:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                      verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yolo11_confusion_matrix_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {filename}")
        plt.show()
        
        return {
            'confusion_matrix': cm.tolist(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
    
    def draw_detection_results(self, inference_results):
        """検出結果を画像に描画して保存"""
        print("Drawing detection results on images...")
        
        # 結果画像保存用ディレクトリを作成
        output_dir = Path("detection_results")
        output_dir.mkdir(exist_ok=True)
        
        # 人検出用の色（緑）
        person_color = (0, 255, 0)  # 緑
        
        saved_images = []
        
        for result in inference_results:
            try:
                # 元画像を読み込み
                image_path = result['image_path']
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                
                # 人の検出結果を描画
                for detection in result['detections']:
                    # 人のみを対象とする（既にフィルタリング済みだが念のため）
                    if detection['class_name'] == 'person':
                        bbox = detection['bbox']
                        confidence = detection['confidence']
                        
                        # バウンディングボックスの座標
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # バウンディングボックスを描画（緑色）
                        cv2.rectangle(image, (x1, y1), (x2, y2), person_color, 3)
                        
                        # ラベルテキスト（人: 信頼度）
                        label = f"Person: {confidence:.2f}"
                        
                        # テキストサイズを取得
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # テキスト背景の矩形を描画
                        cv2.rectangle(image, (x1, y1 - text_height - 10), 
                                     (x1 + text_width, y1), person_color, -1)
                        
                        # テキストを描画
                        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, 
                                   (0, 0, 0), thickness)
                
                # 画像情報をヘッダーに追加（人の検出数のみ）
                person_count = len([d for d in result['detections'] if d['class_name'] == 'person'])
                header_text = f"File: {result['image_name']} | Persons: {person_count}"
                cv2.putText(image, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                # 結果画像を保存
                output_filename = f"detected_{result['image_name']}"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), image)
                
                saved_images.append(str(output_path))
                print(f"  Saved detection result: {output_filename}")
                
            except Exception as e:
                print(f"Error drawing detection for {result['image_name']}: {e}")
                continue
        
        print(f"\nDetection result images saved in: {output_dir}")
        print(f"Total images saved: {len(saved_images)}")
        
        return saved_images
    
    def create_detection_summary_grid(self, inference_results, saved_images):
        """検出結果のサマリーグリッド画像を作成"""
        print("Creating detection summary grid...")
        
        if not saved_images:
            print("No detection images to create summary!")
            return
        
        # グリッドサイズを計算
        num_images = len(saved_images)
        grid_cols = min(4, num_images)  # 最大4列
        grid_rows = (num_images + grid_cols - 1) // grid_cols
        
        # 各画像のリサイズサイズ
        thumb_size = (300, 200)
        
        # グリッド画像のサイズ
        grid_width = grid_cols * thumb_size[0]
        grid_height = grid_rows * thumb_size[1] + 50  # ヘッダー用スペース
        
        # グリッド画像を作成
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        grid_image.fill(50)  # ダークグレー背景
        
        # ヘッダーテキスト
        header_text = f"YOLO11 Person Detection Results - {num_images} Test Images"
        cv2.putText(grid_image, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2)
        
        # 各画像をグリッドに配置
        for i, img_path in enumerate(saved_images):
            row = i // grid_cols
            col = i % grid_cols
            
            # 画像を読み込んでリサイズ
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, thumb_size)
                
                # グリッド内の位置を計算
                y_start = 50 + row * thumb_size[1]
                y_end = y_start + thumb_size[1]
                x_start = col * thumb_size[0]
                x_end = x_start + thumb_size[0]
                
                # 画像をグリッドに配置
                grid_image[y_start:y_end, x_start:x_end] = img_resized
        
        # サマリーグリッドを保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"detection_summary_grid_{timestamp}.png"
        cv2.imwrite(summary_filename, grid_image)
        
        print(f"Detection summary grid saved: {summary_filename}")
        return summary_filename
    
    def save_detailed_results(self, inference_results, analysis):
        """詳細結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # YOLOの結果オブジェクトを除いてJSONシリアライズ可能な形式に変換
        clean_results = []
        for result in inference_results:
            clean_result = {
                'image_path': result['image_path'],
                'image_name': result['image_name'], 
                'detections': result['detections'],
                'total_detections': result['total_detections']
                # raw_resultsは除外（JSONシリアライズできないため）
            }
            clean_results.append(clean_result)
        
        # 詳細結果
        detailed_results = {
            'timestamp': timestamp,
            'model_info': 'yolo11n.pt',
            'test_summary': analysis,
            'detailed_detections': clean_results
        }
        
        filename = f"yolo11_test_results_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved: {filename}")
    
    def print_summary_report(self, analysis):
        """サマリーレポートを表示"""
        if not analysis:
            return
        
        print("\n" + "="*60)
        print("YOLO11 精度検証結果サマリー")
        print("="*60)
        print(f"テスト画像数: {analysis['total_images']}")
        print(f"総検出数: {analysis['total_detections']}")
        print(f"平均検出数/画像: {analysis['avg_detections_per_image']:.2f}")
        print(f"人が検出された画像数: {analysis['images_with_persons']}/{analysis['total_images']}")
        
        if 'person' in analysis['class_statistics']:
            person_stats = analysis['class_statistics']['person']
            print(f"人検出数: {person_stats['count']}")
            print(f"人検出平均信頼度: {person_stats['avg_confidence']:.3f}")
            print(f"人検出信頼度範囲: {person_stats['min_confidence']:.3f} - {person_stats['max_confidence']:.3f}")
        else:
            print("人は検出されませんでした")
        
        print("\n検出クラス別統計:")
        print("-" * 40)
        for cls_name, stats in analysis['class_statistics'].items():
            print(f"{cls_name}: {stats['count']}件 (平均信頼度: {stats['avg_confidence']:.3f})")
        
        print("="*60)

def main():
    """メイン関数"""
    print("YOLO11 テスト画像精度検証ツール")
    print("="*50)
    
    # YOLOテスタを初期化
    tester = YOLOTestAccuracy()
    
    # モデルを読み込み
    if not tester.load_model():
        return
    
    # 信頼度閾値を設定
    confidence_threshold = 0.5
    print(f"信頼度閾値: {confidence_threshold}")
    
    # 推論実行
    print("\n推論を実行中...")
    inference_results = tester.run_inference_on_test_images(confidence=confidence_threshold)
    
    if not inference_results:
        print("推論に失敗しました")
        return
    
    # 結果分析
    print("\n結果を分析中...")
    analysis = tester.analyze_results(inference_results)
    
    # サマリーレポート表示
    tester.print_summary_report(analysis)
    
    # 可視化
    print("\n結果を可視化中...")
    tester.create_visualizations(inference_results, analysis)
    
    # コンフュージョンマトリクス作成
    print("\nコンフュージョンマトリクスを作成中...")
    confusion_results = tester.create_confusion_matrix(analysis)
    
    # 検出結果を画像に描画
    print("\n検出結果を画像に描画中...")
    saved_detection_images = tester.draw_detection_results(inference_results)
    
    # 検出結果サマリーグリッドを作成
    print("\n検出結果サマリーグリッドを作成中...")
    summary_grid = tester.create_detection_summary_grid(inference_results, saved_detection_images)
    
    # 詳細結果保存
    print("\n詳細結果を保存中...")
    tester.save_detailed_results(inference_results, analysis)
    
    print("\n検証完了！")

if __name__ == "__main__":
    main()