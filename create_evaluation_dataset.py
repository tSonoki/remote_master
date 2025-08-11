#!/usr/bin/env python3
"""
YOLO評価用データセット作成ヘルパー
"""

import json
import os
import cv2
from pathlib import Path
from typing import List, Dict
import argparse

def create_dataset_from_yolo_format(images_dir: str, labels_dir: str, 
                                   output_path: str) -> Dict:
    """
    YOLO形式のデータセット（images/labels）から評価用データセットを作成
    
    Args:
        images_dir: 画像ディレクトリパス
        labels_dir: ラベルディレクトリパス（.txtファイル）
        output_path: 出力JSONファイルパス
        
    Returns:
        データセット設定辞書
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    dataset_config = {
        "images": [],
        "annotations": []
    }
    
    # 画像ファイルを取得
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in images_path.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    for image_file in sorted(image_files):
        # 対応するラベルファイルを探す
        label_file = labels_path / (image_file.stem + '.txt')
        
        if not label_file.exists():
            print(f"Warning: Label file not found for {image_file.name}")
            continue
        
        # 画像サイズを取得
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"Warning: Could not load image {image_file}")
            continue
            
        img_height, img_width = img.shape[:2]
        
        # ラベルファイルを読み込み
        annotations = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    # YOLO形式 (center_x, center_y, width, height) -> (x1, y1, x2, y2)
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    
                    annotations.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": class_id
                    })
        
        except Exception as e:
            print(f"Error reading label file {label_file}: {e}")
            continue
        
        dataset_config["images"].append(str(image_file))
        dataset_config["annotations"].append(annotations)
    
    # JSON保存
    with open(output_path, 'w') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset created: {len(dataset_config['images'])} images")
    print(f"Output saved: {output_path}")
    
    return dataset_config

def create_sample_dataset_with_images(output_path: str = "sample_dataset.json"):
    """
    テスト用のサンプルデータセットを作成
    """
    # 現在のディレクトリから画像を探す
    current_dir = Path(".")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    available_images = [str(f) for f in current_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
    
    if not available_images:
        print("No images found in current directory. Creating minimal sample dataset.")
        dataset_config = {
            "images": ["sample1.jpg", "sample2.jpg", "sample3.jpg"],
            "annotations": [
                [{"bbox": [100, 100, 200, 300], "class": 0}],  # 人1人
                [{"bbox": [150, 80, 250, 280], "class": 0}, 
                 {"bbox": [300, 120, 400, 320], "class": 0}],   # 人2人
                []  # 人なし
            ]
        }
    else:
        print(f"Found {len(available_images)} images in current directory")
        dataset_config = {
            "images": available_images[:10],  # 最大10枚まで
            "annotations": []
        }
        
        # 各画像に対してサンプルアノテーションを作成
        for i, image_path in enumerate(dataset_config["images"]):
            # サンプルアノテーション（実際の画像内容とは関係なく、テスト用）
            if i % 3 == 0:
                # 人1人
                annotations = [{"bbox": [100, 100, 200, 300], "class": 0}]
            elif i % 3 == 1:
                # 人2人
                annotations = [
                    {"bbox": [80, 80, 180, 280], "class": 0},
                    {"bbox": [250, 100, 350, 300], "class": 0}
                ]
            else:
                # 人なし
                annotations = []
            
            dataset_config["annotations"].append(annotations)
    
    # JSON保存
    with open(output_path, 'w') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    
    print(f"Sample dataset created: {output_path}")
    return dataset_config

def main():
    parser = argparse.ArgumentParser(description='Create evaluation dataset')
    parser.add_argument('--mode', choices=['yolo', 'sample'], default='sample',
                       help='Dataset creation mode')
    parser.add_argument('--images-dir', type=str,
                       help='Images directory (for yolo mode)')
    parser.add_argument('--labels-dir', type=str,
                       help='Labels directory (for yolo mode)')
    parser.add_argument('--output', type=str, default='evaluation_dataset.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    if args.mode == 'yolo':
        if not args.images_dir or not args.labels_dir:
            print("Error: --images-dir and --labels-dir are required for yolo mode")
            return
        
        create_dataset_from_yolo_format(args.images_dir, args.labels_dir, args.output)
    
    elif args.mode == 'sample':
        create_sample_dataset_with_images(args.output)

if __name__ == "__main__":
    main()