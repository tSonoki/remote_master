import cv2
import numpy as np
import yaml
import sys

def load_camera_params(yaml_file):
    """YAMLファイルからカメラパラメータを読み込む"""
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    
    # カメラマトリックス
    camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
    
    # 歪み係数
    dist_coeffs = np.array(data['distortion_coefficients']['data'])
    
    # 画像サイズ
    image_width = data['image_width']
    image_height = data['image_height']
    
    return camera_matrix, dist_coeffs, (image_width, image_height)

def undistort_image(input_image_path, yaml_file, output_image_path):
    """画像の歪み補正を実行"""
    # カメラパラメータの読み込み
    camera_matrix, dist_coeffs, image_size = load_camera_params(yaml_file)
    
    # 画像の読み込み
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Cannot load image {input_image_path}")
        return False
    
    # 歪み補正
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
    
    # 結果の保存
    cv2.imwrite(output_image_path, undistorted_img)
    print(f"Undistorted image saved to: {output_image_path}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python undistort_image.py <input_image> <yaml_file> <output_image>")
        sys.exit(1)
    
    input_image = sys.argv[1]
    yaml_file = sys.argv[2]
    output_image = sys.argv[3]
    
    success = undistort_image(input_image, yaml_file, output_image)
    if not success:
        sys.exit(1)