import cv2
import numpy as np
import yaml # YAMLを扱うためのライブラリ

# --- 1. パラメータの読み込み ---
# YAMLファイルからカメラパラメータを読み込みます。
yaml_file = 'fish.yaml'
image_file = 'yugamima.jpg' # 補正する画像ファイル（適宜変更してください）

try:
    with open(yaml_file, 'r') as f:
        # YAMLファイルを辞書として読み込む
        params = yaml.safe_load(f)
except FileNotFoundError:
    print(f"エラー: パラメータファイル '{yaml_file}' が見つかりません。")
    exit()

# YAMLファイルから読み込んだ値を取得 
# 'data'キーのリストをNumPy配列に変換し、適切な形に変形します。
camera_matrix = np.array(params['camera_matrix']['data']).reshape(3, 3)
dist_coeffs = np.array(params['distortion_coefficients']['data'])
width = params['image_width']
height = params['image_height']

print("--- 読み込んだパラメータ ---")
print(f"Camera Matrix:\n{camera_matrix}")
print(f"Distortion Coefficients:\n{dist_coeffs}")
print("--------------------------")

# --- 2. 画像の読み込み ---
img = cv2.imread(image_file)

if img is None:
    print(f"エラー: 画像ファイル '{image_file}' を読み込めませんでした。")
else:
    h, w = img.shape[:2]
    
    # --- 3. 歪み補正の実行 ---
    # 前回と同様の処理
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # --- 4. 結果の保存と表示 ---
    output_file = 'corrected_image_from_yaml.jpg'
    cv2.imwrite(output_file, undistorted_img)
    print(f"✅ 補正後の画像を '{output_file}' として保存しました。")

    # (任意) 表示して比較
    resized_original = cv2.resize(img, (w // 2, h // 2))
    resized_undistorted = cv2.resize(undistorted_img, (w // 2, h // 2))
    comparison_image = np.hstack((resized_original, resized_undistorted))
    
    cv2.imshow('Original vs. Corrected (from YAML)', comparison_image)
    print("比較ウィンドウが表示されています。何かキーを押すと終了します。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()