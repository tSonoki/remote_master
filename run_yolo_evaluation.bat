@echo off
echo ===============================================
echo YOLO11 Model Evaluation Script
echo ===============================================
echo.

REM 必要なライブラリをインストール
echo Installing required libraries...
pip install ultralytics opencv-python matplotlib seaborn scikit-learn pandas numpy

echo.
echo ===============================================
echo Creating sample dataset...
echo ===============================================
python create_evaluation_dataset.py --mode sample --output sample_dataset.json

echo.
echo ===============================================  
echo Running YOLO evaluation...
echo ===============================================

REM yolo11n.ptとyolo11n.onnxを比較（.ptファイルがある場合）
if exist yolo11n.pt (
    if exist old_model.pt (
        echo Comparing yolo11n.pt with old_model.pt...
        python yolo_evaluation.py --models yolo11n.pt old_model.pt --dataset sample_dataset.json --output-dir evaluation_results
    ) else (
        echo Evaluating single model yolo11n.pt...
        python yolo_evaluation.py --models yolo11n.pt --dataset sample_dataset.json --output-dir evaluation_results
    )
) else (
    echo Error: yolo11n.pt not found. Please ensure the YOLO model file exists.
    echo.
    echo You can download YOLOv11 models with:
    echo python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
    pause
    exit /b 1
)

echo.
echo ===============================================
echo Evaluation completed!
echo ===============================================
echo Results saved in: evaluation_results/
echo.
echo Generated files:
echo - confusion_matrix_*.png
echo - precision_recall_curves.png  
echo - model_comparison_*.csv
echo - detailed_results_*.json
echo.

pause