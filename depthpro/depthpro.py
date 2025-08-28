import numpy as np
import cv2
import torch
from ultralytics import YOLO
import depth_pro
# Load YOLOv11 model and process image
try:
    yolo_model = YOLO('yolo11n.pt') # Use nano model which will be auto-downloaded
    image_path = 'person.jpg'
    yolo_input = cv2.imread(image_path)
    if yolo_input is None:
        raise ValueError(f"Could not load image from {image_path}")
    results = yolo_model(yolo_input)
except Exception as e:
    print(f"Error loading YOLO model or processing image: {e}")
    exit()

# Detect persons and get bounding boxes
person_boxes = []
for result in results:
 boxes = result.boxes.xyxy.cpu().numpy() # Get bounding boxes
 classes = result.boxes.cls.cpu().numpy() # Get class labels
for box, cls in zip(boxes, classes):
 if result.names[int(cls)] == 'person': # Filter for person class
  x1, y1, x2, y2 = map(int, box[:4])
  person_boxes.append((x1, y1, x2, y2))
  cv2.rectangle(yolo_input, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw rectangle
 
# Load depth model and preprocessing transform
try:
    print("Loading depth model...")
    depth_model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cpu")
    )
    depth_model.eval()
    print("Model loaded successfully")
    
    # Prepare image for depth estimation
    print("Preparing image for depth estimation...")
    image, _, f_px = depth_pro.load_rgb(image_path)
    depth_input = transform(image)
    print("Image prepared")

    # Perform depth inference
    print("Performing depth inference...")
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth = prediction["depth"] # Depth in meters
    # Convert depth to numpy array
    depth_np = depth.squeeze().cpu().numpy()
    print("Depth inference completed")
except Exception as e:
    print(f"Error loading depth model or performing inference: {e}")
    exit()

# Calculate depth for detected persons and display on image
for x1, y1, x2, y2 in person_boxes:
 center_x = (x1 + x2) // 2
 center_y = (y1 + y2) // 2
# Extract depth value at the center of the bounding box
 depth_value = depth_np[center_y, center_x]
 text = f'Depth: {depth_value:.2f}m'
# Define font properties
 font = cv2.FONT_HERSHEY_SIMPLEX
 font_scale = 1.2
 font_thickness = 2
 text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
# Set text position
 text_x = x1
 text_y = y1 - 10
# Create a rectangle for text background
 rect_x1 = text_x - 10
 rect_y1 = text_y - text_size[1] - 5
 rect_x2 = text_x + text_size[0] + 5
 rect_y2 = text_y + 5
# Draw the background rectangle and add text
 cv2.rectangle(yolo_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
 cv2.putText(yolo_input, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
 
# Display person detection with depth values
cv2.imshow('Person Detection with Depth', yolo_input)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save the image with detection and depth annotations
cv2.imwrite('person_detection_with_depth.jpg', yolo_input)