import numpy as np
import cv2
import torch
from ultralytics import YOLO
import depth_pro
import time

class RealTimeDepthEstimator:
    def __init__(self, yolo_model_path='../yolo11n.pt', camera_index=0, device='cpu'):
        self.device = torch.device(device)
        self.camera_index = camera_index
        
        try:
            print("Loading YOLO model...")
            self.yolo_model = YOLO(yolo_model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
        
        try:
            print("Loading depth model...")
            self.depth_model, self.transform = depth_pro.create_model_and_transforms(
                device=self.device
            )
            self.depth_model.eval()
            print("Depth model loaded successfully")
        except Exception as e:
            print(f"Error loading depth model: {e}")
            raise
        
        self.cap = None
        self.fps_counter = 0
        self.start_time = time.time()
        
    def initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open camera with index {self.camera_index}")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f"Camera initialized successfully with resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_persons(self, frame):
        try:
            results = self.yolo_model(frame, verbose=False)
            person_boxes = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    for box, cls in zip(boxes, classes):
                        if result.names[int(cls)] == 'person':
                            x1, y1, x2, y2 = map(int, box[:4])
                            person_boxes.append((x1, y1, x2, y2))
            
            return person_boxes
        except Exception as e:
            print(f"Error in person detection: {e}")
            return []
    
    def estimate_depth(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            height, width = frame_rgb.shape[:2]
            f_px = width * 0.8
            
            image_tensor = self.transform(frame_rgb).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.depth_model.infer(image_tensor, f_px=f_px)
                depth = prediction["depth"]
                depth_np = depth.squeeze().cpu().numpy()
            
            return depth_np
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return None
    
    def draw_annotations(self, frame, person_boxes, depth_map):
        annotated_frame = frame.copy()
        
        for x1, y1, x2, y2 in person_boxes:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if depth_map is not None:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                    depth_value = depth_map[center_y, center_x]
                    text = f'Depth: {depth_value:.2f}m'
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    
                    text_x = x1
                    text_y = y1 - 10
                    
                    rect_x1 = text_x - 5
                    rect_y1 = text_y - text_size[1] - 5
                    rect_x2 = text_x + text_size[0] + 5
                    rect_y2 = text_y + 5
                    
                    cv2.rectangle(annotated_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
                    cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        return annotated_frame
    
    def calculate_fps(self):
        self.fps_counter += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:
            fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.start_time = time.time()
            return fps
        return None
    
    def run(self):
        if not self.initialize_camera():
            return
        
        print("Starting real-time depth estimation...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset camera connection")
        
        frame_count = 0
        depth_estimation_interval = 3
        cached_depth_map = None
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                person_boxes = self.detect_persons(frame)
                
                if frame_count % depth_estimation_interval == 0:
                    cached_depth_map = self.estimate_depth(frame)
                
                annotated_frame = self.draw_annotations(frame, person_boxes, cached_depth_map)
                
                fps = self.calculate_fps()
                if fps is not None:
                    fps_text = f'FPS: {fps:.1f}'
                    cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Real-time Person Detection with Depth', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    try:
                        filename = f'realtime_depth_frame_{int(time.time())}.jpg'
                        cv2.imwrite(filename, annotated_frame)
                        print(f"Frame saved as {filename}")
                    except Exception as e:
                        print(f"Error saving frame: {e}")
                elif key == ord('r'):
                    print("Resetting camera connection...")
                    self.cap.release()
                    time.sleep(1)
                    if not self.initialize_camera():
                        break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping due to keyboard interrupt...")
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

def main():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        estimator = RealTimeDepthEstimator(
            yolo_model_path='../yolo11n.pt',
            camera_index=0,
            device=device
        )
        
        estimator.run()
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()