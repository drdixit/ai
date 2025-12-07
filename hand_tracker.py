"""
Hand Tracking Application with AMD GPU Acceleration (YOLOv8 + ONNX DirectML)
Optimized for Performance and Accuracy
"""
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import argparse
import sys
import os
import time
from collections import deque

class YOLOHandTracker:
    def __init__(self, model_path='models/hand_pose.pt', confidence_threshold=0.3, iou_threshold=0.4):
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # 1. Export Logic
        onnx_path = model_path.replace('.pt', '.onnx')
        if not os.path.exists(onnx_path):
            print(f"Exporting {model_path} to ONNX for GPU acceleration...")
            model = YOLO(model_path)
            model.export(format='onnx', opset=12)
            print("Export complete.")
        
        # 2. Initialize ONNX Runtime with DirectML
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        # Try DirectML first, fallback to CPU if it fails
        try:
            provider_options = [{'device_id': 1}]  # Use device 1 for dedicated GPU
            self.session = ort.InferenceSession(
                onnx_path, 
                sess_options=sess_options,
                providers=[('DmlExecutionProvider', provider_options[0]), 'CPUExecutionProvider']
            )
            print(f"✓ GPU Acceleration Active: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"Trying default DirectML device...")
            try:
                self.session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=['DmlExecutionProvider', 'CPUExecutionProvider']
                )
                print(f"✓ GPU Acceleration Active: {self.session.get_providers()[0]}")
            except Exception as e2:
                print(f"⚠ DirectML failed, using CPU")
                self.session = ort.InferenceSession(
                    onnx_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )

        # Get model details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Model Input Size: {self.input_width}x{self.input_height}")
        
    def preprocess(self, frame):
        """Optimized preprocessing with letterbox resize"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        shape = img.shape[:2]
        
        # Calculate resize ratio
        r = min(self.input_height / shape[0], self.input_width / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        
        # Padding
        dw = (self.input_width - new_unpad[0]) / 2
        dh = (self.input_height - new_unpad[1]) / 2
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Normalize and transpose
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return img, (r, (dw, dh))

    def postprocess(self, output, ratio):
        """Optimized postprocessing with proper NMS"""
        r, (dw, dh) = ratio
        
        # Transpose output: (1, 56, 8400) -> (8400, 56)
        output = output[0].transpose()
        
        # Extract scores (objectness)
        scores = output[:, 4]
        
        # Filter by confidence
        mask = scores > self.conf_threshold
        if not mask.any():
            return []
        
        output = output[mask]
        scores = scores[mask]
        
        # Parse boxes (xywh format)
        boxes = output[:, :4]
        
        # Convert xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Scale boxes back to original image
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / r
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / r
        
        # Extract keypoints (x, y, conf format)
        kpts_data = output[:, 5:]
        num_kpts = kpts_data.shape[1] // 3
        
        keypoints = []
        for i in range(len(output)):
            kpt = kpts_data[i].reshape(num_kpts, 3)
            # Scale keypoints
            kpt[:, 0] = (kpt[:, 0] - dw) / r
            kpt[:, 1] = (kpt[:, 1] - dh) / r
            keypoints.append(kpt)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                results.append({
                    'box': boxes_xyxy[idx],
                    'keypoints': keypoints[idx],
                    'score': scores[idx]
                })
        
        return results

    def detect(self, frame):
        """Run detection on frame"""
        input_tensor, ratio = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return self.postprocess(outputs[0], ratio)

    def draw_results(self, frame, results, show_box=True, show_keypoints=True):
        """Draw detection results with enhanced visualization"""
        for res in results:
            box = res['box'].astype(int)
            score = res['score']
            
            # Draw bounding box
            if show_box:
                color = (0, 255, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Draw confidence score
                label = f"{score:.2f}"
                cv2.putText(frame, label, (box[0], box[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw hand skeleton
            if show_keypoints:
                kpts = res['keypoints']
                
                # Hand skeleton connections (MediaPipe format)
                connections = [
                    # Thumb
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    # Index finger
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    # Middle finger
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    # Ring finger
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    # Pinky
                    (0, 17), (17, 18), (18, 19), (19, 20)
                ]
                
                # Draw connections
                for i, j in connections:
                    if i < len(kpts) and j < len(kpts):
                        x1, y1, c1 = kpts[i]
                        x2, y2, c2 = kpts[j]
                        if c1 > 0.3 and c2 > 0.3:
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                   (255, 0, 0), 2)
                
                # Draw keypoints
                for i, (x, y, conf) in enumerate(kpts):
                    if conf < 0.3:
                        continue
                    
                    # Color code: wrist=red, tips=green, others=blue
                    if i == 0:  # Wrist
                        color = (0, 0, 255)
                        radius = 5
                    elif i in [4, 8, 12, 16, 20]:  # Fingertips
                        color = (0, 255, 0)
                        radius = 4
                    else:
                        color = (255, 0, 0)
                        radius = 3
                    
                    cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        
        return frame
    
    def update_fps(self):
        """Calculate and return current FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)

def main():
    parser = argparse.ArgumentParser(description='Hand Tracking with AMD GPU Acceleration')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    args = parser.parse_args()

    # Initialize tracker
    tracker = YOLOHandTracker(confidence_threshold=args.conf, iou_threshold=args.iou)
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print("\n" + "="*60)
    print("Hand Tracking Started - AMD GPU Accelerated")
    print("="*60)
    print("Controls:")
    print("  q - Quit")
    print("  b - Toggle bounding boxes")
    print("  k - Toggle keypoints")
    print("  s - Toggle stats")
    print("="*60 + "\n")
    
    show_box = True
    show_keypoints = True
    show_stats = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Mirror for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Run detection
        results = tracker.detect(frame)
        
        # Draw results
        tracker.draw_results(frame, results, show_box, show_keypoints)
        
        # Calculate FPS
        fps = tracker.update_fps()
        
        # Draw stats overlay
        if show_stats:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Hands: {len(results)}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"GPU: {tracker.session.get_providers()[0][:3]}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display
        cv2.imshow('Hand Tracking - AMD GPU Accelerated', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            show_box = not show_box
            print(f"Bounding boxes: {'ON' if show_box else 'OFF'}")
        elif key == ord('k'):
            show_keypoints = not show_keypoints
            print(f"Keypoints: {'ON' if show_keypoints else 'OFF'}")
        elif key == ord('s'):
            show_stats = not show_stats
            print(f"Stats: {'ON' if show_stats else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nHand tracking stopped.")

if __name__ == '__main__':
    main()
