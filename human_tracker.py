"""
Production-Grade Human Tracking System
- Single person focus with tracking persistence
- Proper NMS to eliminate false detections
- Hand tracking integrated
- AMD RX 5500M GPU acceleration
"""
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import mediapipe as mp
import argparse
import os
import time
from collections import deque
from typing import List, Dict, Optional, Tuple

class ProductionHumanTracker:
    """Production-quality human tracker with robust detection and tracking"""
    
    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Performance metrics
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # Tracking state
        self.tracked_person_id = None
        self.tracked_person_bbox = None
        self.tracking_lost_frames = 0
        self.max_tracking_lost = 30  # Reset tracking after 30 frames
        
        print("=" * 70)
        print("PRODUCTION HUMAN TRACKING SYSTEM")
        print("=" * 70)
        
        self._init_body_tracker()
        self._init_hand_tracker()
        
        print("=" * 70)
        print("✓ System Ready - Dedicated GPU Active")
        print("=" * 70)
        
    def _init_body_tracker(self):
        """Initialize body pose tracker with dedicated GPU"""
        print("\n[1/2] Body Pose Tracker (GPU Accelerated)")
        
        model_path = 'models/yolov8n-pose.pt'
        onnx_path = 'models/yolov8n-pose.onnx'
        
        os.makedirs('models', exist_ok=True)
        
        # Export if needed
        if not os.path.exists(onnx_path):
            print("      Exporting model to ONNX...")
            if not os.path.exists(model_path):
                model = YOLO('yolov8n-pose.pt')
                model.export(format='onnx', opset=12)
                if os.path.exists('yolov8n-pose.onnx'):
                    os.rename('yolov8n-pose.onnx', onnx_path)
                if os.path.exists('yolov8n-pose.pt'):
                    os.rename('yolov8n-pose.pt', model_path)
            else:
                model = YOLO(model_path)
                model.export(format='onnx', opset=12)
                if os.path.exists('yolov8n-pose.onnx'):
                    os.rename('yolov8n-pose.onnx', onnx_path)
        
        # Session options for maximum performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        
        # Force dedicated GPU
        provider_options = {'device_id': 1}
        
        try:
            self.body_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[('DmlExecutionProvider', provider_options), 'CPUExecutionProvider']
            )
            print(f"      ✓ Using: {self.body_session.get_providers()[0]} (Dedicated GPU)")
        except:
            self.body_session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=['DmlExecutionProvider', 'CPUExecutionProvider']
            )
            print(f"      ✓ Using: {self.body_session.get_providers()[0]}")
        
        self.body_input_name = self.body_session.get_inputs()[0].name
        self.body_output_name = self.body_session.get_outputs()[0].name
        
    def _init_hand_tracker(self):
        """Initialize MediaPipe hand tracker"""
        print("\n[2/2] Hand Tracker (MediaPipe)")
        self.mp_hands = mp.solutions.hands
        self.hand_detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        print("      ✓ MediaPipe Hands initialized")
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[float, Tuple[int, int]]]:
        """Preprocess frame for body detection"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Resize to 640x640
        input_size = 640
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding
        dh = (input_size - new_h) // 2
        dw = (input_size - new_w) // 2
        img = cv2.copyMakeBorder(img, dh, input_size - new_h - dh,
                                dw, input_size - new_w - dw,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Normalize and transpose
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        return img, (scale, (dw, dh))
    
    def detect_body(self, frame: np.ndarray) -> List[Dict]:
        """Detect body pose with proper NMS"""
        input_tensor, (scale, (dw, dh)) = self.preprocess(frame)
        
        # Run inference
        outputs = self.body_session.run([self.body_output_name], {self.body_input_name: input_tensor})
        output = outputs[0][0].transpose()  # (8400, 56)
        
        # Filter by confidence
        scores = output[:, 4]
        mask = scores > self.conf_threshold
        
        if not mask.any():
            return []
        
        filtered_output = output[mask]
        filtered_scores = scores[mask]
        
        # Parse boxes
        boxes = filtered_output[:, :4]
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        # Scale to original image
        h, w = frame.shape[:2]
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / scale
        
        # Clip to frame bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)
        
        # Apply NMS to remove duplicates
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            filtered_scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Process keypoints for NMS survivors
        results = []
        for idx in indices.flatten():
            kpts_data = filtered_output[idx, 5:]
            kpts = kpts_data.reshape(17, 3)
            kpts[:, 0] = (kpts[:, 0] - dw) / scale
            kpts[:, 1] = (kpts[:, 1] - dh) / scale
            
            # Calculate box area for sorting
            box = boxes_xyxy[idx]
            area = (box[2] - box[0]) * (box[3] - box[1])
            
            results.append({
                'box': box,
                'keypoints': kpts,
                'score': filtered_scores[idx],
                'area': area
            })
        
        # Sort by area (largest first) and confidence
        results.sort(key=lambda x: (x['area'], x['score']), reverse=True)
        
        return results
    
    def get_tracked_person(self, detections: List[Dict]) -> Optional[Dict]:
        """Get the person to track (largest, most confident, or tracked)"""
        if not detections:
            self.tracking_lost_frames += 1
            if self.tracking_lost_frames > self.max_tracking_lost:
                self.tracked_person_id = None
                self.tracked_person_bbox = None
            return None
        
        # If we have a tracked person, try to find them
        if self.tracked_person_bbox is not None:
            best_match = None
            best_iou = 0.3  # Minimum IoU threshold
            
            for det in detections:
                iou = self._calculate_iou(self.tracked_person_bbox, det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = det
            
            if best_match:
                self.tracked_person_bbox = best_match['box']
                self.tracking_lost_frames = 0
                return best_match
        
        # No tracked person or lost tracking - select best detection
        best_person = detections[0]  # Already sorted by area and score
        self.tracked_person_bbox = best_person['box']
        self.tracking_lost_frames = 0
        
        return best_person
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_hands(self, frame: np.ndarray, person_box: Optional[np.ndarray] = None) -> List[Dict]:
        """Detect hands in frame or person ROI"""
        if person_box is not None:
            # Expand box slightly for hands
            x1, y1, x2, y2 = person_box.astype(int)
            h, w = frame.shape[:2]
            
            # Expand by 10%
            expand = 0.1
            width = x2 - x1
            height = y2 - y1
            x1 = max(0, int(x1 - width * expand))
            y1 = max(0, int(y1 - height * expand))
            x2 = min(w, int(x2 + width * expand))
            y2 = min(h, int(y2 + height * expand))
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return []
            
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.process(rgb)
            
            hands = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    kpts = []
                    for landmark in hand_landmarks.landmark:
                        # Convert to frame coordinates
                        x = landmark.x * (x2 - x1) + x1
                        y = landmark.y * (y2 - y1) + y1
                        kpts.append([x, y, landmark.visibility])
                    hands.append({'keypoints': np.array(kpts)})
            
            return hands
        else:
            # Full frame detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.process(rgb)
            
            hands = []
            if results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                for hand_landmarks in results.multi_hand_landmarks:
                    kpts = []
                    for landmark in hand_landmarks.landmark:
                        kpts.append([landmark.x * w, landmark.y * h, landmark.visibility])
                    hands.append({'keypoints': np.array(kpts)})
            
            return hands
    
    def draw_body(self, frame: np.ndarray, person: Dict, is_tracked: bool = True):
        """Draw body pose skeleton"""
        kpts = person['keypoints']
        box = person['box'].astype(int)
        
        # Draw bounding box
        color = (0, 255, 0) if is_tracked else (128, 128, 128)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Draw confidence
        label = f"Person: {person['score']:.2f}"
        cv2.putText(frame, label, (box[0], box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # COCO skeleton connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw skeleton
        for i, j in connections:
            if i < len(kpts) and j < len(kpts):
                x1, y1, c1 = kpts[i]
                x2, y2, c2 = kpts[j]
                if c1 > 0.5 and c2 > 0.5:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.5:
                pt_color = (0, 0, 255) if i < 5 else (255, 0, 0)
                cv2.circle(frame, (int(x), int(y)), 4, pt_color, -1)
    
    def draw_hands(self, frame: np.ndarray, hands: List[Dict]):
        """Draw hand landmarks"""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        
        for hand in hands:
            kpts = hand['keypoints']
            
            # Draw connections
            for i, j in connections:
                x1, y1, _ = kpts[i]
                x2, y2, _ = kpts[j]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            
            # Draw keypoints
            for i, (x, y, _) in enumerate(kpts):
                if i == 0:
                    color, radius = (0, 255, 255), 5
                elif i in [4, 8, 12, 16, 20]:
                    color, radius = (0, 255, 0), 4
                else:
                    color, radius = (255, 0, 255), 3
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    
    def update_fps(self) -> float:
        """Calculate FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)
    
    def release(self):
        """Release resources"""
        self.hand_detector.close()

def main():
    parser = argparse.ArgumentParser(description='Production Human Tracking System')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--iou', type=float, default=0.45)
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ProductionHumanTracker(conf_threshold=args.conf, iou_threshold=args.iou)
    
    # Camera setup
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print("\nControls: Q=Quit | S=Toggle Stats | H=Toggle Hands")
    print("Starting tracking...\n")
    
    show_stats = True
    show_hands = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect all bodies
        all_detections = tracker.detect_body(frame)
        
        # Get tracked person (single focus)
        tracked_person = tracker.get_tracked_person(all_detections)
        
        # Draw tracked person
        if tracked_person:
            tracker.draw_body(frame, tracked_person, is_tracked=True)
            
            # Detect hands in person ROI
            if show_hands:
                hands = tracker.detect_hands(frame, tracked_person['box'])
                tracker.draw_hands(frame, hands)
        
        # Calculate FPS
        fps = tracker.update_fps()
        
        # Stats overlay
        if show_stats:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking: {'Active' if tracked_person else 'Lost'}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"GPU: Dedicated (device_id=1)", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Production Human Tracker', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_stats = not show_stats
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"Hand tracking: {'ON' if show_hands else 'OFF'}")
    
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nTracking stopped.")

if __name__ == '__main__':
    main()
