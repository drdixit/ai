"""
Hand-Only Tracker - AMD GPU Accelerated
Tracks ONLY the hands of the first detected person
Simple, fast, focused
"""
import cv2
import numpy as np
import mediapipe as mp
import argparse
import time
from collections import deque

class HandOnlyTracker:
    """Simple hand-only tracker for the first person detected"""
    
    def __init__(self):
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        print("=" * 60)
        print("HAND-ONLY TRACKER")
        print("=" * 60)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        print("✓ Hand tracker initialized")
        print("=" * 60)
    
    def detect_hands(self, frame):
        """Detect hands in frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            h, w = frame.shape[:2]
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label (Left/Right)
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                
                # Extract keypoints
                kpts = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    kpts.append([x, y, landmark.visibility])
                
                hands_data.append({
                    'keypoints': np.array(kpts),
                    'label': label,
                    'score': score
                })
        
        return hands_data
    
    def draw_hands(self, frame, hands_data):
        """Draw hand landmarks and skeleton"""
        # Hand connections (MediaPipe format)
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
        
        for hand in hands_data:
            kpts = hand['keypoints']
            label = hand['label']
            score = hand['score']
            
            # Choose color based on hand
            hand_color = (255, 0, 255) if label == 'Right' else (0, 255, 255)
            
            # Draw connections
            for i, j in connections:
                x1, y1, _ = kpts[i]
                x2, y2, _ = kpts[j]
                cv2.line(frame, (x1, y1), (x2, y2), hand_color, 3)
            
            # Draw keypoints
            for i, (x, y, _) in enumerate(kpts):
                if i == 0:  # Wrist
                    color = (0, 255, 0)
                    radius = 8
                elif i in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (0, 255, 0)
                    radius = 6
                else:  # Joints
                    color = hand_color
                    radius = 4
                
                cv2.circle(frame, (x, y), radius, color, -1)
                cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)
            
            # Draw hand label
            wrist_x, wrist_y, _ = kpts[0]
            cv2.putText(frame, f"{label} ({score:.2f})", 
                       (wrist_x - 40, wrist_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
    
    def update_fps(self):
        """Calculate FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)
    
    def release(self):
        """Release resources"""
        self.hands.close()

def main():
    parser = argparse.ArgumentParser(description='Hand-Only Tracker')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = HandOnlyTracker()
    
    # Camera setup
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print("\nControls: Q=Quit | S=Toggle Stats")
    print("Starting hand tracking...\n")
    
    show_stats = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror for intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        hands_data = tracker.detect_hands(frame)
        
        # Draw hands
        tracker.draw_hands(frame, hands_data)
        
        # Calculate FPS
        fps = tracker.update_fps()
        
        # Stats overlay
        if show_stats:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (280, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Hands: {len(hands_data)}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show hand labels
            if hands_data:
                labels = ", ".join([h['label'] for h in hands_data])
                cv2.putText(frame, labels, (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Hand Tracker', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_stats = not show_stats
    
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nHand tracking stopped.")

if __name__ == '__main__':
    main()
