import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key
import argparse
import time
from collections import deque
import math

class CircularGestureController:
    """Circular motion-based volume controller"""
    
    def __init__(self):
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # Circular motion tracking
        self.finger_trail = deque(maxlen=15)  # Track last 15 positions
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.cooldown_frames = 3  # Reduced for faster response
        
        # Volume tracking
        self.volume_change_history = deque(maxlen=5)
        
        print("=" * 70)
        print("CIRCULAR GESTURE VOLUME CONTROL - AMD GPU")
        print("=" * 70)
        
        self._init_keyboard()
        self._init_hand_tracker()
        
        print("=" * 70)
        print("✓ System Ready")
        print("=" * 70)
    
    def _init_keyboard(self):
        """Initialize keyboard controller"""
        print("\n[1/2] Keyboard Control")
        self.keyboard = Controller()
        print("      ✓ Keyboard controller initialized")
    
    def _init_hand_tracker(self):
        """Initialize MediaPipe hand tracker"""
        print("\n[2/2] Hand Tracker (GPU Optimized)")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        print("      ✓ MediaPipe Hands ready")
    
    def detect_circular_motion(self):
        """Detect circular motion from finger trail"""
        if len(self.finger_trail) < 8:
            return None
        
        # Get recent positions
        points = list(self.finger_trail)
        
        # Calculate center of motion
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        # Calculate angles for each point relative to center
        angles = []
        for x, y in points:
            angle = math.atan2(y - center_y, x - center_x)
            angles.append(angle)
        
        # Calculate total rotation
        total_rotation = 0
        for i in range(1, len(angles)):
            # Calculate angle difference
            diff = angles[i] - angles[i-1]
            
            # Normalize to [-pi, pi]
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            
            total_rotation += diff
        
        # Check if we have significant circular motion
        # Positive rotation = clockwise, negative = counter-clockwise
        # Require almost a full circle (270 degrees = 1.5*pi radians)
        threshold = math.pi * 1.5  # About 270 degrees total rotation
        
        if total_rotation > threshold:
            return "CLOCKWISE"
        elif total_rotation < -threshold:
            return "COUNTER_CLOCKWISE"
        
        return None
    
    def adjust_volume(self, direction):
        """Adjust system volume using keyboard keys"""
        try:
            if direction == "CLOCKWISE":
                # Press volume up key 3 times for faster change
                for _ in range(3):
                    self.keyboard.press(Key.media_volume_up)
                    self.keyboard.release(Key.media_volume_up)
                    time.sleep(0.01)  # Small delay between presses
                action = "Volume UP ↻"
            elif direction == "COUNTER_CLOCKWISE":
                # Press volume down key 3 times for faster change
                for _ in range(3):
                    self.keyboard.press(Key.media_volume_down)
                    self.keyboard.release(Key.media_volume_down)
                    time.sleep(0.01)
                action = "Volume DOWN ↺"
            else:
                return
            
            self.volume_change_history.append((action, time.time()))
            
        except Exception as e:
            print(f"Volume error: {e}")
    
    def detect_hands(self, frame):
        """Detect hands and track finger position"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        hand_data = None
        finger_pos = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            h, w = frame.shape[:2]
            
            # Get index finger tip position
            index_tip = hand_landmarks.landmark[8]
            finger_x = int(index_tip.x * w)
            finger_y = int(index_tip.y * h)
            finger_pos = (finger_x, finger_y)
            
            # Add to trail
            self.finger_trail.append(finger_pos)
            
            # Extract all keypoints
            kpts = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                kpts.append([x, y, landmark.visibility])
            
            hand_data = {'keypoints': np.array(kpts)}
        else:
            # Clear trail if no hand detected
            self.finger_trail.clear()
        
        return hand_data, finger_pos
    
    def draw_hand(self, frame, hand_data):
        """Draw hand skeleton"""
        if hand_data is None:
            return
        
        kpts = hand_data['keypoints']
        
        # Hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Draw connections
        for i, j in connections:
            x1, y1, _ = kpts[i]
            x2, y2, _ = kpts[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        
        # Draw keypoints
        for i, (x, y, _) in enumerate(kpts):
            if i == 8:  # Index fingertip (most important)
                color = (0, 255, 255)
                radius = 10
            elif i == 0:  # Wrist
                color = (255, 0, 255)
                radius = 8
            else:
                color = (255, 255, 255)
                radius = 4
            
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
            cv2.circle(frame, (int(x), int(y)), radius + 2, (0, 0, 0), 1)
    
    def draw_trail(self, frame, gesture):
        """Draw finger trail with color based on gesture"""
        if len(self.finger_trail) < 2:
            return
        
        # Color based on detected gesture
        if gesture == "CLOCKWISE":
            color = (0, 255, 0)  # Green
        elif gesture == "COUNTER_CLOCKWISE":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 200, 0)  # Cyan
        
        # Draw trail
        points = list(self.finger_trail)
        for i in range(1, len(points)):
            thickness = int(2 + (i / len(points)) * 4)
            cv2.line(frame, points[i-1], points[i], color, thickness)
        
        # Draw arrow at the end to show direction
        if len(points) >= 3:
            p1 = points[-3]
            p2 = points[-1]
            cv2.arrowedLine(frame, p1, p2, color, 3, tipLength=0.3)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()
    
    controller = CircularGestureController()
    
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera")
        return
    
    print("\n" + "=" * 70)
    print("GESTURE CONTROLS:")
    print("  ↻ Clockwise circle        → Volume UP")
    print("  ↺ Counter-clockwise circle → Volume DOWN")
    print("  Q → Quit")
    print("=" * 70 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        hand_data, finger_pos = controller.detect_hands(frame)
        
        # Detect circular motion
        gesture = controller.detect_circular_motion()
        
        # Handle gesture with cooldown
        if gesture and controller.gesture_cooldown == 0:
            controller.adjust_volume(gesture)
            controller.last_gesture = gesture
            controller.gesture_cooldown = controller.cooldown_frames
        
        if controller.gesture_cooldown > 0:
            controller.gesture_cooldown -= 1
        
        # Draw visuals
        controller.draw_hand(frame, hand_data)
        controller.draw_trail(frame, gesture)
        
        # Calculate FPS
        fps = controller.update_fps()
        
        # UI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if gesture:
            gesture_text = f"Motion: {gesture}"
            color = (0, 255, 0) if gesture == "CLOCKWISE" else (0, 0, 255)
            cv2.putText(frame, gesture_text, (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if controller.volume_change_history:
            recent = controller.volume_change_history[-1]
            action, timestamp = recent
            cv2.putText(frame, action, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, "Draw circles with index finger", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Circular Gesture Volume Control', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    controller.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\nGesture control stopped.")

if __name__ == '__main__':
    main()
