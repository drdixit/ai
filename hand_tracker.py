"""
Hand Tracking Application with AMD GPU Acceleration (YOLOv8 + ONNX DirectML)
"""
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import argparse
import sys
import os

class YOLOHandTracker:
    def __init__(self, model_path='models/hand_pose.pt', confidence_threshold=0.5, iou_threshold=0.45):
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 1. Export Logic
        onnx_path = model_path.replace('.pt', '.onnx')
        if not os.path.exists(onnx_path):
            print(f"Exporting {model_path} to ONNX for GPU acceleration...")
            model = YOLO(model_path)
            model.export(format='onnx', opset=12)
            print("Export complete.")
        
        # 2. Initialize ONNX Runtime with DirectML
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            print(f"Using providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"Error initializing DirectML: {e}")
            print("Falling back to CPU")
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Get model details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        self.output_name = self.session.get_outputs()[0].name
        
    def preprocess(self, frame):
        # Letterbox resize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = img.copy()
        shape = image.shape[:2]
        
        r = min(self.input_height / shape[0], self.input_width / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        
        dw, dh = self.input_width - new_unpad[0], self.input_height - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        image = image.astype(np.float32)
        image /= 255.0
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        
        return image, (r, (dw, dh))

    def postprocess(self, output, ratio, pad):
        # Decode YOLOv8 Output
        # output shape: (1, 56, 8400) or similar
        output = output[0].transpose() # (8400, 56)
        
        # Filter by confidence
        scores = np.max(output[:, 4:5], axis=1) # Objectness
        mask = scores > self.conf_threshold
        output = output[mask]
        
        if output.shape[0] == 0:
            return []
            
        # Parse boxes
        boxes = output[:, :4]
        # xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        # Scale boxes back to original image
        r, (dw, dh) = ratio
        boxes_xyxy[:, 0] -= dw
        boxes_xyxy[:, 1] -= dh
        boxes_xyxy[:, 2] -= dw
        boxes_xyxy[:, 3] -= dh
        boxes_xyxy /= r
        
        # Get keypoints
        # 5 onwards are keypoints: x, y, conf
        kpts_data = output[:, 5:]
        num_kpts = kpts_data.shape[1] // 3
        keypoints = []
        for i in range(output.shape[0]):
            kpt = kpts_data[i].reshape(num_kpts, 3)
            # Scale keypoints
            kpt[:, 0] = (kpt[:, 0] - dw) / r
            kpt[:, 1] = (kpt[:, 1] - dh) / r
            keypoints.append(kpt)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores[mask].tolist(), self.conf_threshold, self.iou_threshold)
        
        results = []
        for i in indices:
            idx = i # NMSBoxes returns indices directly or list? 
            # In new OpenCV it returns scalar indices
            results.append({
                'box': boxes_xyxy[idx],
                'keypoints': keypoints[idx]
            })
            
        return results

    def detect(self, frame):
        input_tensor, ratio = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return self.postprocess(outputs[0], ratio, (0,0)) # Pass correct pad if needed

    def draw_results(self, frame, results):
        for res in results:
            box = res['box'].astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw skeleton
            # Standard Hand Skeleton connections
            # 0->1->2->3->4 (Thumb)
            # 0->5->6->7->8 (Index)...
            connections = [
                (0,1), (1,2), (2,3), (3,4),
                (0,5), (5,6), (6,7), (7,8),
                (0,9), (9,10), (10,11), (11,12),
                (0,13), (13,14), (14,15), (15,16),
                (0,17), (17,18), (18,19), (19,20)
            ]
            
            kpts = res['keypoints']
            for i, (x, y, conf) in enumerate(kpts):
                if conf < 0.3: continue
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                
            for i, j in connections:
                 if i < len(kpts) and j < len(kpts):
                     x1, y1, c1 = kpts[i]
                     x2, y2, c2 = kpts[j]
                     if c1 > 0.3 and c2 > 0.3:
                         cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                         
        return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    tracker = YOLOHandTracker()
    cap = cv2.VideoCapture(args.camera)
    
    print("Starting... Press 'q' to quit.")
    
    while True:
        # 1. Window Close Check
        if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1:
            # Check if window was created first (it might be 0 initially)
            # Actually, simpler to just rely on waitKey
            pass

        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        results = tracker.detect(frame)
        tracker.draw_results(frame, results)
        
        cv2.imshow('Hand Tracking', frame)
        
        # 2. Robust Quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Also check if 'X' was clicked
        if cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE) < 1 and frame is not None:
             # This property is tricky, usually -1 if closed
             try:
                 visible = cv2.getWindowProperty('Hand Tracking', cv2.WND_PROP_VISIBLE)
                 if visible == 0: # Closed
                     break
             except: pass

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
