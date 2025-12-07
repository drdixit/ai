import urllib.request
import os

def download_models():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # URL candidates based on search
    files = [
        # Candidate 1: Qualcomm HF (if names found)
        # Candidate 2: PINTO0309 (if names found)
        # Fallback: using a known specific commit or raw link from a working project
        ("models/palm_detection.onnx", "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/031_Hand_Detection/01_float32/palm_detection.onnx"),
        ("models/hand_landmark.onnx", "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/031_Hand_Detection/01_float32/hand_landmark.onnx")
    ]
    
    for filename, url in files:
        print(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
            size = os.path.getsize(filename)
            print(f"Done. Size: {size} bytes")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == '__main__':
    download_models()
