import urllib.request
import os

def download_models():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Hugging Face Qualcomm MediaPipe Models
    base = "https://huggingface.co/qualcomm/MediaPipe-Hand-Detection/resolve/main"
    
    files = [
        ("models/palm_detection.onnx", f"{base}/palm_detection.onnx"),
        ("models/hand_landmark.onnx", f"{base}/hand_landmark.onnx")
    ]
    
    for filename, url in files:
        print(f"Downloading {filename} from {url}...")
        try:
            # mimic browser user agent to avoid some blocks
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            
            size = os.path.getsize(filename)
            print(f"Done. Size: {size/1024:.2f} KB")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == '__main__':
    download_models()
