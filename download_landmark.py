import urllib.request
import os

def download_models():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Correct URL based on search
    files = [
        ("models/hand_landmark.onnx", "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx"),
    ]
    
    for filename, url in files:
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Done.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == '__main__':
    download_models()
