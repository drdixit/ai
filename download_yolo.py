import urllib.request
import os

def download_yolo():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # RionDsilvaCS YOLO hand pose model
    # yolov8n-pose is usually around 6MB, so raw link should work.
    
    urls = [
        ("models/hand_pose.pt", "https://github.com/RionDsilvaCS/yolo-hand-pose/raw/main/model/best.pt"),
    ]
    
    for filename, url in urls:
        print(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
            
            s = os.path.getsize(filename)
            print(f"Done. Size: {s/1024/1024:.2f} MB")
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == '__main__':
    download_yolo()
