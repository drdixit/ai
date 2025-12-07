# Hand Tracking Application

A Python-based hand tracking application with **AMD GPU acceleration** using DirectML (DirectX Machine Learning) for your AMD RX 5500M GPU.

## Features

- ✋ **Real-time hand tracking** using MediaPipe
- 🎮 **AMD GPU acceleration** via DirectML (DirectX 12)
- 🤟 **Gesture recognition** (fist, open hand, peace sign, pointing, etc.)
- 📊 **Hand velocity tracking**
- 🎯 **Multi-hand support** (up to 2 hands)
- 📈 **FPS counter and statistics**
- ⚙️ **Configurable parameters**

## Hardware Acceleration

This application uses **DirectML** (DirectX Machine Learning) which provides excellent support for AMD GPUs on Windows. DirectML is built on top of DirectX 12 and is optimized for AMD hardware.

### Why DirectML?

- ✅ Native Windows support
- ✅ Excellent AMD GPU compatibility
- ✅ Easy setup (no complex driver configuration)
- ✅ Automatic hardware detection
- ✅ Better than Vulkan for Windows + AMD combination

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 with DirectX 12 support
- AMD RX 5500M GPU with latest drivers
- Webcam

### Setup

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` - Camera access and image processing
- `mediapipe` - Hand tracking and landmark detection
- `numpy` - Numerical operations
- `onnxruntime-directml` - DirectML acceleration for AMD GPUs

## Usage

### Basic Usage

Run with GPU acceleration (default):

```bash
python hand_tracker.py
```

### Command Line Options

```bash
# Run with CPU only (disable GPU)
python hand_tracker.py --no-gpu

# Use different camera
python hand_tracker.py --camera 1

# Custom resolution
python hand_tracker.py --width 1920 --height 1080

# Adjust detection sensitivity
python hand_tracker.py --detection-confidence 0.8 --tracking-confidence 0.6

# Combine options
python hand_tracker.py --camera 0 --width 1280 --height 720
```

### Controls

- **Q** - Quit the application
- **S** - Toggle statistics display

## Recognized Gestures

The application can recognize the following gestures:

- 🤜 **Fist** - All fingers closed
- ✋ **Open Hand** - All fingers extended
- ✌️ **Peace Sign** - Index and middle fingers extended
- 👉 **Pointing** - Only index finger extended
- 🤙 **Shaka** - Thumb and pinky extended
- 🔫 **Gun** - Thumb and index finger extended
- 🖐️ **N Fingers** - Any other combination

## Performance

With AMD RX 5500M GPU acceleration:
- Expected FPS: 30-60 FPS (depending on resolution)
- Latency: < 50ms
- CPU usage: Low (most processing on GPU)

Without GPU (CPU only):
- Expected FPS: 15-30 FPS
- Higher CPU usage

## Troubleshooting

### GPU not being used

1. Ensure you have the latest AMD drivers installed
2. Verify DirectX 12 is available: `dxdiag` in Windows
3. Check that `onnxruntime-directml` is installed: `pip list | findstr onnxruntime`

### Low FPS

1. Reduce camera resolution: `--width 640 --height 480`
2. Adjust confidence thresholds (lower = faster but less accurate)
3. Close other GPU-intensive applications

### Camera not found

1. Check camera index: `python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"`
2. Use the correct index with `--camera N`

## Technical Details

### Architecture

```
Camera Input → OpenCV → MediaPipe (DirectML) → Landmark Detection → Gesture Recognition → Display
```

### DirectML vs Vulkan

For your AMD RX 5500M on Windows, **DirectML is the better choice**:

| Feature | DirectML | Vulkan |
|---------|----------|--------|
| Windows Support | ✅ Excellent | ⚠️ Good |
| AMD GPU Support | ✅ Optimized | ✅ Good |
| Setup Complexity | ✅ Easy | ❌ Complex |
| Performance | ✅ Excellent | ✅ Excellent |
| Driver Maturity | ✅ Mature | ⚠️ Varies |

### Giving Users Choice

If you want to give users the option to choose between DirectML and CPU:
- Use the `--no-gpu` flag for CPU-only mode
- DirectML is automatically used when GPU is enabled
- No need to expose Vulkan option (adds complexity without benefits on Windows)

## Future Enhancements

Potential additions:
- [ ] Save gesture recordings
- [ ] Custom gesture training
- [ ] Mouse control via hand gestures
- [ ] Virtual keyboard
- [ ] Hand pose estimation
- [ ] Multi-camera support

## License

MIT License - Feel free to modify and use as needed!

## Credits

- **MediaPipe** by Google for hand tracking
- **OpenCV** for camera and image processing
- **ONNX Runtime** with DirectML for AMD GPU acceleration
