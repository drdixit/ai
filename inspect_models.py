import onnxruntime as ort

def inspect_model(path):
    print(f"--- Inspecting {path} ---")
    try:
        session = ort.InferenceSession(path)
        print("Inputs:")
        for i in session.get_inputs():
            print(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
        print("Outputs:")
        for o in session.get_outputs():
            print(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    inspect_model('models/palm_detection.onnx')
    inspect_model('models/hand_landmark.onnx')
