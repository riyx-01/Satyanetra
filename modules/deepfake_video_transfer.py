import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from modules.deepfake_efficientnet_model import EfficientNetDeepFake

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and the newly saved weights
model = EfficientNetDeepFake()
model.load_state_dict(torch.load("C:/Users/riyat/OneDrive/Desktop/Satyanetra/deepfake_detector_demo.pth", map_location=device))
model.eval().to(device)

# Frame transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    preds = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            try:
                input_tensor = transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    score = torch.sigmoid(model(input_tensor)).item()
                preds.append(score)
            except Exception as e:
                print(f"Frame error: {e}")
        frame_count += 1

    cap.release()

    if not preds:
        return "Unknown", 0.0

    avg = np.mean(preds)

    if avg > 0.5:
        label = "Real"
        confidence = avg * 100
    else:
        label = "Fake"
        confidence = (1 - avg) * 100

    confidence = round(confidence, 2)  # Keep two decimal places
    return label, confidence


