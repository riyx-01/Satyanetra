import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn
import cv2
import numpy as np
import joblib

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = resnet50(pretrained=True)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 10 == 0:
            try:
                input_tensor = transform(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = resnet_model(input_tensor).squeeze().cpu()
                features.append(feature)
            except:
                pass
        count += 1
    cap.release()
    if not features:
        return None
    return torch.stack(features).mean(dim=0).numpy()

# Extract features from labeled videos
def process_dataset(data_dir):
    X, y = [], []
    for label in ['real', 'fake']:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            if filename.endswith(".mp4"):
                print(f"Processing {filename}...")
                path = os.path.join(folder, filename)
                feature = extract_features_from_video(path)
                if feature is not None:
                    X.append(feature)
                    y.append(0 if label == 'real' else 1)
    return np.array(X), np.array(y)

# Run and save
X, y = process_dataset("dataset")
joblib.dump((X, y), "video_features.pkl")
print("Saved features to video_features.pkl")
