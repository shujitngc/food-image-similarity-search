import os
import pickle
import numpy as np
from PIL import Image

import torch
import open_clip


image_dir = "images"
feature_path = "features.npy"
paths_path = "image_paths.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
model.eval()

image_paths = [
    os.path.join(image_dir, name)
    for name in os.listdir(image_dir)
    if name.lower().endswith((".jpg", ".jpeg", ".png"))
]

features = []

with torch.no_grad():
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        feature = model.encode_image(image_tensor)
        feature = feature / feature.norm(dim=-1, keepdim=True)

        features.append(feature.cpu().numpy()[0])

features = np.array(features)

np.save(feature_path, features)

with open(paths_path, "wb") as f:
    pickle.dump(image_paths, f)

print("特徴量作成完了")
print("画像枚数:", len(image_paths))
print("特徴量の形:", features.shape)