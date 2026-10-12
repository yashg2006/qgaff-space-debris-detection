from ultralytics import YOLO
import torch
import numpy as np
import os

model = YOLO("runs/detect/train3/weights/best.pt")
model.model.eval()

features = []
labels = []

def hook_fn(module, input, output):
    features.append(output.detach().cpu().numpy())


hook = model.model.model[8].register_forward_hook(hook_fn)

img_folder = "valid/images"
for img in os.listdir(img_folder):
    path = os.path.join(img_folder, img)
    results = model.predict(source=path, verbose=False)
    cls = results[0].boxes.cls.cpu().numpy()
    labels.append(int(cls[0]) if len(cls)>0 else -1)

hook.remove()

features = np.vstack([f.mean(axis=(2,3)) for f in features])  # Global avg pool
np.save("features_before.npy", features)
np.save("labels.npy", labels)
