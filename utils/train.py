import cv2
import os
import numpy as np
import datetime

def load_dataset(dataset_path):
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in sorted(os.listdir(dataset_path)):  # Sort for consistency
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person_name  # Map ID to name
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
            if img is None:
                print(f"Warning: Failed to load {img_path}")
                continue

            img = cv2.resize(img, (100, 100))  # Resize to 100x100 for consistency

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    return np.array(faces, dtype=np.uint8), np.array(labels, dtype=np.int32), label_map

# Load dataset
dataset_path = "../images/"
faces, labels, label_map = load_dataset(dataset_path)

if len(faces) == 0:
    print("Error: No valid images found in dataset.")
    exit()

print(f"Loaded {len(faces)} images for training.")
print(f"Label map: {label_map}")

# Train LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Create directories if they don't exist
dataset_directory = "../dataset/"
os.makedirs(dataset_directory, exist_ok=True)

model_directory = "../models/"
os.makedirs(model_directory, exist_ok=True)

# Save the model and label map
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

recognizer.save(f"../models/face_recognizer_{timestamp}.yml")  # Save with timestamp

np.save(f"../dataset/label_map_{timestamp}.npy", label_map)  # Save with timestamp

print("Training complete!")