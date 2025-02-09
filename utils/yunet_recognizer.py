# By downloading, copying, installing or using the software you agree to this license.
# If you do not agree to this license, do not download, install,
# copy or use the software.


#                   License Agreement For libfacedetection
#                      (3-clause BSD License)

# Copyright (c) 2018-2020, Shiqi Yu, all rights reserved.
# shiqi.yu@gmail.com

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.

#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.

#   * Neither the names of the copyright holders nor the names of the contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.

# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.

import cv2
import numpy as np
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = sorted(os.listdir("../models/"), reverse=True)[0]
print(f"../models/{model_path}")
recognizer.read(f"../models/{model_path}")

# Load label map
dataset_path = sorted(os.listdir("../dataset/"), reverse=True)[0]
label_map = np.load(f"../dataset/{dataset_path}", allow_pickle=True).item()

# Load YuNet Face Detector
face_detector = cv2.FaceDetectorYN_create(
    "../models/face_detection_yunet_2023mar.onnx", "", (300, 300), score_threshold=0.5
)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    h, w, _ = frame.shape
    face_detector.setInputSize((w, h))

    # Detect faces
    _, faces = face_detector.detect(frame)
    
    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])

            # Ensure bounding box is within frame bounds
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue  # Skip invalid detections

            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]

            # Ensure the extracted face is valid
            if face_roi.size == 0:
                continue  # Skip empty detections

            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (100, 100))  # Match training size

            # Recognize face
            label_id, confidence = recognizer.predict(resized_face)

            if confidence > 130:  # Adjust threshold as needed
                name = label_map.get(label_id, "Unknown")
            else:
                name = "Unknown"

            # Draw bounding box & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()