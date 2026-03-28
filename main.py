import cv2
import mediapipe as mp
import numpy as np
import os
import joblib


from fun import normalize_landmarks, extract_landmarks


model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmarks, results = extract_landmarks(frame)
    
    if landmarks is not None:
        features = normalize_landmarks(landmarks)
        
        # IMPORTANT: match training pipeline
        # features = scaler.transform([features])  # uncomment if used
        
        features = np.array(features).reshape(1, -1)
        
        pred = model.predict(features)[0]
        label = le.inverse_transform([pred])[0]
        
        cv2.putText(frame, f"{label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        
        mp_drawing.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )
    else:
        cv2.putText(frame, "No Hand", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
    
    cv2.imshow("ASL Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()