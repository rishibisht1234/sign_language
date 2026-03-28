import cv2
import mediapipe as mp
import numpy as np
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# def extract_landmarks(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image_rgb)

#     if results.multi_hand_landmarks:
#         hand_landmarks = results.multi_hand_landmarks[0]

#         coords = []
#         for lm in hand_landmarks.landmark:
#             coords.extend([lm.x, lm.y, lm.z])

#         return coords
#     else:
#         return None
    
    
    
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        coords = []
        for lm in results.multi_hand_landmarks[0].landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return coords, results
    
    return None, results   
    

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)

    # Use wrist as origin (landmark 0)
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale normalization
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks = landmarks / max_value

    return landmarks.flatten()