import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

classifier_model = load_model('hand_gesture_landmark_model.h5')

class_names = [
    '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
    '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

pred_queue = deque(maxlen=5)  # average over last 5 predictions
frame_counter = 0  # optional frame skip counter

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)
    gesture_label = "No Gesture Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_list = []
            base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
            for landmark in hand_landmarks.landmark:
                landmarks_list.append(landmark.x - base_x)
                landmarks_list.append(landmark.y - base_y)
            input_data = np.array(landmarks_list).reshape(1, -1)
            prediction = classifier_model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)

            pred_queue.append(class_id)
            gesture_label = class_names[max(set(pred_queue), key=pred_queue.count)]
    cv2.putText(
        image,
        gesture_label,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow('Hand Gesture Recognition', image)

    # Exit on 'Esc' or 'q'
    if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
