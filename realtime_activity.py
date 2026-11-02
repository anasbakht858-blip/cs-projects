import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque

# Load model
model = joblib.load("activity_model.pkl")

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

prev_landmarks = None
motion_history = deque(maxlen=7)

# Smoothing buffer
buffer = deque(maxlen=7)

cap = cv2.VideoCapture(0)

label = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:

        features = []

        landmarks = results.pose_landmarks.landmark

        for lm in landmarks:
            features.extend([lm.x, lm.y, lm.z])

        # motion calculation
        motion = 0

        if prev_landmarks is not None:
            for i in range(len(landmarks)):
                dx = landmarks[i].x - prev_landmarks[i].x
                dy = landmarks[i].y - prev_landmarks[i].y
                dz = landmarks[i].z - prev_landmarks[i].z
                motion += abs(dx) + abs(dy) + abs(dz)

        motion_history.append(motion)
        avg_motion = sum(motion_history) / len(motion_history)

        prev_landmarks = landmarks

        # add motion feature (IMPORTANT)


        # Clean features
        features = np.nan_to_num(features)
        features = np.array(features).reshape(1, -1)

        # Prediction
        probs = model.predict_proba(features)[0]
        classes = model.classes_

        best_index = np.argmax(probs)
        confidence = probs[best_index]
        prediction = classes[best_index]

        # ⚡ NO aggressive Unknown filter (this was breaking boxing/running)
        buffer.append(prediction)

        # Smooth final label
        if len(buffer) == buffer.maxlen:
            label = max(set(buffer), key=buffer.count)
        else:
            label = prediction

        # Draw pose
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Display label
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        label = "No Person"

    cv2.imshow("Human Activity Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()