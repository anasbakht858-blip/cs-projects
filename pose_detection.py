import cv2
import mediapipe as mp
from collections import deque
import time

from activity_detector import detect_activity

# ---------------- SETUP ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

activity_buffer = deque(maxlen=7)
jump_buffer = deque(maxlen=3)

prev_hip_y = None
prev_left_ankle_y = None
prev_right_ankle_y = None

stable_activity = "No Person"

prev_time = 0
confidence = 0.0  # IMPORTANT: initialize once

# ---------------- MAIN LOOP ----------------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        activity, prev_hip_y, prev_left_ankle_y, prev_right_ankle_y = detect_activity(
            landmarks,
            prev_hip_y,
            prev_left_ankle_y,
            prev_right_ankle_y
        )

        activity_buffer.append(activity)

        # jumping logic
        if activity == "Jumping":
            jump_buffer.append(1)
        else:
            jump_buffer.append(0)

        most_common = max(set(activity_buffer), key=activity_buffer.count)

        if sum(jump_buffer) >= 2:
            stable_activity = "Jumping"
        else:
            stable_activity = most_common

        # ---------------- CONFIDENCE (SMOOTHED) ----------------
        raw_confidence = activity_buffer.count(stable_activity) / len(activity_buffer)

        alpha = 0.3
        confidence = confidence + alpha * (raw_confidence - confidence)

        # draw pose
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    else:
        stable_activity = "No Person"
        confidence = 0.0

    # ---------------- COLOR THEME ----------------
    color = (0, 255, 0)

    if stable_activity == "Running":
        color = (0, 0, 255)
    elif stable_activity == "Walking":
        color = (255, 255, 0)
    elif stable_activity == "Boxing":
        color = (255, 0, 255)
    elif stable_activity == "Jumping":
        color = (0, 165, 255)
    elif stable_activity == "No Person":
        color = (128, 128, 128)

    # ---------------- FPS ----------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # =========================================================
    #                     UI DASHBOARD
    # =========================================================

    # ---------------- TOP HEADER ----------------
    cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 20), -1)

    cv2.putText(frame,
                "AI HUMAN ACTIVITY DETECTION SYSTEM",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (520, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2)

    # ---------------- LEFT PANEL ----------------
    cv2.rectangle(frame, (10, 70), (300, 220), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, 70), (300, 220), color, 2)

    cv2.putText(frame,
                "ACTIVITY",
                (25, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2)

    cv2.putText(frame,
                stable_activity,
                (25, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.putText(frame,
                "STATUS",
                (25, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1)

    # ---------------- BIG CONFIDENCE BAR ----------------
    bar_x, bar_y = 320, 180
    bar_width, bar_height = 300, 25

    # background
    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (40, 40, 40), -1)

    # fill
    fill_width = int(bar_width * confidence)

    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + fill_width, bar_y + bar_height),
                  color, -1)

    # border
    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 1)

    cv2.putText(frame,
                f"CONFIDENCE {int(confidence * 100)}%",
                (320, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2)

    # ---------------- FINAL DISPLAY ----------------
    cv2.imshow("Activity Detection System", frame)

    key = cv2.waitKey(5) & 0xFF

    if key == 27 or key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()