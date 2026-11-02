prev_hips = []

import cv2
import mediapipe as mp

from pose_utils import calculate_angle
from activity_detector import detect_activity

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    prev_hip_y = 0
    movement = 0

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            current_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y

            prev_hips.append(current_hip_y)

            # keep only last 10 frames
            if len(prev_hips) > 10:
                prev_hips.pop(0)

            # calculate movement
            hip_movement = max(prev_hips) - min(prev_hips)

            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)

            activity = detect_activity(angle, hip_movement)

            cv2.putText(image, f'Activity: {activity}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(image, f'Angle: {int(angle)}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        except:
            pass

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("Activity Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
