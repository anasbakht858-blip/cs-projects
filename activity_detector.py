import mediapipe as mp
from pose_utils import calculate_angle

mp_pose = mp.solutions.pose


def detect_activity(
        landmarks,
        prev_hip_y,
        prev_left_ankle_y,
        prev_right_ankle_y):

    # LEFT SIDE
    left_hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    ]

    left_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    ]

    left_ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    ]

    # RIGHT SIDE
    right_hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    ]

    right_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    ]

    right_ankle = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    ]

    # ANGLES
    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)

    avg_angle = (left_angle + right_angle) / 2

    # CURRENT POSITIONS
    hip_y = (left_hip[1] + right_hip[1]) / 2

    left_ankle_y = left_ankle[1]
    right_ankle_y = right_ankle[1]

    activity = "Unknown"

    # MOVEMENT VALUES
    hip_movement = 0
    ankle_movement = 0

    if prev_hip_y is not None:
        hip_movement = abs(hip_y - prev_hip_y)

    if prev_left_ankle_y is not None and prev_right_ankle_y is not None:
        ankle_movement = (
                                 abs(left_ankle_y - prev_left_ankle_y) +
                                 abs(right_ankle_y - prev_right_ankle_y)
                         ) / 2

        hip_movement *= 0.7
        ankle_movement *= 1.3
        # ACTIVITY LOGIC

        if avg_angle < 120:
            activity = "Sitting"

        else:

            # 🟣 JUMPING (strong vertical hip movement, no ankle rhythm)
            if hip_movement > 0.055 and ankle_movement < 0.02:
                activity = "Jumping"

            # 🔴 RUNNING (rhythmic ankle movement)
            elif ankle_movement > 0.030:
                activity = "Running"

            # 🟠 WALKING (low ankle rhythm)
            elif ankle_movement > 0.012:
                activity = "Walking"

            # 🟢 STANDING
            else:
                activity = "Standing"

        total_motion = hip_movement + ankle_movement

        print("Hip:", hip_movement,
              "Ankle:", ankle_movement,
              "Total:", total_motion)

        print(f"Hip:{hip_movement:.4f} Ankle:{ankle_movement:.4f}")

    return (
        activity,
        hip_y,
        left_ankle_y,
        right_ankle_y
    )