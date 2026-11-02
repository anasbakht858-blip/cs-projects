import cv2
import mediapipe as mp
import pandas as pd
import os

DATASET_PATH = r"C:\Users\HP\Downloads\archive"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

data = []

print("Dataset path:", DATASET_PATH)

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:

        if not file.endswith(".avi"):
            continue

        video_path = os.path.join(root, file)
        print("Processing:", video_path)

        filename = file.lower()

        #  FIXED LABEL MAPPING (no merging confusion)
        if "walking" in filename:
            label = "walking"
        elif "jogging" in filename:
            label = "jogging"
        elif "running" in filename:
            label = "running"
        elif "boxing" in filename:
            label = "boxing"
        elif "handclapping" in filename:
            label = "handclapping"
        elif "handwaving" in filename:
            label = "handwaving"
        else:
            continue

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:

                #  FIX: ensure full landmark set always 33
                if len(results.pose_landmarks.landmark) != 33:
                    continue

                row = []

                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)
                data.append(row)

        cap.release()

df = pd.DataFrame(data)

#  FIX: proper column naming (VERY IMPORTANT)
cols = [f"x{i}" for i in range(33)] + \
       [f"y{i}" for i in range(33)] + \
       [f"z{i}" for i in range(33)] + \
       ["label"]

df.columns = cols

df.to_csv("activity_dataset.csv", index=False)

print("Dataset saved!")
print("Total samples:", len(df))