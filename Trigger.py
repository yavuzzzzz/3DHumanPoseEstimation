import os
import subprocess

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Tek bir video dosyası belirle
video_file = 'Lateraldemo.mp4'

cap = cv2.VideoCapture(video_file)

# Video çözünürlüğünü al
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Landmarks dosyası
landmarks_file = f'{os.path.splitext(video_file)[0]}.txt'

while True:
    ret, img = cap.read()
    if img is None:
        break

    results = pose.process(img)

    if results.pose_landmarks:
        landmark_data = []  # Her bir landmark için koordinatları saklamak için liste
        for lm in results.pose_landmarks.landmark:
            x = lm.x * width
            y = lm.y * height
            z = lm.z * height
            landmark_data.extend([x, y, z])  # x, y, z koordinatlarını listeye ekle

        # Landmarks dosyasına veriyi yaz
        with open(landmarks_file, 'a') as f:
            # Her bir landmark'ın x, y, z koordinatlarını bir satırda yaz
            f.write(" ".join(map(str, landmark_data)) + "\n")

cap.release()
