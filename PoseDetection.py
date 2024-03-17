import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture("C:/Users/yvzyl/OneDrive/Masaüstü/Videolar/lateral raise_1.mov")
#cap = cv2.VideoCapture(0) for real-time detection

while True:
    ret, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img, (600, 400))

    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 255), 3, 3))  # to show landmarks

    cv2.imshow("Pose Estimation", img)

    h, w, c = img.shape
    opImg = np.zeros([h, w, c]) #arka planı siyah bir pencere oluşturur
    opImg.fill(255) #cıkarılan sayfanın arka planını beyaz yapar
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 3, 3),
                           mp_draw.DrawingSpec((0, 255, 0), 3, 3))

    cv2.imshow("Extracted Pose", opImg) #pozu arka planda gösterir

    print(results.pose_landmarks) # koordinatları konsola yazdırma

    cv2.waitKey(1)
