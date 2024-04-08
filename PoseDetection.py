import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture("C:/Users/yvzyl/OneDrive/Masaüstü/Videolar/barbell biceps curl_10.mp4")
#cap = cv2.VideoCapture(0) for real-time detection

while True:
    ret, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img, (640, 480))


    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # to show landmarks
    list=[]
    if not results.pose_landmarks:
        print("nothing")
    else:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x = int(lm.x * 640)
            y = int(lm.y * 480)
            cv2.circle(img, (x, y), 1, (255, 0, 255), -1)
            cv2.putText(img, str(id), (x, y - 1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            list.append([x,y])
        print(list[13], "dirsek")
        print(list[11], "omuz")
        print(list[15], "el bilegi")




    cv2.imshow("Pose Estimation", img)

    h, w, c = img.shape
    opImg = np.zeros([h, w, c]) #arka planı siyah bir pencere oluşturur
    #opImg.fill(255) #cıkarılan sayfanın arka planını beyaz yapar
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 3, 3),
                           mp_draw.DrawingSpec((0, 255, 0), 3, 3))

    cv2.imshow("Extracted Pose", opImg) #pozu arka planda gösterir

    #print(results.pose_landmarks) # koordinatları konsola yazdırma

    cv2.waitKey(1)

