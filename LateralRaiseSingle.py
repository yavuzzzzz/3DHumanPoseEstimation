import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# bel, omuz ve dirsek arasındaki çizgiyi çiz
hip_to_shoulder_to_elbow = [(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW)]

# Tek bir video dosyası belirle
video_file = 'Lateraldemo.mp4'

cap = cv2.VideoCapture(video_file)

### get video resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)

while True:
    ret, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img, (width, height))

    results = pose.process(img)

    list = []
    if not results.pose_landmarks:
        print("nothing")
    else:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x = int(lm.x * width)
            y = int(lm.y * height)
            z = int(lm.z * height)
            cv2.circle(img, (x, y), 1, (255, 0, 255), -1)
            list.append([x, y, z])

        joint_pairs = hip_to_shoulder_to_elbow
        for indices in joint_pairs:
            a = np.array([list[indices[0]][0], list[indices[0]][1], list[indices[0]][2]])
            b = np.array([list[indices[1]][0], list[indices[1]][1], list[indices[1]][2]])
            c = np.array([list[indices[2]][0], list[indices[2]][1], list[indices[2]][2]])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            angle_deg = round(np.degrees(angle))

            # Calculate color based on angle_deg
            green = max(0, int((angle_deg / 90) * 255))
            red = max(0, int((1 - angle_deg / 90) * 255))
            color = (0, red, green)

            # Draw colored circle at shoulder joint
            cv2.circle(img, (b[0], b[1]), 10, color, -1)

            # Calculate percentage based on angle_deg
            percentage = min(100, round((angle_deg / 90) * 100))
            percentage_text = "{}%".format(percentage)

            # Put percentage text on shoulder joint
            cv2.putText(img, percentage_text, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Pose Estimation", img)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()