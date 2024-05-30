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
            angle_text = "Angle: {} degrees".format(angle_deg)

            # Calculate color based on angle_deg
            green = max(0, int((180 - angle_deg) / 180 * 255))
            red = max(0, int((angle_deg / 180) * 255))
            color = (0, red, green)

            percentage = round((angle_deg / 180) * 100)
            cv2.putText(img, str(100 - percentage) + "%", (list[indices[1]][0], list[indices[1]][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            center = (b[0], b[1])

            radius = 10
            mask = np.zeros_like(img)
            cv2.circle(mask, center, radius, (255, 255, 255), 5)

            arc_mask = np.zeros_like(img)
            start_angle = np.degrees(np.arctan2(ba[1], ba[0]))
            end_angle = np.degrees(np.arctan2(bc[1], bc[0]))

            if start_angle > end_angle:
                start_angle, end_angle = end_angle, start_angle

            if end_angle - start_angle > 180:
                start_angle, end_angle = end_angle, start_angle + 360

            cv2.ellipse(arc_mask, center, (radius, radius), 0, start_angle, end_angle, (255, 255, 255), 1)
            arc = cv2.bitwise_and(mask, arc_mask)
            img = cv2.bitwise_or(img, arc)

            mp_draw.draw_landmarks(img, results.pose_landmarks, hip_to_shoulder_to_elbow,
                                   mp_draw.DrawingSpec((0, 0, 0), 2, 1),
                                   mp_draw.DrawingSpec(color, 5, 1))

    cv2.imshow("Pose Estimation", img)

    h, w, c = img.shape
    opImg = np.zeros([h, w, c])
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, hip_to_shoulder_to_elbow,
                           mp_draw.DrawingSpec((255, 0, 0), 3, 3),
                           mp_draw.DrawingSpec((0, 255, 0), 10, 3))

    cv2.imshow("Extracted Pose", opImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()