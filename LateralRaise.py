import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# videos/lateral raise/ dizinindeki tüm .mp4 dosyalarını listele
video_dir = "videos/lateral raise/"
video_files = [f for f in os.listdir(video_dir) if (f.endswith('.mp4') | f.endswith('.MOV'))]

# bel, omuz ve dirsek arasındaki çizgiyi çiz
hip_to_shoulder_to_elbow = [(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW)]

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    # get video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)

    # get video fps and calculate wait time
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps * 2)  # double the wait time to half the fps

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

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()