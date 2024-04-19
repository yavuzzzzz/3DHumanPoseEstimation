# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture("videos/lateral raise/Dumbbell_Lateral_Raise.mp4")

### get video resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, img = cap.read()
    if img is None:
        break
    img = cv2.resize(img, (width, height))

    results = pose.process(img)
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
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

        a = np.array([list[11][0], list[11][1], list[11][2]])
        b = np.array([list[13][0], list[13][1], list[13][2]])
        c = np.array([list[15][0], list[15][1], list[15][2]])
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        angle_deg = round(np.degrees(angle))
        angle_text = "Angle: {} degrees".format(angle_deg)
        cv2.putText(img, str(angle_deg) + " deg", (list[13][0], list[13][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 2)

        # Calculate the center for the circle that will be drawn
        center = (b[0], b[1])

        # Set the radius to a constant value
        radius = 10

        # Create a black image with the same size as the original image
        mask = np.zeros_like(img)

        # Draw a circle on the mask image
        cv2.circle(mask, center, radius, (255, 255, 255), 5)

        # Create a mask for the arc
        arc_mask = np.zeros_like(img)

        # Calculate the start and end angles for the arc
        start_angle = np.degrees(np.arctan2(ba[1], ba[0]))
        end_angle = np.degrees(np.arctan2(bc[1], bc[0]))

        # Ensure the start angle is always less than the end angle
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle

        # If the angle is greater than 180 degrees, draw the arc on the other side
        if end_angle - start_angle > 180:
            start_angle, end_angle = end_angle, start_angle + 360

        # Draw the arc on the mask
        cv2.ellipse(arc_mask, center, (radius, radius), 0, start_angle, end_angle, (255, 255, 255), 1)

        # Bitwise-AND the mask and the circle to get the arc
        arc = cv2.bitwise_and(mask, arc_mask)

        # Bitwise-OR the arc with the original image
        img = cv2.bitwise_or(img, arc)

    cv2.imshow("Pose Estimation", img)

    h, w, c = img.shape
    opImg = np.zeros([h, w, c])
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 3, 3),
                           mp_draw.DrawingSpec((0, 255, 0), 3, 3))

    cv2.imshow("Extracted Pose", opImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()