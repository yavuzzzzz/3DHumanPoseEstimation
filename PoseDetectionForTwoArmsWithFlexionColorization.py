# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture("videos/biceps curl/barbell biceps curl_40.mp4")

### get video resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.5)

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

            # save the list data to a file
        with open('twoarmslandmarks.txt', 'a') as f:
            for item in list:
                f.write("%s\n" % item)

        for indices in [(11, 13, 15), (12, 14, 16)]:
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

            # Calculate color based on angle_deg
            green = max(0, int((180 - angle_deg) / 180 * 255))
            red = max(0, int((angle_deg / 180) * 255))
            color = (0, red, green)  # Create color tuple

            ## R G B
            ## AÇI 0 İKEN 255 0 0
            ## AÇI 180 İKEN 0 255 0


            cv2.putText(img, str(angle_deg) + " deg", (list[indices[1]][0], list[indices[1]][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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

            # Draw landmarks with the new color
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 0), 2, 1),
                                   mp_draw.DrawingSpec(color, 1, 1))

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