import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict
import platform

# TANGRAM_LABELS = ["Small Triangle 1", "Small Triangle 2", "Medium Triangle",
#                   "Large Triangle 1", "Large Triangle 2", "Square", "Parallelogram"]


def assign_tangram_label(contour: np.ndarray, approx: np.ndarray, _color: str) -> str:
    vertices = len(approx)
    # we dont need to use vertices ?
    shape = "Unknown"
    if _color == "blue":
        shape = "Piece 1"
    elif _color == "green":
        shape = "Piece 2"
    elif _color == "red":
        shape = "Piece 3"
    elif _color == "yellow":
        shape = "Piece 4"
    return shape


COLOR_RANGES = {  # based on my lighting
    "red": [(np.array([0, 192, 100]), np.array([5, 255, 255])),
            (np.array([170, 192, 100]), np.array([180, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "green": [(np.array([40, 100, 100]), np.array([90, 255, 255]))],
    "blue": [(np.array([90, 150, 100]), np.array([140, 255, 255]))],
}

# mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# centroid, mvt, rotation


def detect_movement_rotation(contour: np.ndarray, prev_centroid: Optional[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[float]]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None, None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    current_centroid = (cx, cy)

    movement_vector = None
    if prev_centroid:
        movement_vector = (cx - prev_centroid[0], cy - prev_centroid[1])

    rect = cv2.minAreaRect(contour)
    rotation_angle = rect[-1]

    return current_centroid, movement_vector, rotation_angle


def main():
    cap = ""
    # camera device for each OS
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(0)
    
    prev_pos = {}
    try:
        while True:
            ret, frame = cap.read()
            mask_frame = np.array(frame)
            mask_frame[:] = 255
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Mediapipe: Input image must contain three channel rgb data.
            # can't use grayscale image.
            results = hands.process(frame_rgb)
            hand_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # create hand mask
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = frame.shape
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                    cv2.rectangle(hand_mask, (x_min, y_min),
                                (x_max, y_max), 255, -1)

            inverse_hand_mask = cv2.bitwise_not(hand_mask)
            for color_name, ranges in COLOR_RANGES.items():
                color_mask = np.zeros_like(hand_mask)
                for lower, upper in ranges:
                    mask = cv2.inRange(hsv, lower, upper)
                    color_mask = cv2.bitwise_or(color_mask, mask)

                color_mask = cv2.bitwise_and(
                    color_mask, color_mask, mask=inverse_hand_mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
                color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(
                    color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 500:
                        continue

                    epsilon = 0.02 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    centroid, movement_vector, rotation_angle = detect_movement_rotation(
                        cnt, prev_pos.get(color_name)
                    )

                    if centroid:
                        prev_pos[color_name] = centroid
                        cv2.circle(frame, centroid, 5, (255, 0, 0), -1)

                    tangram_label = assign_tangram_label(
                        cnt, approx, _color=color_name)
                    cv2.putText(frame, tangram_label, (centroid[0], centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 2555, 255), 2)

                    cv2.drawContours(frame, [approx], -1, (255, 255, 255), 2)
                    cv2.drawContours(mask_frame, [approx], -1, color=(0, 0, 0), thickness=cv2.FILLED)
                    
                    print(prev_pos)

            # output
            # contours1 = contours1[0].reshape(-1,2)
            cv2.imshow("Tangram", frame)
            cv2.imshow("Mask", mask_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise InterruptedError
    except:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()