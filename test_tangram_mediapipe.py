import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict

# TANGRAM_LABELS = ["Small Triangle 1", "Small Triangle 2", "Medium Triangle",
#                   "Large Triangle 1", "Large Triangle 2", "Square", "Parallelogram"]


def assign_tangram_label(contour: np.ndarray, approx: np.ndarray, _color: str) -> str:
    vertices = len(approx)
    # we dont need to use vertices ?
    if vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h if h > 0 else 0
        if 0.9 < aspect_ratio < 1.1:
            if _color == "yellow":
                return "Square"
        else:
            if _color == "purple":
                return "Parallelogram"

    elif vertices == 3:
        if _color == "blue":
            return "Large Triangle 1"
        elif _color == "red":
            return "Large Triangle 2"
        elif _color == "green":
            return "Medium Triangle"
        elif _color == "pink":
            return "Small Triangle 1"
        elif _color == "orange":
            return "Small Triangle 2"

    return "Unknown"


COLOR_RANGES = {  # based on my lighting
    "red": [(np.array([0, 130, 110]), np.array([5, 255, 255])),
            (np.array([170, 130, 110]), np.array([180, 255, 255]))],
    "orange": [(np.array([7, 100, 110]), np.array([15, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "green": [(np.array([40, 50, 60]), np.array([90, 255, 255]))],
    "blue": [(np.array([90, 150, 60]), np.array([140, 255, 255]))],
    "purple": [(np.array([110, 50, 70]), np.array([150, 255, 255]))],
    "pink": [(np.array([0, 50, 70]), np.array([5, 140, 255])),
             (np.array([170, 50, 70]), np.array([180, 140, 255]))],
}

# mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.5)
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
    # camera device for Tonsak
    # cap = cv2.VideoCapture(1)
    # camera device for mac/Linux user
    cap = cv2.VideoCapture(0)
    
    prev_pos = {}
    try:
        while True:
            ret, frame = cap.read()
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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.drawContours(frame, [approx], -1, (255, 255, 255), 2)
                    print(prev_pos)

            # output
            cv2.imshow("Tangram", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise InterruptedError
    except:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()