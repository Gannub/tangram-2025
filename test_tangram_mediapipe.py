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
    if vertices == 4:
        if _color == "yellow":
            shape = "Square"
        elif _color == "purple":
            shape = "Parallelogram"
    elif vertices == 3:
        if _color == "blue":
            shape = "Large Triangle 1"
        elif _color == "red":
            shape = "Large Triangle 2"
        elif _color == "green":
            shape = "Medium Triangle"
        elif _color == "pink":
            shape = "Small Triangle 1"
        elif _color == "orange":
            shape = "Small Triangle 2"
    return shape

COLOR_RANGES = {  # based on gannub's lighting
    "red": [(np.array([0, 130, 110]), np.array([5, 255, 255])),
            (np.array([170, 130, 110]), np.array([180, 255, 255]))],
    "orange": [(np.array([5, 100, 110]), np.array([15, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "green": [(np.array([40, 50, 60]), np.array([90, 255, 255]))],
    "blue": [(np.array([90, 150, 60]), np.array([140, 255, 255]))],
    "purple": [(np.array([110, 50, 70]), np.array([150, 255, 255]))],
    "pink": [(np.array([0, 30, 70]), np.array([5, 140, 255])),
             (np.array([170, 30, 70]), np.array([180, 140, 255]))],
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


def crop_picture(picture: np.array):
    """
    Crop picture.

    Detect bounding box then crop picture. Only work if it's axis-aligned and gray scale
    Return ([size_x, size_y], picture)
    """
    img_edge = cv2.Canny(picture, 30, 200) 
    img_loc = np.where(img_edge != 0) 
    img_arr = np.array([[img_loc[1][i],img_loc[0][i]] for i in range(len(img_loc[0]))])
    y, x, sy, sx = cv2.boundingRect(img_arr)
    return [sx, sy], picture[x:x+sx,y:y+sy]

debug = False

def main():
    cap = ""
    # camera device for each OS
    if not debug:
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(1)
        else:
            cap = cv2.VideoCapture(0)
    
    prev_pos = {}
    cntLoop = 0

    try:
        while True:
            ret = True
            cntLoop = cntLoop+1
            if debug:
                frame = cv2.imread("../data/img2170.jpg")
                # frame = cv2.imread("../data/img{}.jpg".format((cntLoop//1) * 5))
            else:
                ret, frame = cap.read()
            frame = cv2.resize(frame, (640*3, 480*3))
            frame = cv2.GaussianBlur(frame,(3,3),1)
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

                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    centroid, movement_vector, rotation_angle = detect_movement_rotation(
                        cnt, prev_pos.get(color_name)
                    )
                    if len(approx) > 10:  # Example threshold for smoothness
                        continue
                    if centroid:
                        prev_pos[color_name] = centroid
                        cv2.circle(frame, centroid, 5, (255, 0, 0), -1)

                    tangram_label = assign_tangram_label(
                        cnt, approx, _color=color_name)
                    cv2.putText(frame, tangram_label, (centroid[0], centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.drawContours(frame, [approx], -1, (255, 255, 255), 2)
                    cv2.drawContours(mask_frame, [approx], -1, color=(0, 0, 0), thickness=cv2.FILLED)
                    
            
            erosion_mask_frame = cv2.erode(mask_frame,np.ones((11,11)))
            denoised_mask_frame = cv2.dilate(erosion_mask_frame,np.ones((11,11)))
            
            denoised_mask_frame = cv2.cvtColor(denoised_mask_frame, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(denoised_mask_frame, 30, 200) 
            
            loc = np.where(edge != 0) 
            
            arr = np.array([])
            if len(loc[0]) > 0:
                arr = np.array([[loc[1][i],loc[0][i]] for i in range(len(loc[0]))])
                
                bounding_box = cv2.minAreaRect(arr)
                # print(bounding_box)
                
                
                box_points = cv2.boxPoints(bounding_box)
                box_points = np.expand_dims(box_points, axis=1)
                
                img_template = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
                template_sizes, img_template = crop_picture(img_template)
                rot = cv2.getRotationMatrix2D(bounding_box[0], bounding_box[2], 1)
                denoised_mask_frame = cv2.warpAffine(src = denoised_mask_frame, M = rot,
                                                     dsize = (denoised_mask_frame.shape[1], denoised_mask_frame.shape[0]),
                                                     flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT,
                                                     borderValue = 255)
                
                # TODO
                # Check 4 rotation 
                # rot_index = 0
                # for angle in range(0,360,90):
                #     sizes, use_img_template = crop_picture(img_template)
                #     rot_index = (rot_index + 1) % 2
                    
                # use_img_template = cv2.rotate(img_template,cv2.ROTATE_90_CLOCKWISE)
                use_img_template = img_template
                # scale = cv2.getRotationMatrix2D(bounding_box[0], 0, template_sizes[rot_index] / bounding_box[1][0])
                scale = cv2.getRotationMatrix2D(bounding_box[0], 0, template_sizes[0] / bounding_box[1][0])
                use_denoised_mask_frame = cv2.warpAffine(src = denoised_mask_frame, M = scale,
                                                            dsize = (denoised_mask_frame.shape[1], denoised_mask_frame.shape[0]),
                                                            flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT,
                                                            borderValue = 255)
                
                cv2.imshow("Use Mask", use_denoised_mask_frame)
                
                sim_frame = cv2.matchTemplate(use_denoised_mask_frame, use_img_template, cv2.TM_CCORR_NORMED)
                # sim_frame /= np.max(sim_frame)
                # sim_frame *= 255.0
                # sim_frame = sim_frame.astype('uint8')
                
                sim = np.where(sim_frame == np.max(sim_frame))
                print(np.min(sim_frame),np.max(sim_frame))
                yy, xx = sim[0][0] ,sim[1][0]
                # TODO
                # Detect max possible matching
                
                cv2.imshow("Diff", sim_frame)
                    
            
            cv2.imshow("Tangram", frame)
            cv2.imshow("Mask", denoised_mask_frame)
            # cv2.imshow("Use Image", img_template)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise InterruptedError
    except Exception as e:
        print(e)
        if not debug:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()