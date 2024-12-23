import cv2

def detect_movement_rotation(contour, prev_centroid):
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


def log_movement(movement_logs, shape, color, centroid, movement_vector, rotation_angle, timestamp):
    movement_logs.append({
        "shape": shape,
        "color": color,
        "centroid": centroid,
        "movement_vector": movement_vector,
        "rotation_angle": rotation_angle,
        "timestamp": timestamp
    })
