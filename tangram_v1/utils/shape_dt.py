import cv2
def assign_tangram_label(contour, _color):
    vertices = len(cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True))
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
