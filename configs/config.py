import numpy as np

TANGRAM_PIECES = {
    "Small Triangle 1": {"vertices": 3, "aspect_ratio": (0.5, 1.5)},
    "Small Triangle 2": {"vertices": 3, "aspect_ratio": (0.5, 1.5)},
    "Medium Triangle": {"vertices": 3, "aspect_ratio": (1.0, 2.0)},
    "Large Triangle 1": {"vertices": 3, "aspect_ratio": (1.5, 3.0)},
    "Large Triangle 2": {"vertices": 3, "aspect_ratio": (1.5, 3.0)},
    "Square": {"vertices": 4, "aspect_ratio": (0.9, 1.1)},
    "Parallelogram": {"vertices": 4, "aspect_ratio": (1.5, 3.0)},
}

COLOR_RANGES = {  # based on my lighting
    "red": [(np.array([0, 192, 100]), np.array([5, 255, 255])),
            (np.array([170, 192, 100]), np.array([180, 255, 255]))],
    "orange": [(np.array([7, 100, 100]), np.array([15, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "green": [(np.array([40, 100, 100]), np.array([90, 255, 255]))],
    "blue": [(np.array([90, 150, 100]), np.array([140, 255, 255]))],
    "purple": [(np.array([110, 64, 127]), np.array([150, 255, 255]))],
    "pink": [(np.array([0, 40, 127]), np.array([7, 140, 255])),
             (np.array([155, 40, 127]), np.array([180, 140, 255]))],
}
