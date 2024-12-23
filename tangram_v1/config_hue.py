import numpy as np

COLOR_RANGES_GANNUB = {  # based on gannub's lighting
    "red": [(np.array([0, 130, 110]), np.array([2, 255, 255])),
            (np.array([170, 130, 110]), np.array([180, 255, 255]))],
    "orange": [(np.array([5, 100, 110]), np.array([15, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    "green": [(np.array([40, 50, 60]), np.array([90, 255, 255]))],
    "blue": [(np.array([90, 150, 60]), np.array([140, 255, 255]))],
    "purple": [(np.array([110, 50, 70]), np.array([150, 255, 255]))],
    "pink": [(np.array([0, 50, 70]), np.array([5, 140, 255])),
             (np.array([170, 50, 70]), np.array([180, 140, 255]))],
}

