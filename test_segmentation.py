import numpy as np
import cv2
from matplotlib import pyplot as plt

# Test K-means segmentation

img = cv2.imread('tangram.jpg')

assert img is not None, "file could not be read, check with os.path.exists()"

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
k = 7
attamps = 10

ret, label, center = cv2.kmeans(np.float32(rgb.reshape(-1, 3)), k, None, criteria, attamps, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((rgb.shape))

cv2.imshow('thresh', result_image)
cv2.waitKey(0)

# Result: 7 clusters of colors
# can't be used for real segmentation, but can be used for color quantization.