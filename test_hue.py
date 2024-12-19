import cv2
import numpy as np


def analyze_hsv_ranges(frame):
    def nothing(x):
        pass

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H Lower", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("S Lower", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V Lower", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("H Upper", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S Upper", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V Upper", "Trackbars", 255, 255, nothing)

    while True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_lower = cv2.getTrackbarPos("H Lower", "Trackbars")
        s_lower = cv2.getTrackbarPos("S Lower", "Trackbars")
        v_lower = cv2.getTrackbarPos("V Lower", "Trackbars")
        h_upper = cv2.getTrackbarPos("H Upper", "Trackbars")
        s_upper = cv2.getTrackbarPos("S Upper", "Trackbars")
        v_upper = cv2.getTrackbarPos("V Upper", "Trackbars")

        lower_bound = np.array([h_lower, s_lower, v_lower])
        upper_bound = np.array([h_upper, s_upper, v_upper])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to capture a frame.")
        return

    analyze_hsv_ranges(frame1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
