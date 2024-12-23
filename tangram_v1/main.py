import platform
import cv2
from utils.timer import Timer
from utils.movements import detect_movement_rotation, log_movement
from utils.hand_dt import create_hand_mask
from utils.shape_dt import assign_tangram_label
from utils.logger import save_logs
from config_hue import COLOR_RANGES_GANNUB

def main():
    cap = cv2.VideoCapture(0)
    timer = Timer()
    prev_positions = {}
    movement_logs = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # create hand mask
            hand_mask, results = create_hand_mask(frame, frame_rgb)
            inverse_hand_mask = cv2.bitwise_not(hand_mask)

            for color_name, ranges in COLOR_RANGES_GANNUB.items():
                color_mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
                for lower, upper in ranges[1:]:
                    color_mask |= cv2.inRange(hsv, lower, upper)

                color_mask &= inverse_hand_mask
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv2.contourArea(cnt) < 1500:
                        continue

                    centroid, movement_vector, rotation_angle = detect_movement_rotation(
                        cnt, prev_positions.get(color_name)
                    )

                    if centroid:
                        prev_positions[color_name] = centroid
                        tangram_label = assign_tangram_label(cnt, color_name)
                        if (movement_vector):
                            log_movement(
                                movement_logs, tangram_label, color_name, centroid, movement_vector, rotation_angle, timer.elapsed_time()
                            )

                        cv2.circle(frame, centroid, 5, (255, 0, 0), -1)
                        cv2.putText(frame, tangram_label, (centroid[0], centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        time_text = f'Elasped Time : {timer.elapsed_time_hr()}'
                        cv2.putText(frame, time_text, (0,40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 2)

                        cv2.drawContours(frame, [cnt], -1, (255, 255, 255), 2)

            # Display frame
            cv2.imshow("Tangram", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Session ended. Saving logs...")
    finally:
        save_logs(movement_logs, "tangram_v1/logs/movement_logs.json")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
