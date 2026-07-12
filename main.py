import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import os
import numpy as np
import time

model_path = os.path.join(os.path.dirname(__file__), "models/hand_landmarker.task")

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)


def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers = []
    for tip, pip in zip(tips, pips):
        fingers.append(1 if hand_landmarks[tip].y < hand_landmarks[pip].y else 0)
    return fingers


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Canvas", 1280, 720)

ret, first_frame = cap.read()
h, w, c = first_frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
smooth_x, smooth_y = None, None
smoothing_factor = 0.6

is_drawing = False
stop_counter = 0

save_counter = 0
save_message_timer = 0

snapshot_counter = 0
snapshot_message_timer = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    result = hand_landmarker.detect_for_video(mp_image, frame_timestamp)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            fingers = fingers_up(hand_landmarks)
            total = sum(fingers)

            index_tip = hand_landmarks[8]
            raw_x, raw_y = int(index_tip.x * w), int(index_tip.y * h)

            if smooth_x is None:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = int(smooth_x + smoothing_factor * (raw_x - smooth_x))
                smooth_y = int(smooth_y + smoothing_factor * (raw_y - smooth_y))

            x, y = smooth_x, smooth_y

            if total == 1:
                is_drawing = True
                stop_counter = 0
            else:
                stop_counter += 1
                if stop_counter > 3:
                    is_drawing = False

            if is_drawing:
                if prev_x is not None:
                    distance = np.hypot(x - prev_x, y - prev_y)
                    steps = max(min(int(distance / 2), 20), 1)
                    for i in range(steps):
                        ix = int(prev_x + (x - prev_x) * i / steps)
                        iy = int(prev_y + (y - prev_y) * i / steps)
                        ix2 = int(prev_x + (x - prev_x) * (i + 1) / steps)
                        iy2 = int(prev_y + (y - prev_y) * (i + 1) / steps)
                        cv2.line(canvas, (ix, iy), (ix2, iy2), (0, 255, 0), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

            if total == 0:
                save_counter += 1
                if save_counter == 15:
                    filename = f"signature_{int(time.time())}.png"
                    cv2.imwrite(filename, canvas)
                    print(f"Saved: {filename}")
                    save_message_timer = 30
            else:
                save_counter = 0

            if total == 4:
                snapshot_counter += 1
                if snapshot_counter == 15:
                    snap_filename = f"snapshot_{int(time.time())}.png"
                    cv2.imwrite(snap_filename, frame)
                    print(f"Snapshot saved: {snap_filename}")
                    snapshot_message_timer = 30
            else:
                snapshot_counter = 0
    else:
        prev_x, prev_y = None, None

    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    if save_message_timer > 0:
        cv2.rectangle(combined, (0, 0), (w - 1, h - 1), (0, 255, 0), 15)
        cv2.putText(combined, "Art Saved", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        save_message_timer -= 1

    if snapshot_message_timer > 0:
        cv2.rectangle(combined, (0, 0), (w - 1, h - 1), (255, 255, 0), 15)
        cv2.putText(combined, "Snap!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        snapshot_message_timer -= 1

    cv2.imshow("Air Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('c'):  # press 'c' to clear canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()