from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.utils.screen import get_screen_size
import cv2
import numpy as np

est = GazeEstimator()
run_9_point_calibration(est)

sw, sh = get_screen_size()

# calibrated inset bounds (matches compute_grid_points default margin_ratio=0.10)
margin_ratio = 0.10
mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
xmin, xmax = mx, sw - mx
ymin, ymax = my, sh - my

OUT_N = 5  # frames to confirm off-screen
IN_N = 5  # frames to confirm on-screen
out_ctr = in_ctr = 0
is_offscreen = False

cap = cv2.VideoCapture(4)

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    features, blink = est.extract_features(frame)
    if features is None or blink:
        continue

    x, y = est.predict(np.array([features]))[0]
    x, y = float(x), float(y)

    outside = (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax)

    if outside:
        out_ctr += 1
        in_ctr = 0
    else:
        in_ctr += 1
        out_ctr = 0

    if not is_offscreen and out_ctr >= OUT_N:
        is_offscreen = True
    if is_offscreen and in_ctr >= IN_N:
        is_offscreen = False

    print("OFFSCREEN" if is_offscreen else "ONSCREEN", x, y)
