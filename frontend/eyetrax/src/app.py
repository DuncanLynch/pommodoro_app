import time
import os
import cv2
import numpy as np
import pygame

from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.utils.screen import get_screen_size

pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)
SOUND_PATH = os.path.join(os.path.dirname(__file__), "sound.mp3")


def start_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(SOUND_PATH)
        pygame.mixer.music.play(-1)


def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


est = GazeEstimator()
run_9_point_calibration(est)

sw, sh = get_screen_size()

margin_ratio = 0.10
mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
xmin, xmax = mx, sw - mx
ymin, ymax = my, sh - my

OFF_MS = 1500
ON_MS = 50

is_offscreen = False
outside_since = None
inside_since = None

cap = cv2.VideoCapture(0)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.monotonic()

        features, blink = est.extract_features(frame)

        if features is None or blink:
            outside = True
            x = y = None
        else:
            x, y = est.predict(np.array([features]))[0]
            x, y = float(x), float(y)
            outside = (x < xmin) or (x > xmax) or (y < ymin) or (y > ymax)

        if outside:
            inside_since = None
            if outside_since is None:
                outside_since = now
            if (not is_offscreen) and (now - outside_since) * 1000 >= OFF_MS:
                is_offscreen = True
                start_alarm()
        else:
            outside_since = None
            if inside_since is None:
                inside_since = now
            if is_offscreen and (now - inside_since) * 1000 >= ON_MS:
                is_offscreen = False
                stop_alarm()

        print(
            ("OFFSCREEN" if is_offscreen else "ONSCREEN"),
            f"x={x:.1f}, y={y:.1f}" if x is not None else "(no face)",
        )

except KeyboardInterrupt:
    pass
finally:
    stop_alarm()
    cap.release()
    pygame.mixer.quit()
