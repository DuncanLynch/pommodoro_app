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

OFF_MS = 5000
ON_MS = 50

RESET_AFTER_OFFSCREEN = 30.0
ALARM_AFTER_S = 5.0
RETURN_DEBOUNCE_S = 0.05 # idk what this is for but chat said it would help

focus_accum_s = 0.0
focus_start = None

is_offscreen = False
outside_since = None
inside_since = None

cap = cv2.VideoCapture(0)

# courtesy of le chat meow
def format_hms(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

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

            if not is_offscreen:
                is_offscreen = True
                if focus_start is not None:
                    focus_accum_s += (now - focus_start)
                    focus_start = None
            off_dur = now - outside_since

            if off_dur >= ALARM_AFTER_S:
                start_alarm()

            if off_dur >= RESET_AFTER_OFFSCREEN:
                focus_accum_s = 0.0
                focus_start = None 

            #if (not is_offscreen) and (now - outside_since) * 1000 >= OFF_MS:
            #    is_offscreen = True
            #    start_alarm()
        else:
            outside_since = None
            if inside_since is None:
                inside_since = now
            #if is_offscreen and (now - inside_since) * 1000 >= ON_MS:
            if is_offscreen and (now - inside_since) >= RETURN_DEBOUNCE_S:
                is_offscreen = False
                stop_alarm()

            if not is_offscreen and focus_start is None:
                focus_start = now

        focus_total = focus_accum_s
        if focus_start is not None:
            focus_total += (now - focus_start)

        print(
            ("OFFSCREEN" if is_offscreen else "ONSCREEN"),
            f"x={x:.1f}, y={y:.1f}" if x is not None else "(no face)",
        )

        # courtesy of le chat 
        # ----------------------------
        # Display window
        # ----------------------------
        display = np.zeros((420, 900, 3), dtype=np.uint8)

        status = "OFFSCREEN" if is_offscreen else "ONSCREEN"
        cv2.putText(display, "Focus Timer", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(display, format_hms(focus_total), (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(display, f"Status: {status}", (30, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        if outside_since is not None:
            off_for = now - outside_since
            cv2.putText(display, f"Offscreen: {off_for:0.1f}s", (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Resets at: {RESET_AFTER_OFFSCREEN:.0f}s", (30, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"Alarm at: {ALARM_AFTER_S:.0f}s", (30, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "Offscreen: 0.0s", (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Optional: show last gaze coords
        if x is not None:
            cv2.putText(display, f"Gaze: x={x:.0f}, y={y:.0f}", (520, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display, "Gaze: (no face / blink)", (520, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Focus Timer", display)

        # Press q or ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break


except KeyboardInterrupt:
    pass
finally:
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
