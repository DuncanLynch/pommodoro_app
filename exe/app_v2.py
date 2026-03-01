import time
import os, sys
import cv2
import numpy as np
import pygame
import requests
import threading

from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.utils.screen import get_screen_size

pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)

def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

ALARM_PATH = resource_path("sound.mp3")
CHEER_PATH = resource_path("cheer.mp3")


def start_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.set_volume(0.35)
        pygame.mixer.music.play(-1)


def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


def start_cheer():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(CHEER_PATH)
        pygame.mixer.music.set_volume(0.7)
        pygame.mixer.music.play(-1)


def stop_cheer():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


est = GazeEstimator()
run_9_point_calibration(est)

sw, sh = get_screen_size()

margin_ratio = 0.10
mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
xmin, xmax = mx, sw - mx
ymin, ymax = my, sh - my
threshold = 150

RESET_AFTER_OFFSCREEN = 30.0
ALARM_AFTER_S = 10.0
RETURN_DEBOUNCE_S = 0.05

POMO_STUDY_DUR_M = 1.0
POMO_BREAK_DUR_M = 0.5


def format_hms(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


class App:
    def __init__(self):
        self.state = "WAITING"
        self.states = {"wait": "WAITING", "study": "STUDYING", "break": "BREAK"}

        self.session = requests.Session()
        self.loggedin = False

        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_time = 0.0

        # added for GUI
        self.screen_status = "UNKNOWN"  # "ONSCREEN" / "OFFSCREEN" / "UNKNOWN"
        self.last_gaze = None           # (x, y) or None
        self.offscreen_for_s = 0.0      # how long currently offscreen

        self._worker = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

        with self._lock:
            self.state = self.states["wait"]
            self.end_time = time.monotonic()
            if self.start_time:
                self.elapsed_time = max(0.0, self.end_time - self.start_time)

        # TODO request logic here

    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return

        self._stop_event.clear()

        with self._lock:
            self.state = self.states["study"]
            self.start_time = time.monotonic()
            self.end_time = 0.0
            self.elapsed_time = 0.0

            self.screen_status = "UNKNOWN"
            self.last_gaze = None
            self.offscreen_for_s = 0.0

        # TODO request logic here

        self._worker = threading.Thread(target=self.run, daemon=True)
        self._worker.start()

    def pomoduro(self):
        while True:
            self.run()
            with self._lock:
                self.state = self.states["break"]
            self.chill()
            with self._lock:
                self.state = self.states["study"]

    def chill(self):
        while True:
            if self._stop_event.is_set():
                return

            now = time.monotonic()
            with self._lock:
                if self.start_time:
                    self.elapsed_time = max(0.0, now - self.start_time)

            if now - self.start_time > POMO_BREAK_DUR_M * 60:
                return

            time.sleep(0.05)

    def run(self):
        focus_accum_s = 0.0
        focus_start = None

        is_offscreen = False
        outside_since = None
        inside_since = None

        cap = cv2.VideoCapture(0)

        try:
            while True:
                if self._stop_event.is_set():
                    return

                now = time.monotonic()
                with self._lock:
                    if self.start_time:
                        self.elapsed_time = max(0.0, now - self.start_time)

                # optional: end after study duration
                if now - self.start_time > POMO_STUDY_DUR_M * 60:
                    return

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                features, __blink = est.extract_features(frame)

                if features is None:
                    outside = True
                    x = y = None
                else:
                    x, y = est.predict(np.array([features]))[0]
                    x, y = float(x), float(y)
                    outside = (
                        (x < xmin - threshold)
                        or (x > xmax + threshold)
                        or (y < ymin - threshold)
                        or (y > ymax + threshold)
                    )

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

                else:
                    outside_since = None
                    if inside_since is None:
                        inside_since = now

                    if is_offscreen and (now - inside_since) >= RETURN_DEBOUNCE_S:
                        is_offscreen = False
                        stop_alarm()

                    if not is_offscreen and focus_start is None:
                        focus_start = now

                focus_total = focus_accum_s
                if focus_start is not None:
                    focus_total += (now - focus_start)

                # update GUI-visible fields safely
                with self._lock:
                    self.screen_status = "OFFSCREEN" if is_offscreen else "ONSCREEN"
                    self.last_gaze = None if x is None else (x, y)
                    self.offscreen_for_s = 0.0 if outside_since is None else (now - outside_since)

                # keep CPU reasonable
                time.sleep(0.005)

        except KeyboardInterrupt:
            pass
        finally:
            stop_alarm()
            cap.release()
            pygame.mixer.quit()


def draw_button(screen, rect, label, font, enabled=True):
    color = (70, 70, 70) if enabled else (40, 40, 40)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, (180, 180, 180), rect, width=2, border_radius=8)
    text = font.render(label, True, (255, 255, 255) if enabled else (140, 140, 140))
    tx = rect.x + (rect.w - text.get_width()) // 2
    ty = rect.y + (rect.h - text.get_height()) // 2
    screen.blit(text, (tx, ty))


def run_gui(app: App):
    pygame.init()
    screen = pygame.display.set_mode((620, 260))
    pygame.display.set_caption("Focus Timer Controls")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)
    big = pygame.font.SysFont(None, 28)

    start_btn = pygame.Rect(20, 20, 140, 45)
    stop_btn = pygame.Rect(180, 20, 140, 45)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                app.stop()
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if start_btn.collidepoint(mx, my):
                    app.start()
                elif stop_btn.collidepoint(mx, my):
                    app.stop()

        with app._lock:
            state = app.state
            st = app.start_time
            et = app.end_time
            el = app.elapsed_time
            status = app.screen_status
            gaze = app.last_gaze
            off_for = app.offscreen_for_s
            worker_alive = app._worker.is_alive() if app._worker else False

        screen.fill((20, 20, 20))

        draw_button(screen, start_btn, "Start", big, enabled=not worker_alive)
        draw_button(screen, stop_btn, "Stop", big, enabled=worker_alive)

        y = 85
        title = font.render("Session", True, (220, 220, 220))
        screen.blit(title, (20, y))
        y += 22

        lines = [
            f"State: {state}",
            f"Status: {status}" + (f"  (off for {off_for:0.1f}s)" if status == "OFFSCREEN" else ""),
            f"Start time (monotonic): {st:.3f}" if st else "Start time (monotonic): -",
            f"End time (monotonic):   {et:.3f}" if et else "End time (monotonic): -",
            f"Elapsed: {format_hms(el)} ({el:0.1f}s)" if st else "Elapsed: -",
        ]

        if gaze is None:
            lines.append("Gaze: (no face / blink)")
        else:
            gx, gy = gaze
            lines.append(f"Gaze: x={gx:0.0f}, y={gy:0.0f}")

        for line in lines:
            txt = font.render(line, True, (200, 200, 200))
            screen.blit(txt, (20, y))
            y += 22

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    app = App()
    run_gui(app)