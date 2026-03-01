import time
import os, sys
import cv2
import numpy as np
import pygame
import requests
import threading

from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.utils.screen import get_screen_size

def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


ALARM_PATH = resource_path("sound.mp3")
CHEER_PATH = resource_path("cheer.mp3")

_AUDIO_READY = False
_ALARM_SOUND = None
_CHEER_SOUND = None
_ALARM_CHANNEL = None
_CHEER_CHANNEL = None


def init_audio():
    global _AUDIO_READY, _ALARM_SOUND, _CHEER_SOUND
    if _AUDIO_READY:
        return True

    try:
        if not pygame.get_init():
            pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()

        _ALARM_SOUND = pygame.mixer.Sound(ALARM_PATH)
        _CHEER_SOUND = pygame.mixer.Sound(CHEER_PATH)
        _ALARM_SOUND.set_volume(0.1)
        _CHEER_SOUND.set_volume(0.7)
        _AUDIO_READY = True
        return True
    except Exception as exc:
        print(f"[audio] Failed to initialize audio: {exc}")
        _AUDIO_READY = False
        return False


def start_alarm():
    global _ALARM_CHANNEL
    if not init_audio():
        return
    if _CHEER_CHANNEL is not None and _CHEER_CHANNEL.get_busy():
        _CHEER_CHANNEL.stop()
    if _ALARM_CHANNEL is None or not _ALARM_CHANNEL.get_busy():
        _ALARM_CHANNEL = _ALARM_SOUND.play(loops=-1)


def stop_alarm():
    if _ALARM_CHANNEL is not None and _ALARM_CHANNEL.get_busy():
        _ALARM_CHANNEL.stop()


def start_cheer():
    global _CHEER_CHANNEL
    if not init_audio():
        return
    if _ALARM_CHANNEL is not None and _ALARM_CHANNEL.get_busy():
        _ALARM_CHANNEL.stop()
    if _CHEER_CHANNEL is None or not _CHEER_CHANNEL.get_busy():
        _CHEER_CHANNEL = _CHEER_SOUND.play(loops=0)


def stop_cheer():
    if _CHEER_CHANNEL is not None and _CHEER_CHANNEL.get_busy():
        _CHEER_CHANNEL.stop()


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

        # camera preview for GUI (small RGB frame)
        self.last_cam_rgb = None        # np.ndarray (h, w, 3) in RGB
        self.last_cam_ts = 0.0

        self._worker = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._cap = None

    def stop(self):
        self._stop_event.set()
        stop_alarm()
        stop_cheer()

        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.5)

        with self._lock:
            self.state = self.states["wait"]
            self.end_time = time.monotonic()
            if self.start_time:
                self.elapsed_time = max(0.0, self.end_time - self.start_time)
            self.screen_status = "UNKNOWN"
            self.last_gaze = None
            self.offscreen_for_s = 0.0
            self.last_cam_rgb = None
            self.last_cam_ts = 0.0

        if worker is not None and not worker.is_alive():
            self._worker = None

        # TODO request logic here

    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return

        self._stop_event.clear()
        stop_alarm()
        stop_cheer()

        with self._lock:
            self.state = self.states["study"]
            self.start_time = time.monotonic()
            self.end_time = 0.0
            self.elapsed_time = 0.0

            self.screen_status = "UNKNOWN"
            self.last_gaze = None
            self.offscreen_for_s = 0.0
            self.last_cam_rgb = None
            self.last_cam_ts = 0.0

        # TODO request logic here

        self._worker = threading.Thread(target=self.pomoduro, daemon=True)
        self._worker.start()

    def pomoduro(self):
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    self.state = self.states["study"]
                    self.start_time = time.monotonic()
                    self.end_time = 0.0
                    self.elapsed_time = 0.0
                    self.screen_status = "UNKNOWN"
                    self.last_gaze = None
                    self.offscreen_for_s = 0.0
                    self.last_cam_rgb = None
                    self.last_cam_ts = 0.0

                study_completed = self.run()
                if self._stop_event.is_set() or not study_completed:
                    break

                with self._lock:
                    self.state = self.states["break"]
                    self.start_time = time.monotonic()
                    self.end_time = 0.0
                    self.elapsed_time = 0.0
                    self.screen_status = "BREAK"
                    self.last_gaze = None
                    self.offscreen_for_s = 0.0

                start_cheer()
                break_completed = self.chill()
                stop_cheer()

                if self._stop_event.is_set() or not break_completed:
                    break
        finally:
            stop_alarm()
            stop_cheer()
            with self._lock:
                if self.state != self.states["wait"]:
                    self.state = self.states["wait"]
                    self.end_time = time.monotonic()
            self._worker = None

    def chill(self):
        phase_start = time.monotonic()
        while True:
            if self._stop_event.is_set():
                return False

            now = time.monotonic()
            with self._lock:
                self.elapsed_time = max(0.0, now - phase_start)

            if now - phase_start >= POMO_BREAK_DUR_M * 60:
                return True

            time.sleep(0.05)

    def run(self):
        focus_accum_s = 0.0
        focus_start = None

        is_offscreen = False
        outside_since = None
        inside_since = None

        with self._lock:
            phase_start = self.start_time or time.monotonic()
            if not self.start_time:
                self.start_time = phase_start

        cap = cv2.VideoCapture(0)
        with self._cap_lock:
            self._cap = cap

        try:
            while True:
                if self._stop_event.is_set():
                    return False

                now = time.monotonic()
                with self._lock:
                    self.elapsed_time = max(0.0, now - phase_start)

                if now - phase_start >= POMO_STUDY_DUR_M * 60:
                    with self._lock:
                        self.end_time = now
                    return True

                ok, frame = cap.read()
                # --- store a small RGB preview for the GUI ---
                # keep it small to reduce CPU + lock contention
                preview_w, preview_h = 160, 120
                small = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                with self._lock:
                    self.last_cam_rgb = small_rgb
                    self.last_cam_ts = now
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
            with self._cap_lock:
                if self._cap is cap:
                    self._cap = None
            cap.release()


def draw_button(screen, rect, label, font, enabled=True):
    color = (70, 70, 70) if enabled else (40, 40, 40)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, (180, 180, 180), rect, width=2, border_radius=8)
    text = font.render(label, True, (255, 255, 255) if enabled else (140, 140, 140))
    tx = rect.x + (rect.w - text.get_width()) // 2
    ty = rect.y + (rect.h - text.get_height()) // 2
    screen.blit(text, (tx, ty))


def rgb_array_to_surface(rgb: np.ndarray) -> pygame.Surface:
    # rgb is (h, w, 3). pygame.surfarray.make_surface expects (w, h, 3)
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))

def run_gui(app: App):
    pygame.init()
    init_audio()
    screen = pygame.display.set_mode((620, 260))
    pygame.display.set_caption("Focus Timer Controls")
    clock = pygame.time.Clock()
    W, H = screen.get_size()
    cam_w, cam_h = 160, 120
    cam_rect = pygame.Rect(W - cam_w - 20, 20, cam_w, cam_h)

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

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    start_alarm()
                elif event.key == pygame.K_c:
                    start_cheer()
                elif event.key == pygame.K_s:
                    stop_alarm()
                    stop_cheer()

        with app._lock:
            state = app.state
            st = app.start_time
            et = app.end_time
            el = app.elapsed_time
            status = app.screen_status
            gaze = app.last_gaze
            off_for = app.offscreen_for_s
            worker_alive = app._worker.is_alive() if app._worker else False
            cam_rgb = None if app.last_cam_rgb is None else app.last_cam_rgb.copy()

        screen.fill((20, 20, 20))

        draw_button(screen, start_btn, "Start", big, enabled=not worker_alive)
        draw_button(screen, stop_btn, "Stop", big, enabled=worker_alive)

        # --- camera preview (top-right) ---
        pygame.draw.rect(screen, (60, 60, 60), cam_rect, border_radius=8)
        pygame.draw.rect(screen, (180, 180, 180), cam_rect, width=2, border_radius=8)

        if cam_rgb is not None:
            cam_surf = rgb_array_to_surface(cam_rgb)  # already 160x120
            screen.blit(cam_surf, cam_rect.topleft)
        else:
            # fallback label if no frame yet
            lbl = font.render("Camera: -", True, (200, 200, 200))
            screen.blit(lbl, (cam_rect.x + 10, cam_rect.y + 10))

        y = 85
        title = font.render("Session", True, (220, 220, 220))
        screen.blit(title, (20, y))
        y += 22

        if st:
            elapsed_label = "Break elapsed" if state == app.states["break"] else "Study elapsed" if state == app.states["study"] else "Elapsed"
            elapsed_line = f"{elapsed_label}: {format_hms(el)} ({el:0.1f}s)"
        else:
            elapsed_line = "Elapsed: -"

        lines = [
            f"State: {state}",
            f"Status: {status}" + (f"  (off for {off_for:0.1f}s)" if status == "OFFSCREEN" else ""),
            f"Start time (monotonic): {st:.3f}" if st else "Start time (monotonic): -",
            f"End time (monotonic):   {et:.3f}" if et else "End time (monotonic): -",
            elapsed_line,
            "Sound test hotkeys: A=alarm, C=cheer, S=stop",
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

    stop_alarm()
    stop_cheer()
    if pygame.mixer.get_init():
        pygame.mixer.quit()
    pygame.quit()


if __name__ == "__main__":
    app = App()
    run_gui(app)
