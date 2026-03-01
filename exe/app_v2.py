import ctypes
import math
import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
import requests

from eyetrax import GazeEstimator, run_9_point_calibration
from eyetrax.utils.screen import get_screen_size


def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)

IP = "localhost"
PORT = "5000"
API_BASE_URL = f"http://{IP}:{PORT}"
API_TIMEOUT_S = 3.0


ALARM_PATH = resource_path("sound.mp3")
CHEER_PATH = resource_path("cheer.mp3")

WINDOW_W = 1200
WINDOW_H = 780
FPS = 60

UI_MARGIN = 24
HEADER_Y = 20
HEADER_H = 98
MAIN_Y = 136
MAIN_H = 414
SETTINGS_Y = 566
SETTINGS_H = 190

SETTINGS_MINUS_X_OFFSET = 416
SETTINGS_VALUE_X_OFFSET = 466
SETTINGS_PLUS_X_OFFSET = 624
SETTINGS_ROW_TOP_OFFSET = 60
SETTINGS_ROW_HEIGHT = 30
SETTINGS_VALUE_SIZE = (146, 28)
SETTINGS_BUTTON_SIZE = (34, 26)

STATE_WAITING = "WAITING"
STATE_STUDY = "STUDYING"
STATE_BREAK = "BREAK"

MARGIN_RATIO = 0.10
THRESHOLD_PX = 150
RETURN_DEBOUNCE_S = 0.05

COL_BG_TOP = (14, 22, 39)
COL_BG_BOTTOM = (6, 10, 18)
COL_CARD = (23, 34, 56)
COL_CARD_ALT = (29, 41, 67)
COL_BORDER = (68, 90, 129)
COL_TEXT = (236, 242, 255)
COL_TEXT_DIM = (168, 184, 214)
COL_ACCENT = (66, 176, 255)
COL_ACCENT_2 = (74, 222, 195)
COL_DANGER = (255, 112, 122)
COL_WARNING = (255, 190, 93)
COL_MUTED = (130, 143, 168)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def format_hms(seconds):
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def rgb_array_to_surface(rgb):
    return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


def build_gradient_surface(width, height, top, bottom):
    surface = pygame.Surface((width, height))
    for y in range(height):
        t = y / max(1, height - 1)
        color = (
            int(top[0] + (bottom[0] - top[0]) * t),
            int(top[1] + (bottom[1] - top[1]) * t),
            int(top[2] + (bottom[2] - top[2]) * t),
        )
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface


def draw_ambient_orbs(screen, tick_s):
    layer = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    x1 = int(WINDOW_W * 0.2 + math.sin(tick_s * 0.35) * 40)
    y1 = int(WINDOW_H * 0.2 + math.cos(tick_s * 0.45) * 30)
    x2 = int(WINDOW_W * 0.8 + math.cos(tick_s * 0.28) * 45)
    y2 = int(WINDOW_H * 0.72 + math.sin(tick_s * 0.32) * 35)
    pygame.draw.circle(layer, (52, 141, 255, 52), (x1, y1), 190)
    pygame.draw.circle(layer, (71, 215, 185, 40), (x2, y2), 220)
    screen.blit(layer, (0, 0))


def draw_card(screen, rect, fill, border, radius=18):
    shadow = rect.move(0, 6)
    pygame.draw.rect(screen, (7, 11, 19), shadow, border_radius=radius)
    pygame.draw.rect(screen, fill, rect, border_radius=radius)
    pygame.draw.rect(screen, border, rect, width=2, border_radius=radius)


def draw_progress_bar(screen, rect, progress):
    pygame.draw.rect(screen, (36, 49, 76), rect, border_radius=10)
    fill_w = int(rect.w * clamp(progress, 0.0, 1.0))
    if fill_w > 0:
        fill_rect = pygame.Rect(rect.x, rect.y, fill_w, rect.h)
        pygame.draw.rect(screen, COL_ACCENT, fill_rect, border_radius=10)
    pygame.draw.rect(screen, (74, 98, 141), rect, width=2, border_radius=10)


def draw_progress_ring(screen, center, radius, progress):
    width = 12
    pygame.draw.circle(screen, (45, 63, 95), center, radius, width=width)
    if progress > 0:
        rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        start = -math.pi / 2
        end = start + 2 * math.pi * clamp(progress, 0.0, 1.0)
        pygame.draw.arc(screen, COL_ACCENT_2, rect, start, end, width=width)


def brighten(color, delta):
    return (
        clamp(color[0] + delta, 0, 255),
        clamp(color[1] + delta, 0, 255),
        clamp(color[2] + delta, 0, 255),
    )


def fit_text(font, text, max_w):
    if max_w <= 0:
        return ""
    if font.size(text)[0] <= max_w:
        return text
    suffix = "..."
    for idx in range(len(text), -1, -1):
        candidate = text[:idx] + suffix
        if font.size(candidate)[0] <= max_w:
            return candidate
    return suffix


def enable_windows_dpi_awareness():
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


@dataclass(frozen=True)
class ButtonSpec:
    key: str
    rect: pygame.Rect
    label: str
    style: str
    enabled: bool


def draw_button(screen, button, font, hovered):
    if button.style == "primary":
        base = (44, 154, 255)
        border = (95, 188, 255)
    elif button.style == "danger":
        base = (198, 67, 84)
        border = (239, 110, 126)
    elif button.style == "ghost":
        base = (48, 64, 96)
        border = (86, 109, 149)
    else:
        base = (58, 78, 114)
        border = (96, 120, 164)

    if not button.enabled:
        fill = (44, 53, 70)
        edge = (73, 84, 105)
        text_col = (128, 138, 157)
    else:
        fill = brighten(base, 16) if hovered else base
        edge = brighten(border, 10) if hovered else border
        text_col = COL_TEXT

    pygame.draw.rect(screen, fill, button.rect, border_radius=12)
    pygame.draw.rect(screen, edge, button.rect, width=2, border_radius=12)
    txt = font.render(button.label, True, text_col)
    tx = button.rect.x + (button.rect.w - txt.get_width()) // 2
    ty = button.rect.y + (button.rect.h - txt.get_height()) // 2
    screen.blit(txt, (tx, ty))


class AudioManager:
    def __init__(self, alarm_path, cheer_path):
        self.alarm_path = alarm_path
        self.cheer_path = cheer_path
        self.ready = False
        self.muted = False
        self.lock = threading.RLock()
        self.alarm_sound = None
        self.cheer_sound = None
        self.alarm_channel = None
        self.cheer_channel = None

    def initialize(self):
        with self.lock:
            if self.ready:
                return True
            try:
                if not pygame.get_init():
                    pygame.init()
                if not pygame.mixer.get_init():
                    pygame.mixer.pre_init(44100, -16, 2, 512)
                    pygame.mixer.init()

                if not os.path.exists(self.alarm_path):
                    raise FileNotFoundError(self.alarm_path)

                cheer_path = self.cheer_path if os.path.exists(self.cheer_path) else self.alarm_path
                self.alarm_sound = pygame.mixer.Sound(self.alarm_path)
                self.cheer_sound = pygame.mixer.Sound(cheer_path)
                self.alarm_sound.set_volume(0.28)
                self.cheer_sound.set_volume(0.72)
                self.ready = True
                return True
            except Exception as exc:
                print(f"[audio] Initialization failed: {exc}")
                self.ready = False
                return False

    def start_alarm(self):
        with self.lock:
            if self.muted:
                return
            if not self.initialize():
                return
            if self.cheer_channel is not None and self.cheer_channel.get_busy():
                self.cheer_channel.stop()
            if self.alarm_channel is None or not self.alarm_channel.get_busy():
                self.alarm_channel = self.alarm_sound.play(loops=-1)

    def stop_alarm(self):
        with self.lock:
            if self.alarm_channel is not None and self.alarm_channel.get_busy():
                self.alarm_channel.stop()

    def play_cheer(self):
        with self.lock:
            if self.muted:
                return
            if not self.initialize():
                return
            if self.alarm_channel is not None and self.alarm_channel.get_busy():
                self.alarm_channel.stop()
            if self.cheer_channel is None or not self.cheer_channel.get_busy():
                self.cheer_channel = self.cheer_sound.play(loops=0)

    def stop_cheer(self):
        with self.lock:
            if self.cheer_channel is not None and self.cheer_channel.get_busy():
                self.cheer_channel.stop()

    def toggle_mute(self):
        with self.lock:
            self.muted = not self.muted
            if self.muted:
                if self.alarm_channel is not None and self.alarm_channel.get_busy():
                    self.alarm_channel.stop()
                if self.cheer_channel is not None and self.cheer_channel.get_busy():
                    self.cheer_channel.stop()
            return self.muted

    def shutdown(self):
        with self.lock:
            if self.alarm_channel is not None and self.alarm_channel.get_busy():
                self.alarm_channel.stop()
            if self.cheer_channel is not None and self.cheer_channel.get_busy():
                self.cheer_channel.stop()
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            self.ready = False


@dataclass(frozen=True)
class AppSnapshot:
    state: str
    running: bool
    elapsed_s: float
    remaining_s: float
    phase_goal_s: float
    progress: float
    focus_s: float
    cycle_count: int
    screen_status: str
    offscreen_for_s: float
    gaze_text: str
    cam_rgb: np.ndarray | None
    study_minutes: float
    break_minutes: float
    alarm_after_s: float
    reset_after_s: float
    muted: bool
    loggedin: bool
    auth_status: str

class App:
    def __init__(self):
        self.states = {"wait": STATE_WAITING, "study": STATE_STUDY, "break": STATE_BREAK}

        self.session = requests.Session()
        self.loggedin = False
        self.auth_username = ""
        self.auth_password = ""
        self.auth_status = "Not logged in"
        self.api_base_url = API_BASE_URL
        self.api_timeout_s = API_TIMEOUT_S

        self.study_duration_s = 25 * 60
        self.break_duration_s = 5 * 60
        self.alarm_after_s = 10.0
        self.reset_after_s = 30.0

        self.state = STATE_WAITING
        self.phase_goal_s = 0.0
        self.elapsed_s = 0.0
        self.focus_s = 0.0
        self.cycle_count = 0

        self.screen_status = "IDLE"
        self.last_gaze = None
        self.offscreen_for_s = 0.0
        self.last_cam_rgb = None

        self._worker = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._cap = None

        self.audio = AudioManager(ALARM_PATH, CHEER_PATH)

        self.est = GazeEstimator()
        run_9_point_calibration(self.est)

        sw, sh = get_screen_size()
        mx, my = int(sw * MARGIN_RATIO), int(sh * MARGIN_RATIO)
        self.xmin, self.xmax = mx, sw - mx
        self.ymin, self.ymax = my, sh - my

    def _worker_alive_unlocked(self):
        return self._worker is not None and self._worker.is_alive()

    def _safe_post(self, endpoint, payload=None, require_login=True):
        if require_login and not self.loggedin:
            return False, "Not logged in"

        try:
            response = self.session.post(
                f"{self.api_base_url}{endpoint}",
                json=payload,
                timeout=self.api_timeout_s,
            )
        except requests.RequestException as exc:
            return False, f"Network error: {exc}"

        if response.status_code >= 400:
            if response.status_code in (401, 403):
                self.loggedin = False
            detail = ""
            try:
                body = response.json()
                detail = body.get("error") or body.get("message") or ""
            except ValueError:
                detail = response.text.strip()
            detail = detail[:120]
            if detail:
                return False, f"HTTP {response.status_code}: {detail}"
            return False, f"HTTP {response.status_code}"
        return True, "OK"

    def login(self, username, password):
        username = (username or "").strip()
        password = password or ""
        self.auth_username = username
        self.auth_password = password

        if not username or not password:
            self.loggedin = False
            self.auth_status = "Username and password required"
            return False

        ok, message = self._safe_post(
            "/login",
            payload={"username": username, "password": password},
            require_login=False,
        )
        if ok:
            self.loggedin = True
            self.auth_status = f"Logged in as {username}"
            return True

        self.loggedin = False
        self.auth_status = f"Login failed: {message}"
        return False

    def _post_session_event(self, endpoint):
        if not self.loggedin:
            self.auth_status = "Not logged in; session sync skipped"
            return False
        ok, message = self._safe_post(endpoint, require_login=False)
        if not ok:
            self.auth_status = f"Sync failed: {message}"
        return ok

    def start(self):
        with self._lock:
            if self._worker_alive_unlocked():
                return

        self._stop_event.clear()
        self.audio.stop_alarm()
        self.audio.stop_cheer()

        with self._lock:
            self.state = STATE_WAITING
            self.phase_goal_s = 0.0
            self.elapsed_s = 0.0
            self.focus_s = 0.0
            self.cycle_count = 0
            self.screen_status = "IDLE"
            self.last_gaze = None
            self.offscreen_for_s = 0.0
            self.last_cam_rgb = None

        self._worker = threading.Thread(target=self._pomodoro_loop, daemon=True)
        self._post_session_event("/start_session")
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        self.audio.stop_alarm()
        self.audio.stop_cheer()

        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

        worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=3.0)

        with self._lock:
            self.state = STATE_WAITING
            self.phase_goal_s = 0.0
            self.elapsed_s = 0.0
            self.focus_s = 0.0
            self.screen_status = "IDLE"
            self.last_gaze = None
            self.offscreen_for_s = 0.0
            self.last_cam_rgb = None
        self._post_session_event("/end_session")

        if worker is not None and not worker.is_alive():
            self._worker = None

    def shutdown(self):
        self.stop()
        self.audio.shutdown()

    def reset_stats(self):
        with self._lock:
            if self._worker_alive_unlocked():
                return
            self.cycle_count = 0
            self.elapsed_s = 0.0
            self.focus_s = 0.0

    def adjust_minutes(self, target, delta_minutes):
        with self._lock:
            if self._worker_alive_unlocked():
                return
            delta_s = int(delta_minutes * 60)
            if target == "study":
                self.study_duration_s = int(clamp(self.study_duration_s + delta_s, 60, 120 * 60))
            elif target == "break":
                self.break_duration_s = int(clamp(self.break_duration_s + delta_s, 60, 60 * 60))

    def adjust_alarm_after(self, delta_s):
        with self._lock:
            if self._worker_alive_unlocked():
                return
            self.alarm_after_s = float(clamp(self.alarm_after_s + delta_s, 1.0, 60.0))

    def adjust_reset_after(self, delta_s):
        with self._lock:
            if self._worker_alive_unlocked():
                return
            self.reset_after_s = float(clamp(self.reset_after_s + delta_s, 5.0, 240.0))

    def toggle_mute(self):
        self.audio.toggle_mute()

    def test_sound(self):
        self.audio.play_cheer()

    def snapshot(self):
        with self._lock:
            running = self._worker_alive_unlocked()
            goal = self.phase_goal_s
            elapsed = self.elapsed_s
            remaining = max(0.0, goal - elapsed) if goal > 0 else 0.0
            progress = clamp(elapsed / goal, 0.0, 1.0) if goal > 0 else 0.0
            if self.last_gaze is None:
                gaze_text = "(no face / blink)"
            else:
                gx, gy = self.last_gaze
                gaze_text = f"x={gx:0.0f}, y={gy:0.0f}"
            cam_rgb = None if self.last_cam_rgb is None else self.last_cam_rgb.copy()
            return AppSnapshot(
                state=self.state,
                running=running,
                elapsed_s=elapsed,
                remaining_s=remaining,
                phase_goal_s=goal,
                progress=progress,
                focus_s=self.focus_s,
                cycle_count=self.cycle_count,
                screen_status=self.screen_status,
                offscreen_for_s=self.offscreen_for_s,
                gaze_text=gaze_text,
                cam_rgb=cam_rgb,
                study_minutes=self.study_duration_s / 60.0,
                break_minutes=self.break_duration_s / 60.0,
                alarm_after_s=self.alarm_after_s,
                reset_after_s=self.reset_after_s,
                muted=self.audio.muted,
                loggedin=self.loggedin,
                auth_status=self.auth_status,
            )

    def _begin_phase(self, state, duration_s):
        with self._lock:
            self.state = state
            self.phase_goal_s = float(duration_s)
            self.elapsed_s = 0.0
            self.focus_s = 0.0
            if state == STATE_BREAK:
                self.screen_status = "BREAK"
                self.last_gaze = None
                self.offscreen_for_s = 0.0
                self.last_cam_rgb = None

    def _pomodoro_loop(self):
        try:
            while not self._stop_event.is_set():
                self._begin_phase(STATE_STUDY, self.study_duration_s)
                study_ok = self._run_study_phase(self.study_duration_s)
                if self._stop_event.is_set() or not study_ok:
                    break

                with self._lock:
                    self.cycle_count += 1

                self.audio.play_cheer()

                self._begin_phase(STATE_BREAK, self.break_duration_s)
                break_ok = self._run_break_phase(self.break_duration_s)
                if self._stop_event.is_set() or not break_ok:
                    break
        finally:
            self.audio.stop_alarm()
            self.audio.stop_cheer()
            with self._cap_lock:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
            with self._lock:
                self.state = STATE_WAITING
                self.phase_goal_s = 0.0
                self.elapsed_s = 0.0
                self.focus_s = 0.0
                self.screen_status = "IDLE"
                self.last_gaze = None
                self.offscreen_for_s = 0.0
                self.last_cam_rgb = None
            self._worker = None

    def _run_break_phase(self, duration_s):
        phase_start = time.monotonic()
        while True:
            if self._stop_event.is_set():
                return False

            now = time.monotonic()
            elapsed = now - phase_start
            with self._lock:
                self.elapsed_s = elapsed
                self.focus_s = 0.0
                self.screen_status = "BREAK"
                self.last_gaze = None
                self.offscreen_for_s = 0.0
                self.last_cam_rgb = None

            if elapsed >= duration_s:
                return True

            time.sleep(0.05)

    def _run_study_phase(self, duration_s):
        focus_accum_s = 0.0
        focus_start = None
        is_offscreen = False
        outside_since = None
        inside_since = None

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        with self._cap_lock:
            self._cap = cap

        if not cap.isOpened():
            with self._lock:
                self.screen_status = "CAMERA ERROR"
            return False

        phase_start = time.monotonic()
        try:
            while True:
                if self._stop_event.is_set():
                    return False

                now = time.monotonic()
                elapsed = now - phase_start
                with self._lock:
                    self.elapsed_s = elapsed

                if elapsed >= duration_s:
                    return True

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                small = cv2.resize(frame, (210, 156), interpolation=cv2.INTER_AREA)
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                features, blink = self.est.extract_features(frame)
                if features is None or blink:
                    outside = True
                    x = y = None
                else:
                    x, y = self.est.predict(np.array([features]))[0]
                    x, y = float(x), float(y)
                    outside = (
                        (x < self.xmin - THRESHOLD_PX)
                        or (x > self.xmax + THRESHOLD_PX)
                        or (y < self.ymin - THRESHOLD_PX)
                        or (y > self.ymax + THRESHOLD_PX)
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
                    if off_dur >= self.alarm_after_s:
                        self.audio.start_alarm()
                    if off_dur >= self.reset_after_s:
                        focus_accum_s = 0.0
                        focus_start = None
                else:
                    outside_since = None
                    if inside_since is None:
                        inside_since = now

                    if is_offscreen and (now - inside_since) >= RETURN_DEBOUNCE_S:
                        is_offscreen = False
                        self.audio.stop_alarm()

                    if not is_offscreen and focus_start is None:
                        focus_start = now

                focus_total = focus_accum_s
                if focus_start is not None:
                    focus_total += (now - focus_start)

                with self._lock:
                    self.focus_s = focus_total
                    self.screen_status = "OFFSCREEN" if is_offscreen else "ONSCREEN"
                    self.last_gaze = None if x is None else (x, y)
                    self.offscreen_for_s = 0.0 if outside_since is None else (now - outside_since)
                    self.last_cam_rgb = small_rgb

                time.sleep(0.01)
        finally:
            self.audio.stop_alarm()
            with self._cap_lock:
                if self._cap is cap:
                    self._cap = None
            cap.release()

def build_buttons(snapshot):
    header = pygame.Rect(UI_MARGIN, HEADER_Y, WINDOW_W - (UI_MARGIN * 2), HEADER_H)
    settings = pygame.Rect(UI_MARGIN, SETTINGS_Y, WINDOW_W - (UI_MARGIN * 2), SETTINGS_H)

    buttons = [
        ButtonSpec("login", pygame.Rect(header.right - 630, header.y + 28, 80, 42), "Reauth" if snapshot.loggedin else "Login", "secondary", True),
        ButtonSpec("start", pygame.Rect(header.right - 540, header.y + 28, 120, 42), "Start", "primary", not snapshot.running),
        ButtonSpec("stop", pygame.Rect(header.right - 410, header.y + 28, 120, 42), "Stop", "danger", snapshot.running),
        ButtonSpec("reset", pygame.Rect(header.right - 280, header.y + 28, 120, 42), "Reset", "secondary", not snapshot.running),
        ButtonSpec("mute", pygame.Rect(header.right - 150, header.y + 28, 120, 42), "Unmute" if snapshot.muted else "Mute", "ghost", True),
        ButtonSpec("test", pygame.Rect(settings.right - 150, settings.y + 74, 120, 42), "Test Sound", "ghost", True),
    ]

    enabled = not snapshot.running
    minus_x = settings.x + SETTINGS_MINUS_X_OFFSET
    plus_x = settings.x + SETTINGS_PLUS_X_OFFSET
    row_top = settings.y + SETTINGS_ROW_TOP_OFFSET
    row_gap = SETTINGS_ROW_HEIGHT
    button_w, button_h = SETTINGS_BUTTON_SIZE
    keys = [
        ("study_minus", "study_plus"),
        ("break_minus", "break_plus"),
        ("alarm_minus", "alarm_plus"),
        ("reset_minus", "reset_plus"),
    ]
    for idx, (k_minus, k_plus) in enumerate(keys):
        y = row_top + idx * row_gap + (row_gap - button_h) // 2
        buttons.append(ButtonSpec(k_minus, pygame.Rect(minus_x, y, button_w, button_h), "-", "secondary", enabled))
        buttons.append(ButtonSpec(k_plus, pygame.Rect(plus_x, y, button_w, button_h), "+", "secondary", enabled))

    return buttons


def handle_button(app, key, username="", password=""):
    app.auth_username = username
    app.auth_password = password
    if key == "start":
        app.start()
    elif key == "stop":
        app.stop()
    elif key == "reset":
        app.reset_stats()
    elif key == "mute":
        app.toggle_mute()
    elif key == "test":
        app.test_sound()
    elif key == "study_minus":
        app.adjust_minutes("study", -1)
    elif key == "study_plus":
        app.adjust_minutes("study", 1)
    elif key == "break_minus":
        app.adjust_minutes("break", -1)
    elif key == "break_plus":
        app.adjust_minutes("break", 1)
    elif key == "alarm_minus":
        app.adjust_alarm_after(-1)
    elif key == "alarm_plus":
        app.adjust_alarm_after(1)
    elif key == "reset_minus":
        app.adjust_reset_after(-5)
    elif key == "reset_plus":
        app.adjust_reset_after(5)
    elif key == "login":
        app.login(username, password)


def run_gui(app):
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    enable_windows_dpi_awareness()
    pygame.init()
    app.audio.initialize()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Pomo")
    clock = pygame.time.Clock()

    background = build_gradient_surface(WINDOW_W, WINDOW_H, COL_BG_TOP, COL_BG_BOTTOM)

    font_title = pygame.font.SysFont("Segoe UI Semibold", 40)
    font_h1 = pygame.font.SysFont("Segoe UI Semibold", 30)
    font_h2 = pygame.font.SysFont("Segoe UI Semibold", 22)
    font_body = pygame.font.SysFont("Segoe UI", 19)
    font_small = pygame.font.SysFont("Segoe UI", 16)
    font_timer = pygame.font.SysFont("Consolas", 56, bold=True)
    font_value = pygame.font.SysFont("Consolas", 24, bold=True)

    ui_running = True
    username_input = app.auth_username
    password_input = app.auth_password
    active_field = None

    while ui_running:
        snapshot = app.snapshot()
        buttons = build_buttons(snapshot)
        mouse = pygame.mouse.get_pos()

        header_rect = pygame.Rect(UI_MARGIN, HEADER_Y, WINDOW_W - (UI_MARGIN * 2), HEADER_H)
        main_rect = pygame.Rect(UI_MARGIN, MAIN_Y, 730, MAIN_H)
        status_rect = pygame.Rect(770, MAIN_Y, WINDOW_W - 794, MAIN_H)
        settings_rect = pygame.Rect(UI_MARGIN, SETTINGS_Y, WINDOW_W - (UI_MARGIN * 2), SETTINGS_H)
        username_rect = pygame.Rect(header_rect.x + 210, header_rect.y + 24, 150, 30)
        password_rect = pygame.Rect(header_rect.x + 368, header_rect.y + 24, 150, 30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ui_running = False
            elif event.type == pygame.KEYDOWN:
                if active_field is not None:
                    if event.key == pygame.K_TAB:
                        active_field = "password" if active_field == "username" else "username"
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        handle_button(app, "login", username_input, password_input)
                    elif event.key == pygame.K_ESCAPE:
                        active_field = None
                    elif event.key == pygame.K_BACKSPACE:
                        if active_field == "username":
                            username_input = username_input[:-1]
                        else:
                            password_input = password_input[:-1]
                    else:
                        typed = event.unicode
                        if typed and typed.isprintable():
                            if active_field == "username" and len(username_input) < 48:
                                username_input += typed
                            elif active_field == "password" and len(password_input) < 64:
                                password_input += typed
                    continue

                if event.key == pygame.K_ESCAPE:
                    ui_running = False
                elif event.key == pygame.K_SPACE:
                    if snapshot.running:
                        app.stop()
                    else:
                        app.start()
                elif event.key == pygame.K_m:
                    app.toggle_mute()
                elif event.key == pygame.K_t:
                    app.test_sound()
                elif event.key == pygame.K_r:
                    app.reset_stats()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if username_rect.collidepoint(event.pos):
                    active_field = "username"
                    continue
                if password_rect.collidepoint(event.pos):
                    active_field = "password"
                    continue
                active_field = None
                for button in buttons:
                    if button.enabled and button.rect.collidepoint(event.pos):
                        handle_button(app, button.key, username_input, password_input)
                        break
        app.auth_username = username_input
        app.auth_password = password_input

        screen.blit(background, (0, 0))
        draw_ambient_orbs(screen, time.monotonic())

        draw_card(screen, header_rect, COL_CARD, COL_BORDER)
        draw_card(screen, main_rect, COL_CARD_ALT, COL_BORDER)
        draw_card(screen, status_rect, COL_CARD, COL_BORDER)
        draw_card(screen, settings_rect, COL_CARD, COL_BORDER)

        title = font_title.render("Pomo", True, COL_TEXT)
        screen.blit(title, (header_rect.x + 24, header_rect.y + 16))

        user_fill = (20, 30, 46) if active_field == "username" else (18, 27, 44)
        pass_fill = (20, 30, 46) if active_field == "password" else (18, 27, 44)
        user_border = COL_ACCENT if active_field == "username" else (69, 90, 130)
        pass_border = COL_ACCENT if active_field == "password" else (69, 90, 130)
        pygame.draw.rect(screen, user_fill, username_rect, border_radius=8)
        pygame.draw.rect(screen, user_border, username_rect, width=2, border_radius=8)
        pygame.draw.rect(screen, pass_fill, password_rect, border_radius=8)
        pygame.draw.rect(screen, pass_border, password_rect, width=2, border_radius=8)

        shown_username = username_input if username_input else "username"
        shown_password = ("*" * len(password_input)) if password_input else "password"
        user_color = COL_TEXT if username_input else COL_TEXT_DIM
        pass_color = COL_TEXT if password_input else COL_TEXT_DIM
        user_text = font_small.render(shown_username, True, user_color)
        pass_text = font_small.render(shown_password, True, pass_color)
        screen.blit(user_text, (username_rect.x + 8, username_rect.y + (username_rect.h - user_text.get_height()) // 2))
        screen.blit(pass_text, (password_rect.x + 8, password_rect.y + (password_rect.h - pass_text.get_height()) // 2))

        auth_color = COL_ACCENT_2 if snapshot.loggedin else COL_WARNING
        auth_status = fit_text(font_small, snapshot.auth_status, 330)
        auth_text = font_small.render(auth_status, True, auth_color)
        screen.blit(auth_text, (header_rect.x + 210, header_rect.y + 62))

        for button in buttons:
            draw_button(screen, button, font_small, button.rect.collidepoint(mouse))

        badge_rect = pygame.Rect(main_rect.x + 26, main_rect.y + 22, 176, 34)
        pygame.draw.rect(screen, (38, 50, 78), badge_rect, border_radius=16)
        badge_color = COL_ACCENT if snapshot.state == STATE_STUDY else COL_WARNING if snapshot.state == STATE_BREAK else COL_MUTED
        pygame.draw.rect(screen, badge_color, badge_rect, width=2, border_radius=16)
        badge_text = font_small.render(snapshot.state, True, COL_TEXT)
        screen.blit(badge_text, (badge_rect.x + (badge_rect.w - badge_text.get_width()) // 2, badge_rect.y + (badge_rect.h - badge_text.get_height()) // 2))

        if snapshot.state in (STATE_STUDY, STATE_BREAK):
            primary_time = snapshot.remaining_s
            secondary_line = f"Elapsed {format_hms(snapshot.elapsed_s)}"
        else:
            primary_time = snapshot.study_minutes * 60
            secondary_line = "Press Start to begin"

        timer_text = font_timer.render(format_hms(primary_time), True, COL_TEXT)
        screen.blit(timer_text, (main_rect.x + 58, main_rect.y + 108))
        secondary_text = font_small.render(secondary_line, True, COL_TEXT_DIM)
        screen.blit(secondary_text, (main_rect.x + 62, main_rect.y + 188))

        ring_center = (main_rect.x + 548, main_rect.y + 206)
        draw_progress_ring(screen, ring_center, 98, snapshot.progress)
        pct = font_h1.render(f"{int(snapshot.progress * 100):02d}%", True, COL_TEXT)
        screen.blit(pct, (ring_center[0] - pct.get_width() // 2, ring_center[1] - pct.get_height() // 2))

        facts = [
            f"Focus in phase: {format_hms(snapshot.focus_s)}",
            f"Completed cycles: {snapshot.cycle_count}",
            f"Offscreen: {snapshot.offscreen_for_s:0.1f}s",
            f"Gaze: {snapshot.gaze_text}",
        ]
        fy = main_rect.y + 250
        for line in facts:
            txt = font_body.render(line, True, COL_TEXT_DIM)
            screen.blit(txt, (main_rect.x + 58, fy))
            fy += 32

        status_title = font_h2.render("Live Sensor Status", True, COL_TEXT)
        screen.blit(status_title, (status_rect.x + 20, status_rect.y + 18))

        cam_rect = pygame.Rect(status_rect.x + 20, status_rect.y + 52, status_rect.w - 40, 176)
        pygame.draw.rect(screen, (14, 21, 36), cam_rect, border_radius=12)
        pygame.draw.rect(screen, (73, 96, 136), cam_rect, width=2, border_radius=12)
        if snapshot.cam_rgb is not None:
            cam_surface = rgb_array_to_surface(snapshot.cam_rgb)
            cam_surface = pygame.transform.smoothscale(cam_surface, (cam_rect.w, cam_rect.h))
            screen.blit(cam_surface, cam_rect.topleft)
        else:
            no_cam = font_body.render("Camera preview unavailable", True, COL_TEXT_DIM)
            screen.blit(no_cam, (cam_rect.x + (cam_rect.w - no_cam.get_width()) // 2, cam_rect.y + (cam_rect.h - no_cam.get_height()) // 2))

        status_color = COL_ACCENT_2 if snapshot.screen_status == "ONSCREEN" else COL_DANGER if snapshot.screen_status in ("OFFSCREEN", "CAMERA ERROR") else COL_MUTED
        pygame.draw.circle(screen, status_color, (status_rect.x + 30, cam_rect.bottom + 24), 7)
        info_lines = [
            f"Eye status: {snapshot.screen_status}",
            f"Study length: {snapshot.study_minutes:0.0f} min",
            f"Break length: {snapshot.break_minutes:0.0f} min",
            "Camera is released automatically on Stop/Quit",
        ]
        iy = cam_rect.bottom + 12
        for idx, line in enumerate(info_lines):
            txt = font_small.render(line, True, COL_TEXT if idx == 0 else COL_TEXT_DIM)
            screen.blit(txt, (status_rect.x + 46, iy + idx * 26))

        settings_title = font_h2.render("Session Settings", True, COL_TEXT)
        settings_sub = font_small.render("Adjust values while stopped", True, COL_TEXT_DIM)
        screen.blit(settings_title, (settings_rect.x + 20, settings_rect.y + 14))
        screen.blit(settings_sub, (settings_rect.x + 20, settings_rect.y + 40))

        rows = [
            ("Study duration", f"{snapshot.study_minutes:0.0f} min"),
            ("Break duration", f"{snapshot.break_minutes:0.0f} min"),
            ("Alarm after", f"{snapshot.alarm_after_s:0.0f} s"),
            ("Reset after offscreen", f"{snapshot.reset_after_s:0.0f} s"),
        ]
        row_y = settings_rect.y + SETTINGS_ROW_TOP_OFFSET
        row_h = SETTINGS_ROW_HEIGHT
        value_w, value_h = SETTINGS_VALUE_SIZE
        for idx, (label, value) in enumerate(rows):
            y = row_y + idx * row_h
            ltxt = font_body.render(label, True, COL_TEXT_DIM)
            label_y = y + (row_h - ltxt.get_height()) // 2
            screen.blit(ltxt, (settings_rect.x + 28, label_y))
            value_rect = pygame.Rect(
                settings_rect.x + SETTINGS_VALUE_X_OFFSET,
                y + (row_h - value_h) // 2,
                value_w,
                value_h,
            )
            pygame.draw.rect(screen, (18, 27, 44), value_rect, border_radius=8)
            pygame.draw.rect(screen, (69, 90, 130), value_rect, width=2, border_radius=8)
            vtxt = font_value.render(value, True, COL_TEXT)
            screen.blit(vtxt, (value_rect.x + (value_rect.w - vtxt.get_width()) // 2, value_rect.y + (value_rect.h - vtxt.get_height()) // 2))

        hotkeys = font_small.render("Hotkeys: Space start/stop | M mute | T test sound | R reset | Tab switch login field | Enter login", True, COL_TEXT_DIM)
        screen.blit(hotkeys, (UI_MARGIN, WINDOW_H - hotkeys.get_height() - 6))

        if snapshot.muted:
            muted_text = font_small.render("Audio muted", True, COL_WARNING)
            screen.blit(muted_text, (header_rect.right - 140, header_rect.y + 74))

        pygame.display.flip()
        clock.tick(FPS)

    app.shutdown()
    pygame.quit()


if __name__ == "__main__":
    app = App()
    run_gui(app)
