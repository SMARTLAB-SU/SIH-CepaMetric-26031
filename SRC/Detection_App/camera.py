"""
================================================================================
Intel RealSense D455f Camera Pipeline & Box Scene Simulator Worker
================================================================================
Handles non-blocking camera capture, RealSense depth-to-color alignment,
device diagnostics, dynamic start/stop controls, and multi-box simulation.
================================================================================
"""

import time
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False


class CameraDeviceInfo:
    """Helper data structure holding RealSense device metadata."""
    def __init__(self, name="Unknown", serial="N/A", firmware="N/A", usb_type="N/A", depth_scale=0.001):
        self.name = name
        self.serial = serial
        self.firmware = firmware
        self.usb_type = usb_type
        self.depth_scale = depth_scale


class MultiObjectSceneSimulator:
    """
    Simulates realistic scenes with physical box-shaped objects:
    Cardboard boxes, wooden boxes, square boxes, and rectangular packaging.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.table_depth_mm = 850  # 85 cm supporting surface

        class MockIntrinsics:
            def __init__(self, w, h):
                self.width = w
                self.height = h
                self.fx = 385.0
                self.fy = 385.0
                self.ppx = w / 2.0
                self.ppy = h / 2.0
                self.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.intrinsics = MockIntrinsics(width, height)

    def generate_frame(self):
        """Generates synchronized RGB frame and 16-bit depth frame with multiple box objects."""
        time.sleep(1.0 / self.fps)

        # 1. Base Supporting Surface (Table / Floor) at 850 mm depth
        depth_raw = np.full((self.height, self.width), self.table_depth_mm, dtype=np.uint16)
        color_bgr = np.full((self.height, self.width, 3), (210, 215, 220), dtype=np.uint8)

        # Subtle grid texture on table surface
        for y in range(0, self.height, 40):
            cv2.line(color_bgr, (0, y), (self.width, y), (185, 190, 195), 1)
        for x in range(0, self.width, 40):
            cv2.line(color_bgr, (x, 0), (x, self.height), (185, 190, 195), 1)

        # 2. OBJECT 1: CARDBOARD BOX (approx 320 x 210 x 160 mm) at 520 mm depth
        b1_x, b1_y, b1_w, b1_h = 60, 70, 185, 130
        depth_raw[b1_y:b1_y + b1_h, b1_x:b1_x + b1_w] = 520
        # Cardboard brown texture
        cv2.rectangle(color_bgr, (b1_x, b1_y), (b1_x + b1_w, b1_y + b1_h), (60, 120, 190), -1)
        cv2.rectangle(color_bgr, (b1_x, b1_y), (b1_x + b1_w, b1_y + b1_h), (30, 60, 100), 2)
        # Packaging tape
        cv2.line(color_bgr, (b1_x, b1_y + b1_h // 2), (b1_x + b1_w, b1_y + b1_h // 2), (180, 170, 130), 3)
        cv2.putText(color_bgr, "CARDBOARD BOX", (b1_x + 8, b1_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # 3. OBJECT 2: WOODEN BOX (approx 310 x 220 x 140 mm) at 560 mm depth
        w_x, w_y, w_w, w_h = 410, 80, 175, 135
        depth_raw[w_y:w_y + w_h, w_x:w_x + w_w] = 560
        # Wood grain tan color
        cv2.rectangle(color_bgr, (w_x, w_y), (w_x + w_w, w_y + w_h), (40, 100, 160), -1)
        cv2.rectangle(color_bgr, (w_x, w_y), (w_x + w_w, w_y + w_h), (20, 50, 90), 2)
        # Wood plank lines
        for py in range(w_y + 30, w_y + w_h, 35):
            cv2.line(color_bgr, (w_x, py), (w_x + w_w, py), (30, 70, 120), 1)
        cv2.putText(color_bgr, "WOODEN BOX", (w_x + 8, w_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # 4. OBJECT 3: SQUARE BOX (approx 170 x 170 x 120 mm) at 540 mm depth
        sq_x, sq_y, sq_s = 80, 280, 140
        depth_raw[sq_y:sq_y + sq_s, sq_x:sq_x + sq_s] = 540
        # Green square package
        cv2.rectangle(color_bgr, (sq_x, sq_y), (sq_x + sq_s, sq_y + sq_s), (60, 165, 85), -1)
        cv2.rectangle(color_bgr, (sq_x, sq_y), (sq_x + sq_s, sq_y + sq_s), (25, 95, 45), 2)
        cv2.putText(color_bgr, "SQUARE BOX", (sq_x + 8, sq_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # 5. OBJECT 4: RECTANGULAR PARCEL (approx 280 x 190 x 130 mm) at 500 mm depth
        r_x, r_y, r_w, r_h = 390, 285, 180, 130
        depth_raw[r_y:r_y + r_h, r_x:r_x + r_w] = 500
        # Blue rectangular carton
        cv2.rectangle(color_bgr, (r_x, r_y), (r_x + r_w, r_y + r_h), (180, 110, 45), -1)
        cv2.rectangle(color_bgr, (r_x, r_y), (r_x + r_w, r_y + r_h), (110, 60, 20), 2)
        cv2.line(color_bgr, (r_x + r_w // 2, r_y), (r_x + r_w // 2, r_y + r_h), (220, 210, 180), 2)
        cv2.putText(color_bgr, "RECT BOX", (r_x + 8, r_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

        # Realistic depth sensor noise (+- 0.8 mm)
        noise = np.random.normal(0, 0.8, depth_raw.shape).astype(np.int16)
        noisy_depth = np.clip(depth_raw.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

        return color_bgr, noisy_depth, self.intrinsics


class ConveyorMovingBoxSimulator:
    """
    Simulates a continuous conveyor belt with multiple cuboid packaging boxes
    moving continuously along the conveyor belt across the entire camera field of view.
    """
    def __init__(self, width=640, height=480, fps=30, speed_px=4.5):
        self.width = width
        self.height = height
        self.fps = fps
        self.speed_px = speed_px
        self.table_depth_mm = 850  # Conveyor surface at 850 mm depth
        self.frame_idx = 0

        # Optical intrinsics (D455f)
        class MockIntrinsics:
            def __init__(self, w, h):
                self.width = w
                self.height = h
                self.fx = 385.0
                self.fy = 385.0
                self.ppx = w / 2.0
                self.ppy = h / 2.0
                self.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.intrinsics = MockIntrinsics(width, height)

        # Sequence of distinct physical boxes moving on the conveyor
        self.box_defs = [
            # Box 1: Cardboard Box ~320x210x150 mm
            {"x": -180.0, "y": 240, "w": 176, "h": 116, "h_mm": 150, "color": (55, 115, 185), "name": "CARDBOARD A"},
            # Box 2: Wooden Box ~400x250x180 mm
            {"x": -520.0, "y": 235, "w": 230, "h": 144, "h_mm": 180, "color": (40, 95, 155), "name": "WOODEN BOX"},
            # Box 3: Square Box ~220x220x130 mm
            {"x": -820.0, "y": 245, "w": 136, "h": 136, "h_mm": 130, "color": (50, 155, 75), "name": "SQUARE BOX"},
            # Box 4: Shipping Carton ~350x220x160 mm
            {"x": -1140.0, "y": 240, "w": 195, "h": 123, "h_mm": 160, "color": (175, 105, 45), "name": "CARDBOARD B"},
        ]
        self.loop_span = 1400.0  # Reset span for continuous loop

    def generate_frame(self):
        """Generates aligned RGB and 16-bit depth map with moving conveyor packages."""
        time.sleep(1.0 / self.fps)
        self.frame_idx += 1

        # Conveyor belt base at 850 mm
        depth_raw = np.full((self.height, self.width), self.table_depth_mm, dtype=np.uint16)
        color_bgr = np.full((self.height, self.width, 3), (210, 215, 220), dtype=np.uint8)

        # Conveyor boundaries & moving textured slats
        belt_top = int(self.height * 0.12)
        belt_bot = int(self.height * 0.88)
        color_bgr[:belt_top, :] = (150, 155, 160)
        color_bgr[belt_bot:, :] = (150, 155, 160)
        cv2.line(color_bgr, (0, belt_top), (self.width, belt_top), (80, 85, 90), 2)
        cv2.line(color_bgr, (0, belt_bot), (self.width, belt_bot), (80, 85, 90), 2)

        # Moving belt texture lines
        offset_x = int((self.frame_idx * self.speed_px) % 40)
        for x in range(-40 + offset_x, self.width + 40, 40):
            cv2.line(color_bgr, (x, belt_top), (x, belt_bot), (195, 200, 205), 1)

        # Draw moving boxes
        for b in self.box_defs:
            cur_x = (b["x"] + self.frame_idx * self.speed_px) % self.loop_span - 200
            bx = int(cur_x)
            by = int(b["y"] - b["h"] // 2)
            bw = int(b["w"])
            bh = int(b["h"])

            # Check if box is inside viewport
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(self.width, bx + bw)
            y2 = min(self.height, by + bh)

            if x2 > x1 and y2 > y1:
                box_top_depth = self.table_depth_mm - b["h_mm"]
                depth_raw[y1:y2, x1:x2] = box_top_depth

                # Render box face with crisp borders
                cv2.rectangle(color_bgr, (bx, by), (bx + bw, by + bh), b["color"], -1)
                # Outer edge line (simulating packaging edge)
                cv2.rectangle(color_bgr, (bx, by), (bx + bw, by + bh), (25, 30, 35), 2)
                # Box tape / top seam
                mid_y = by + bh // 2
                cv2.line(color_bgr, (bx, mid_y), (bx + bw, mid_y), (200, 190, 140), 2)

                if bx > 0 and bx + bw < self.width:
                    cv2.putText(color_bgr, b["name"], (bx + 8, by + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        # Add realistic sensor depth noise (+- 0.8 mm)
        noise = np.random.normal(0, 0.8, depth_raw.shape).astype(np.int16)
        noisy_depth = np.clip(depth_raw.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

        return color_bgr, noisy_depth, self.intrinsics


class RealSenseThread(QThread):
    """
    QThread worker for continuous, asynchronous capture from Intel RealSense D455f
    or continuous high-fidelity conveyor simulation.
    """
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, object)
    status_changed = pyqtSignal(str, bool, object)
    error_occurred = pyqtSignal(str)

    COLORMAPS = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "OCEAN": cv2.COLORMAP_OCEAN,
        "BONE": cv2.COLORMAP_BONE,
    }

    def __init__(self, width=640, height=480, fps=30, force_simulate=False):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.force_simulate = force_simulate
        self.is_running = False
        self.is_paused = False
        self.is_connected = False
        self.current_colormap_key = "TURBO"
        self.max_depth_visualization_mm = 1600

        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_intrinsics = None
        self.device_info = CameraDeviceInfo()

        # Conveyor Multi-Box Moving Scene Simulator
        self.simulator = ConveyorMovingBoxSimulator(width, height, fps)

    def pause_stream(self):
        self.is_paused = True

    def resume_stream(self):
        self.is_paused = False

    def set_colormap(self, colormap_name):
        if colormap_name in self.COLORMAPS:
            self.current_colormap_key = colormap_name

    def colorize_depth(self, depth_raw):
        if depth_raw is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        depth_clipped = np.clip(depth_raw, 200, self.max_depth_visualization_mm)
        depth_scaled = cv2.convertScaleAbs(depth_clipped, alpha=(255.0 / self.max_depth_visualization_mm))
        depth_inverted = 255 - depth_scaled

        cmap_flag = self.COLORMAPS.get(self.current_colormap_key, cv2.COLORMAP_TURBO)
        colorized = cv2.applyColorMap(depth_inverted, cmap_flag)
        colorized[depth_raw == 0] = [0, 0, 0]
        return colorized

    def _query_hardware_device(self):
        if not PYREALSENSE_AVAILABLE:
            return None

        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                dev = devices[0]
                name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Intel RealSense D455f"
                serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "Unknown"
                fw = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else "Unknown"
                usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "3.x"
                return CameraDeviceInfo(name=name, serial=serial, firmware=fw, usb_type=usb)
        except Exception as e:
            print(f"[RealSenseThread] Device query error: {e}")
        return None

    def _init_hardware_pipeline(self):
        if not PYREALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed.")

        # Clean up any existing pipeline
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        try:
            profile = self.pipeline.start(self.config)
        except Exception as start_err:
            print(f"[RealSenseThread] Standard config failed ({start_err}), attempting compatible config...")
            self.config = rs.config()
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(self.config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.device_info.depth_scale = depth_sensor.get_depth_scale()

        self.align = rs.align(rs.stream.color)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.depth_intrinsics = color_stream.get_intrinsics()

        return True

    def reconnect(self):
        """Attempts to discover and connect to physical RealSense camera."""
        print("[*] Attempting camera reconnection...")
        if not PYREALSENSE_AVAILABLE:
            msg = "pyrealsense2 is not installed. Running in simulation mode."
            self.status_changed.emit(msg, False, CameraDeviceInfo(name="Simulator", serial="SIM-001"))
            return False

        dev_info = self._query_hardware_device()
        if dev_info is None:
            msg = "No physical RealSense camera detected. Check USB connection."
            print(f"[WARN] {msg}")
            self.error_occurred.emit(msg)
            return False

        try:
            self.device_info = dev_info
            self._init_hardware_pipeline()
            self.is_connected = True
            self.is_paused = False
            status_msg = f"{self.device_info.name} Connected (USB {self.device_info.usb_type})"
            self.status_changed.emit(status_msg, True, self.device_info)
            print(f"[+] Reconnection successful: {status_msg}")
            return True
        except Exception as e:
            err_msg = f"Failed to initialize camera pipeline: {e}"
            print(f"[ERROR] {err_msg}")
            self.error_occurred.emit(err_msg)
            self.is_connected = False
            return False

    def run(self):
        self.is_running = True

        dev_info = None if self.force_simulate else self._query_hardware_device()

        if dev_info is not None:
            self.device_info = dev_info
            try:
                self._init_hardware_pipeline()
                self.is_connected = True
                status_msg = f"{self.device_info.name} Connected (USB {self.device_info.usb_type})"
                self.status_changed.emit(status_msg, True, self.device_info)
                print(f"[+] RealSense Pipeline active: {status_msg}")
            except Exception as e:
                err_msg = f"Failed to start D455f pipeline: {e}"
                print(f"[WARN] {err_msg}. Falling back to simulation...")
                self.error_occurred.emit(err_msg)
                self.is_connected = False
        else:
            self.is_connected = False
            status_msg = "Multi-Box Simulator Active (Offline Mode)"
            self.status_changed.emit(status_msg, False, CameraDeviceInfo(name="RealSense D455f Simulator", serial="SIM-001"))
            print(f"[*] {status_msg}")

        consecutive_errors = 0
        while self.is_running:
            if self.is_paused:
                time.sleep(0.08)
                continue

            try:
                if self.is_connected and self.pipeline is not None:
                    frames = self.pipeline.wait_for_frames(2500)
                    aligned_frames = self.align.process(frames)

                    aligned_depth = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not aligned_depth or not color_frame:
                        continue

                    intrinsics = aligned_depth.profile.as_video_stream_profile().get_intrinsics()
                    self.depth_intrinsics = intrinsics

                    depth_raw = np.asanyarray(aligned_depth.get_data())
                    color_bgr = np.asanyarray(color_frame.get_data())
                    consecutive_errors = 0
                else:
                    color_bgr, depth_raw, intrinsics = self.simulator.generate_frame()
                    self.depth_intrinsics = intrinsics

                depth_colormap = self.colorize_depth(depth_raw)
                self.frame_ready.emit(color_bgr, depth_colormap, depth_raw, intrinsics)

            except Exception as e:
                if self.is_connected:
                    consecutive_errors += 1
                    print(f"[RealSenseThread] Frame error: {e}")
                    if consecutive_errors >= 5:
                        err_msg = f"Lost connection to camera: {e}"
                        self.error_occurred.emit(err_msg)
                        self.is_connected = False
                        self.status_changed.emit("Camera Disconnected. Switched to Simulator.", False, CameraDeviceInfo(name="RealSense D455f Simulator", serial="SIM-001"))
                time.sleep(0.05)

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                print("[*] RealSense Pipeline stopped cleanly.")
            except Exception:
                pass
            self.pipeline = None

    def stop(self):
        self.is_running = False
        self.wait(1500)
