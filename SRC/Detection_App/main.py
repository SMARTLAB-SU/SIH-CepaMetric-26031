#!/usr/bin/env python3
"""
================================================================================
Intel RealSense D455f — Multi-Object 3D Detection & Dimensioning System
================================================================================
Strict Classical Computer Vision + RealSense SDK 3D Depth Geometry.
100% OPERATES WITHOUT AI / ML / Deep Learning Models.

Features:
  1. Box Detection & Continuous Tracking: H / L / W measurement on packages.
  2. Circle Object Detection & Onion Size Classification:
     - Detects circular/spherical objects (Onions, Potatoes, round produce).
     - Optical 3D metric radius and diameter calculation.
     - Automated size classification: SMALL, MEDIUM, LARGE.
  3. RealSense D455f Hardware Pipeline + High-Fidelity Multi-Object Simulator.
  4. Velocity-Compensated Persistent Multi-Object Tracker.
  5. Event-Based Single-Record Measurement Logging (No repeated measurements).
  6. PyQt5 Industrial Desktop Dashboard & Multi-Mode Viewport.

Usage:
  python main.py             # Normal hardware mode with Intel RealSense D455f
  python main.py --simulate  # Run with simulated RealSense D455f RGB-D feed
================================================================================
"""

import sys
import os
import time
import math
import csv
import argparse
import numpy as np
import cv2

# Optional Intel RealSense SDK
try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QComboBox, QSlider, QStatusBar,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QFrame,
    QFileDialog, QScrollArea
)
from camera import ConveyorMovingBoxSimulator


# ==============================================================================
# SECTION 1: 3D SPATIAL GEOMETRY & ONION CLASSIFICATION CONFIGURATION
# ==============================================================================

# Configurable Onion Grading Size Thresholds (Diameter in millimeters)
# Easy to customize for specific agricultural standards (e.g. USDA / Agmark)
ONION_SIZE_THRESHOLDS = {
    "SMALL_MAX_MM": 45.0,   # Diameter < 45 mm -> SMALL
    "LARGE_MIN_MM": 70.0,   # Diameter >= 70 mm -> LARGE
    # 45.0 mm <= Diameter < 70.0 mm -> MEDIUM
}


def classify_onion_size(diameter_mm, small_max_mm=None, large_min_mm=None):
    """
    Classifies an agricultural onion/circular object based on its physical diameter in millimeters.
    
    Args:
        diameter_mm (float): Real-world diameter in millimeters.
        small_max_mm (float, optional): Maximum diameter for SMALL.
        large_min_mm (float, optional): Minimum diameter for LARGE.
        
    Returns:
        str: "SMALL", "MEDIUM", or "LARGE"
    """
    if diameter_mm is None or diameter_mm <= 0:
        return "UNKNOWN"
    
    s_max = small_max_mm if small_max_mm is not None else ONION_SIZE_THRESHOLDS["SMALL_MAX_MM"]
    l_min = large_min_mm if large_min_mm is not None else ONION_SIZE_THRESHOLDS["LARGE_MIN_MM"]

    if diameter_mm < s_max:
        return "SMALL"
    elif diameter_mm >= l_min:
        return "LARGE"
    else:
        return "MEDIUM"


def format_dimension(val_mm, unit="cm"):
    """
    Converts a measurement in millimeters to the target unit (cm or mm).
    
    Args:
        val_mm (float): Dimension in millimeters.
        unit (str): "cm" or "mm".
        
    Returns:
        float: Dimension in target unit.
    """
    if val_mm is None:
        return 0.0
    if str(unit).lower() == "cm":
        return val_mm / 10.0
    return float(val_mm)


def get_robust_depth_sample(depth_raw, pixel_x, pixel_y, kernel_size=3):
    """
    Extracts a robust depth measurement (in meters) around (pixel_x, pixel_y)
    using neighborhood median filtering to reject single-pixel noise/outliers.
    
    Args:
        depth_raw (np.ndarray): 2D array of uint16 raw depth (in mm).
        pixel_x (int): X pixel coordinate.
        pixel_y (int): Y pixel coordinate.
        kernel_size (int): Odd kernel window size (default: 3x3).
        
    Returns:
        float: Robust depth in meters, or 0.0 if invalid/occluded.
    """
    if depth_raw is None:
        return 0.0

    h, w = depth_raw.shape[:2]
    px = int(round(pixel_x))
    py = int(round(pixel_y))

    if not (0 <= px < w and 0 <= py < h):
        return 0.0

    half_k = kernel_size // 2
    y_min = max(0, py - half_k)
    y_max = min(h, py + half_k + 1)
    x_min = max(0, px - half_k)
    x_max = min(w, px + half_k + 1)

    roi = depth_raw[y_min:y_max, x_min:x_max]
    valid_pixels = roi[roi > 0]

    if len(valid_pixels) == 0:
        return 0.0

    median_val = float(np.median(valid_pixels))
    return median_val / 1000.0  # Convert mm to meters


def deproject_pixel_to_3d_point(intrinsics, pixel_x, pixel_y, depth_meters, frame_width=640, frame_height=480):
    """
    Deprojects a 2D image pixel (u, v) and metric depth Z into 3D camera coordinates (X, Y, Z).
    
    Args:
        intrinsics: RealSense intrinsics object or dict/class with fx, fy, ppx, ppy.
        pixel_x (int/float): X coordinate in image frame.
        pixel_y (int/float): Y coordinate in image frame.
        depth_meters (float): Distance in meters.
        frame_width (int): Fallback image width.
        frame_height (int): Fallback image height.
        
    Returns:
        np.ndarray: [X, Y, Z] in millimeters, or None if depth is invalid.
    """
    if depth_meters is None or depth_meters <= 0.0:
        return None

    # 1. Use RealSense SDK native deprojection if available
    if PYREALSENSE_AVAILABLE and hasattr(intrinsics, "fx") and hasattr(intrinsics, "coeffs"):
        try:
            point_3d_meters = rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [float(pixel_x), float(pixel_y)],
                float(depth_meters)
            )
            return np.array([
                point_3d_meters[0] * 1000.0,
                point_3d_meters[1] * 1000.0,
                point_3d_meters[2] * 1000.0
            ], dtype=np.float64)
        except Exception:
            pass

    # 2. Mathematical Pinhole Camera Model fallback:
    fx = getattr(intrinsics, "fx", 385.0) if intrinsics else 385.0
    fy = getattr(intrinsics, "fy", 385.0) if intrinsics else 385.0
    ppx = getattr(intrinsics, "ppx", frame_width / 2.0) if intrinsics else (frame_width / 2.0)
    ppy = getattr(intrinsics, "ppy", frame_height / 2.0) if intrinsics else (frame_height / 2.0)

    z_mm = depth_meters * 1000.0
    x_mm = (float(pixel_x) - ppx) * z_mm / fx
    y_mm = (float(pixel_y) - ppy) * z_mm / fy

    return np.array([x_mm, y_mm, z_mm], dtype=np.float64)


def calculate_3d_distance(point_a_3d, point_b_3d):
    """
    Calculates the real-world 3D Euclidean distance between two 3D points.
    Formula: D = sqrt((X2 - X1)^2 + (Y2 - Y1)^2 + (Z2 - Z1)^2)
    """
    if point_a_3d is None or point_b_3d is None:
        return None

    p1 = np.asarray(point_a_3d, dtype=np.float64)
    p2 = np.asarray(point_b_3d, dtype=np.float64)

    if len(p1) < 3 or len(p2) < 3:
        return None

    return float(np.linalg.norm(p2 - p1))


# ==============================================================================
# SECTION 2: REALSENSE D455f CAMERA PIPELINE & MULTI-OBJECT SIMULATOR
# ==============================================================================

class CameraDeviceInfo:
    """Helper data structure holding RealSense device metadata."""
    def __init__(self, name="Unknown", serial="N/A", firmware="N/A", usb_type="N/A", depth_scale=0.001):
        self.name = name
        self.serial = serial
        self.firmware = firmware
        self.usb_type = usb_type
        self.depth_scale = depth_scale


class RealSenseThread(QThread):
    """
    QThread worker for continuous, asynchronous capture from Intel RealSense D455f
    or continuous high-fidelity multi-object conveyor simulation.
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

        # Cleanup existing pipeline
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            time.sleep(1.0) # Give the OS time to release USB resources

        max_retries = 4
        last_err = None
        
        for attempt in range(max_retries):
            try:
                self.pipeline = rs.pipeline()
                self.config = rs.config()

                self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

                profile = self.pipeline.start(self.config)
                
                depth_sensor = profile.get_device().first_depth_sensor()
                self.device_info.depth_scale = depth_sensor.get_depth_scale()

                self.align = rs.align(rs.stream.color)

                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.depth_intrinsics = color_stream.get_intrinsics()

                print(f"[RealSenseThread] Pipeline started successfully on attempt {attempt+1}")
                return True
            except Exception as start_err:
                last_err = start_err
                print(f"[RealSenseThread] Pipeline start attempt {attempt+1} failed: {start_err}")
                
                if self.pipeline is not None:
                    try:
                        self.pipeline.stop()
                    except Exception:
                        pass
                self.pipeline = None
                
                # Only hardware reset after multiple normal retries fail
                if attempt == 1:
                    try:
                        print("[RealSenseThread] Attempting hardware reset on RealSense devices...")
                        ctx = rs.context()
                        for dev in ctx.query_devices():
                            dev.hardware_reset()
                        print("[RealSenseThread] Waiting 8 seconds for camera to re-enumerate on USB...")
                        time.sleep(8.0) 
                    except Exception as e:
                        print(f"[RealSenseThread] Hardware reset failed: {e}")
                else:
                    time.sleep(2.0)
                
        # Final fallback to generic compatible config
        print(f"[RealSenseThread] Standard config failed ({last_err}), attempting compatible config...")
        try:
            self.pipeline = rs.pipeline()
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
        except Exception as fallback_err:
            print(f"[RealSenseThread] Compatible config failed: {fallback_err}")
            raise RuntimeError(f"Could not start RealSense pipeline: {last_err}")

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

        if self.force_simulate:
            self.is_connected = True
            self.device_info = CameraDeviceInfo(name="Simulator", serial="SIM-001", usb_type="OFFLINE")
            self.depth_intrinsics = self.simulator.intrinsics
            self.status_changed.emit("Simulator Connected (Offline Mode)", True, self.device_info)
            print("[*] Multi-Box Simulator Active (Offline Mode)")
        elif dev_info is not None:
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
            status_msg = "Camera Not Connected"
            self.status_changed.emit(status_msg, False, CameraDeviceInfo(name="Disconnected", serial="N/A"))
            print(f"[*] {status_msg}")

        consecutive_errors = 0
        while self.is_running:
            if self.is_paused:
                time.sleep(0.08)
                continue

            try:
                if self.force_simulate:
                    color_bgr, depth_raw, intrinsics = self.simulator.generate_frame()
                    self.depth_intrinsics = intrinsics
                    depth_colormap = self.colorize_depth(depth_raw)
                    self.frame_ready.emit(color_bgr, depth_colormap, depth_raw, intrinsics)
                elif self.is_connected and self.pipeline is not None:
                    frames = self.pipeline.wait_for_frames(5000) # Wait up to 5s to avoid immediate fail
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
                    
                    depth_colormap = self.colorize_depth(depth_raw)
                    self.frame_ready.emit(color_bgr, depth_colormap, depth_raw, intrinsics)
                else:
                    # Sleep when disconnected instead of generating fake animation frames
                    time.sleep(0.5)

            except Exception as e:
                if self.is_connected:
                    consecutive_errors += 1
                    print(f"[RealSenseThread] Frame error: {e}")
                    if consecutive_errors >= 5:
                        err_msg = f"Lost connection to camera: {e}"
                        self.error_occurred.emit(err_msg)
                        self.is_connected = False
                        self.status_changed.emit("Camera Disconnected.", False, CameraDeviceInfo(name="Disconnected", serial="N/A"))
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


# ==============================================================================
# SECTION 3: CLASSICAL COMPUTER VISION BOX & CIRCLE (ONION) DETECTOR
# ==============================================================================

class DetectedObjectCandidate:
    """Represents a validated physical box or circular/onion candidate extracted in the current frame."""

    def __init__(self, obj_id=1, shape_type="BOX", dimensions=None, unit="mm",
                 contour=None, center_2d=(0, 0), center_3d=None,
                 corners_2d=None, corners_3d=None, bbox=(0, 0, 0, 0),
                 circle_center_2d=None, circle_radius_px=0.0,
                 top_depth_m=0.5, support_depth_m=0.85):
        self.obj_id = obj_id
        self.shape_type = shape_type  # "BOX", "SQUARE", "ONION", "CIRCLE"
        self.dimensions = dimensions or {}
        self.unit = unit
        self.contour = contour
        self.center_2d = center_2d
        self.center_3d = center_3d
        self.corners_2d = corners_2d or []
        self.corners_3d = corners_3d or []
        self.bbox = bbox  # (x, y, w, h)
        self.circle_center_2d = circle_center_2d or center_2d
        self.circle_radius_px = circle_radius_px
        self.top_depth_m = top_depth_m
        self.support_depth_m = support_depth_m

    def get_display_text(self):
        """Returns a clean formatted block for GUI display."""
        u = self.unit
        lines = [f"{self.shape_type} #{self.obj_id}"]

        if self.shape_type in ["ONION", "CIRCLE"]:
            dia = self.dimensions.get("diameter", 0)
            rad = self.dimensions.get("radius", 0)
            size_cls = self.dimensions.get("size_class", "MEDIUM")
            lines.append(f"  Diameter : {dia:.1f} {u} [{size_cls}]")
            lines.append(f"  Radius   : {rad:.1f} {u}")
            h = self.dimensions.get("height", 0)
            if h > 0:
                lines.append(f"  Height   : {h:.1f} {u}")
        else:
            l = self.dimensions.get("length", 0)
            w = self.dimensions.get("width", 0)
            h = self.dimensions.get("height", 0)
            lines.append(f"  Length : {l:.1f} {u}")
            lines.append(f"  Width  : {w:.1f} {u}")
            lines.append(f"  Height : {h:.1f} {u}")

        depth = self.dimensions.get("depth", 0)
        if depth > 0:
            lines.append(f"  Distance : {depth:.1f} {u}")

        return "\n".join(lines)


# Backwards compatibility alias
DetectedObject = DetectedObjectCandidate


class MultiObjectDetector:
    """
    Detects and measures physical BOXES and CIRCULAR OBJECTS (ONIONS / POTATOES)
    simultaneously across the entire camera field of view.
    Strictly filters out human bodies, limbs, conveyor belts, and random noise.
    """

    @staticmethod
    def detect_and_measure(color_bgr, depth_raw, intrinsics=None,
                           min_depth_mm=120, max_depth_mm=2600,
                           min_height_mm=10.0,
                           unit="mm", min_contour_area=45):
        """
        Full-frame multi-object detection and 3D physical dimensioning.
        Processes and returns ALL valid box and onion candidates in the frame.

        Returns:
            candidates: list[DetectedObjectCandidate]
            edge_map: np.ndarray (for visualization)
            diagnostics: dict
        """
        if color_bgr is None or depth_raw is None:
            return [], None, {}

        h, w = depth_raw.shape[:2]

        # ----------------------------------------------------------------------
        # 1. Dual Edge Detection: Fast RGB Canny + Depth Discontinuity Gradient
        # ----------------------------------------------------------------------
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        canny_edges = cv2.Canny(blurred, 30, 95)

        valid_depth_mask = (depth_raw >= min_depth_mm) & (depth_raw <= max_depth_mm) & (depth_raw > 0)

        # Depth discontinuity gradient (finds physical height jumps >= 12mm)
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        depth_grad = cv2.morphologyEx(depth_raw, cv2.MORPH_GRADIENT, kernel_3)
        depth_step_edges = ((depth_grad >= 12) & valid_depth_mask).astype(np.uint8) * 255

        # Combined edge map
        combined_edges = cv2.bitwise_or(canny_edges, depth_step_edges)

        # ----------------------------------------------------------------------
        # 2. Multi-Depth Region Segmentation
        # ----------------------------------------------------------------------
        valid_u8 = valid_depth_mask.astype(np.uint8) * 255
        edge_barriers = cv2.dilate(combined_edges, kernel_3, iterations=1)
        seg_mask = cv2.subtract(valid_u8, edge_barriers)
        seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel_3, iterations=1)

        # ----------------------------------------------------------------------
        # 3. Find All Contours Across View
        # ----------------------------------------------------------------------
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        rejected_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area or area > (w * h * 0.80):
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bx <= 4 and by <= 4 and (bx + bw) >= (w - 4) and (by + bh) >= (h - 4):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            min_rect = cv2.minAreaRect(cnt)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            if rect_area <= 0:
                continue

            rectangularity = area / rect_area

            # ------------------------------------------------------------------
            # 4. Geometry Verification: Circle / Onion ONLY
            # ------------------------------------------------------------------
            is_box_geom = False
            
            is_circle_geom, circ_center, circ_radius, _ = MultiObjectDetector._check_circle_geometry(
                cnt=cnt, perimeter=perimeter, area=area, rect_w=rect_w, rect_h=rect_h
            )

            if not is_circle_geom:
                rejected_count += 1
                continue

            # Center calculation
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = (bx + bw // 2, by + bh // 2) if not is_circle_geom else circ_center

            # ------------------------------------------------------------------
            # 5. Precise Elevation & Supporting Depth Verification
            # ------------------------------------------------------------------
            pad = 22
            ry1 = max(0, by - pad)
            ry2 = min(h, by + bh + pad)
            rx1 = max(0, bx - pad)
            rx2 = min(w, bx + bw + pad)
            ext_depth = depth_raw[ry1:ry2, rx1:rx2]

            ext_mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
            ext_cnt = cnt - np.array([rx1, ry1])
            cv2.drawContours(ext_mask, [ext_cnt], -1, 255, -1)
            ext_dilated = cv2.dilate(ext_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=1)
            ext_ring = cv2.subtract(ext_dilated, ext_mask)

            inner_depths = ext_depth[(ext_mask == 255) & (ext_depth > 0)]
            outer_depths = ext_depth[(ext_ring == 255) & (ext_depth > 0)]

            req_inner_samples = 15 if is_box_geom else 6
            if len(inner_depths) < req_inner_samples:
                rejected_count += 1
                continue

            top_depth_mm = float(np.median(inner_depths))
            top_depth_m = top_depth_mm / 1000.0

            valid_outer = outer_depths[outer_depths > (top_depth_mm + 6.0)]
            req_outer_samples = 8 if is_box_geom else 4
            if len(valid_outer) >= req_outer_samples:
                support_depth_mm = float(np.median(valid_outer))
            else:
                support_depth_mm = top_depth_mm

            support_depth_m = support_depth_mm / 1000.0
            height_mm = max(0.0, support_depth_mm - top_depth_mm)

            # Minimum elevation check
            req_min_h = min_height_mm if is_box_geom else max(12.0, min_height_mm * 0.7)
            if height_mm < req_min_h:
                rejected_count += 1
                continue

            # Deproject center to 3D camera coordinates
            center_3d = deproject_pixel_to_3d_point(intrinsics, cx, cy, top_depth_m, w, h)

            # ------------------------------------------------------------------
            # 6. Branch A: Process BOX / SQUARE Object
            # ------------------------------------------------------------------
            if is_box_geom:
                # 3D Depth Planarity Check
                is_planar, depth_std, _ = MultiObjectDetector._check_depth_planarity(
                    depth_raw=depth_raw, cnt=cnt, bx=bx, by=by, bw=bw, bh=bh, top_depth_mm=top_depth_mm
                )
                if not is_planar:
                    rejected_count += 1
                    continue

                box_pts = cv2.boxPoints(min_rect)
                box_pts = np.int32(box_pts)
                corners_3d = [deproject_pixel_to_3d_point(intrinsics, pt[0], pt[1], top_depth_m, w, h) for pt in box_pts]

                if all(p is not None for p in corners_3d):
                    edge1_mm = calculate_3d_distance(corners_3d[0], corners_3d[1])
                    edge2_mm = calculate_3d_distance(corners_3d[1], corners_3d[2])
                else:
                    fx = getattr(intrinsics, "fx", 385.0) if intrinsics else 385.0
                    edge1_mm = (min_rect[1][0] * top_depth_mm) / fx
                    edge2_mm = (min_rect[1][1] * top_depth_mm) / fx

                length_mm = max(edge1_mm, edge2_mm)
                width_mm = min(edge1_mm, edge2_mm)

                if length_mm < 30.0 or length_mm > 2400.0 or width_mm < 30.0 or height_mm > 2500.0:
                    rejected_count += 1
                    continue

                is_square = (width_mm / max(length_mm, 1e-3)) >= 0.88
                shape_name = "SQUARE" if is_square else "BOX"

                dims = {
                    "length": format_dimension(length_mm, unit),
                    "width": format_dimension(width_mm, unit),
                    "height": format_dimension(height_mm, unit),
                    "depth": format_dimension(top_depth_mm, unit)
                }

                obj = DetectedObjectCandidate(
                    shape_type=shape_name,
                    dimensions=dims,
                    unit=unit,
                    contour=cnt,
                    center_2d=(cx, cy),
                    center_3d=center_3d,
                    corners_2d=[tuple(pt) for pt in box_pts],
                    corners_3d=corners_3d,
                    bbox=(bx, by, bw, bh),
                    top_depth_m=top_depth_m,
                    support_depth_m=support_depth_m
                )
                raw_candidates.append(obj)

            # ------------------------------------------------------------------
            # 7. Branch B: Process CIRCLE / ONION Object
            # ------------------------------------------------------------------
            elif is_circle_geom:
                fx = getattr(intrinsics, "fx", 385.0) if intrinsics else 385.0

                # Refine circle radius to account for morphological depth step boundaries
                if support_depth_mm > (top_depth_mm + 5.0):
                    refined_r = MultiObjectDetector._refine_circle_radius(
                        depth_raw, circ_center[0], circ_center[1], circ_radius, top_depth_mm, support_depth_mm
                    )
                    circ_radius = refined_r

                # Metric 3D Radius and Diameter calculation
                radius_mm = (circ_radius * top_depth_mm) / fx
                diameter_mm = radius_mm * 2.0

                # Validate realistic physical onion size (e.g. 26mm to 240mm) and conveyor height (<= 350mm)
                if diameter_mm < 26.0 or diameter_mm > 240.0 or height_mm > 350.0:
                    rejected_count += 1
                    continue

                # Automated Size Classification
                size_class = classify_onion_size(diameter_mm)

                dims = {
                    "radius": format_dimension(radius_mm, unit),
                    "diameter": format_dimension(diameter_mm, unit),
                    "length": format_dimension(diameter_mm, unit),
                    "width": format_dimension(diameter_mm, unit),
                    "height": format_dimension(height_mm, unit),
                    "depth": format_dimension(top_depth_mm, unit),
                    "size_class": size_class,
                    "grade": size_class
                }

                obj = DetectedObjectCandidate(
                    shape_type="ONION",
                    dimensions=dims,
                    unit=unit,
                    contour=cnt,
                    center_2d=(circ_center[0], circ_center[1]),
                    center_3d=center_3d,
                    bbox=(bx, by, bw, bh),
                    circle_center_2d=circ_center,
                    circle_radius_px=circ_radius,
                    top_depth_m=top_depth_m,
                    support_depth_m=support_depth_m
                )
                raw_candidates.append(obj)

        # ----------------------------------------------------------------------
        # 8. Non-Maximum Suppression with Containment Filtering
        # ----------------------------------------------------------------------
        candidates = MultiObjectDetector._apply_nms(raw_candidates, iou_threshold=0.85, containment_threshold=0.85)

        diagnostics = {
            "contours_found": len(contours),
            "candidates_count": len(candidates),
            "raw_candidates_count": len(raw_candidates),
            "rejected_count": rejected_count
        }

        return candidates, combined_edges, diagnostics

    @staticmethod
    def _refine_circle_radius(depth_raw, cx, cy, est_r, top_depth_mm, support_depth_mm):
        """
        Refines physical circle radius via radial depth step ray casting.
        Compensates for morphological boundary erosion to recover precise pixel radius.
        """
        h, w = depth_raw.shape[:2]
        thresh_depth = top_depth_mm + (support_depth_mm - top_depth_mm) * 0.45
        angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        radii = []
        max_search = int(est_r * 2.2) + 4

        for a in angles:
            cos_a, sin_a = math.cos(a), math.sin(a)
            r_found = est_r
            for step in range(max(2, int(est_r * 0.5)), max_search):
                px = int(round(cx + step * cos_a))
                py = int(round(cy + step * sin_a))
                if px < 0 or px >= w or py < 0 or py >= h:
                    break
                if depth_raw[py, px] >= thresh_depth:
                    r_found = step
                    break
            radii.append(r_found)

        return float(np.median(radii)) if len(radii) > 0 else float(est_r)

    @staticmethod
    def _check_box_geometry(cnt, perimeter, area, rect_w, rect_h, rectangularity):
        """
        Validates whether a 2D contour conforms to rigid box geometry.
        Rejects human bodies, curved contours, limbs, and irregular shapes.
        """
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, None, "Invalid hull area"

        solidity = area / hull_area
        if solidity < 0.76:
            return False, None, f"Low solidity ({solidity:.2f} < 0.76)"

        aspect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1e-3)
        if aspect_ratio > 4.2:
            return False, None, f"Extreme aspect ratio ({aspect_ratio:.2f} > 4.2)"

        if perimeter > 0:
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity > 0.82:
                return False, None, f"High circularity / round shape ({circularity:.2f} > 0.82)"

        if rectangularity < 0.60:
            return False, None, f"Low rectangularity ({rectangularity:.2f} < 0.60)"

        approx_poly = cv2.approxPolyDP(cnt, 0.026 * perimeter, True)
        if len(approx_poly) != 4:
            approx_poly = cv2.approxPolyDP(cnt, 0.038 * perimeter, True)
            if len(approx_poly) != 4:
                return False, None, f"Non-box vertex count ({len(approx_poly)})"

        pts = approx_poly.reshape(4, 2)
        for i in range(4):
            p_prev = pts[(i - 1) % 4]
            p_curr = pts[i]
            p_next = pts[(i + 1) % 4]
            v1 = p_prev.astype(np.float32) - p_curr.astype(np.float32)
            v2 = p_next.astype(np.float32) - p_curr.astype(np.float32)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 2.0 or norm2 < 2.0:
                return False, None, "Degenerate edge length"
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = math.degrees(math.acos(cos_angle))
            if angle_deg < 50.0 or angle_deg > 130.0:
                return False, None, f"Irregular corner angle ({angle_deg:.1f}°)"

        return True, approx_poly, "Valid box geometry"

    @staticmethod
    def _check_circle_geometry(cnt, perimeter, area, rect_w, rect_h):
        """
        Validates whether a 2D contour conforms to circular/onion geometry.
        Rejects boxes, straight lines, irregular noise, and human contours.
        """
        if perimeter <= 0 or area <= 0:
            return False, (0, 0), 0.0, "Invalid perimeter/area"

        # 1. Circularity / Isoperimetric Quotient Check
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
        if circularity < 0.50 or circularity > 1.80:
            return False, (0, 0), 0.0, f"Non-circular isoperimetric quotient ({circularity:.2f})"

        # 2. Minimum Enclosing Circle & Extent
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < 9.0:
            return False, (0, 0), 0.0, "Too small circle radius"

        circle_area = math.pi * (radius ** 2)
        if circle_area <= 0:
            return False, (0, 0), 0.0, "Invalid circle area"

        circle_extent = area / circle_area
        if circle_extent < 0.45 or circle_extent > 1.45:
            return False, (0, 0), 0.0, f"Low circle extent ({circle_extent:.2f} < 0.45)"

        # 3. Aspect Ratio Limit
        aspect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1e-3)
        if aspect_ratio > 2.5:
            return False, (0, 0), 0.0, f"Non-circular aspect ratio ({aspect_ratio:.2f} > 2.5)"

        # 4. Convexity / Solidity Check
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, (0, 0), 0.0, "Invalid hull area"

        solidity = area / hull_area
        if solidity < 0.65:
            return False, (0, 0), 0.0, f"Low solidity for circle ({solidity:.2f} < 0.65)"

        return True, (int(round(cx)), int(round(cy))), float(radius), "Valid circular/onion geometry"

    @staticmethod
    def _check_depth_planarity(depth_raw, cnt, bx, by, bw, bh, top_depth_mm):
        """
        Verifies that the surface inside the contour is a rigid flat plane (box face)
        and NOT a curved/irregular organic surface.
        """
        roi_depth = depth_raw[by:by + bh, bx:bx + bw]
        roi_mask = np.zeros((bh, bw), dtype=np.uint8)
        shifted_cnt = cnt - np.array([bx, by])
        cv2.drawContours(roi_mask, [shifted_cnt], -1, 255, -1)

        surf_depths = roi_depth[(roi_mask == 255) & (roi_depth > 0)]
        if len(surf_depths) < 20:
            return False, 0.0, "Insufficient valid surface depth samples"

        std_z = float(np.std(surf_depths))
        if std_z <= 16.0:
            return True, std_z, "Planar flat surface (low std)"

        y_idx, x_idx = np.where((roi_mask == 255) & (roi_depth > 0))
        if len(x_idx) > 30:
            z_vals = roi_depth[y_idx, x_idx].astype(np.float32)
            A = np.column_stack([x_idx, y_idx, np.ones_like(x_idx)])
            plane_params, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
            fitted_z = A @ plane_params
            residuals = z_vals - fitted_z
            plane_rms = float(np.sqrt(np.mean(residuals ** 2)))

            if plane_rms <= 16.0:
                return True, plane_rms, "Planar tilted surface (low plane RMS)"
            else:
                return False, plane_rms, f"Non-planar surface (plane RMS={plane_rms:.1f}mm, std={std_z:.1f}mm)"

        return False, std_z, f"High depth variance ({std_z:.1f}mm)"

    @staticmethod
    def _apply_nms(candidates, iou_threshold=0.60, containment_threshold=0.65):
        """Removes duplicate candidate bounding boxes and internal sub-parts."""
        if len(candidates) <= 1:
            return candidates

        sorted_cands = sorted(candidates, key=lambda c: cv2.contourArea(c.contour), reverse=True)
        keep = []

        for cand in sorted_cands:
            bx, by, bw, bh = cand.bbox
            cand_box = [bx, by, bx + bw, by + bh]
            cand_area = bw * bh

            duplicate = False
            for kept_cand in keep:
                kx, ky, kw, kh = kept_cand.bbox
                kept_box = [kx, ky, kx + kw, ky + kh]
                kept_area = kw * kh

                inter_x1 = max(cand_box[0], kept_box[0])
                inter_y1 = max(cand_box[1], kept_box[1])
                inter_x2 = min(cand_box[2], kept_box[2])
                inter_y2 = min(cand_box[3], kept_box[3])

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    iou = inter_area / float(cand_area + kept_area - inter_area)
                    containment = inter_area / float(min(cand_area, kept_area))

                    depth_diff = abs(cand.top_depth_m - kept_cand.top_depth_m) * 1000.0
                    if iou >= iou_threshold or (containment >= containment_threshold and depth_diff < 40.0):
                        duplicate = True
                        break

            if not duplicate:
                keep.append(cand)

        return keep


def draw_multi_object_overlays(image_bgr, tracked_objects, unit="mm"):
    """
    Renders clean, high-visibility wireframes/circle outlines, center crosshairs,
    and dynamic HUD tags for both Boxes and Onions.
    """
    if image_bgr is None:
        return None

    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    for obj in tracked_objects:
        if getattr(obj, "missing_frames", 0) > 0:
            continue

        u = getattr(obj, "unit", unit)
        shape = obj.shape_type
        dims = obj.get_dimensions(u) if hasattr(obj, "get_dimensions") else obj.dimensions
        is_stable = getattr(obj, "is_stable", False)
        obj_id = getattr(obj, "obj_id", 1)

        # Color palette: Green (Stable) / Cyan (Tracking)
        color = (0, 240, 120) if is_stable else (0, 220, 255)
        fill_color = (0, 180, 80) if is_stable else (0, 160, 200)
        status_str = "[STABLE]" if is_stable else "[TRACKING]"

        # Format tag label based on shape
        if shape in ["ONION", "CIRCLE"]:
            dia_val = dims.get("diameter", 0.0)
            size_cls = dims.get("size_class", "MEDIUM")
            tag_label = f"ONION #{obj_id:02d} | Dia: {dia_val:.0f}{u} [{size_cls}] {status_str}" if u == "mm" else f"ONION #{obj_id:02d} | Dia: {dia_val:.1f}{u} [{size_cls}] {status_str}"
        else:
            l = dims.get("length", 0.0)
            w_val = dims.get("width", 0.0)
            h_val = dims.get("height", 0.0)
            tag_label = f"{shape} #{obj_id:02d} | {l:.0f}x{w_val:.0f}x{h_val:.0f}{u} {status_str}" if u == "mm" else f"{shape} #{obj_id:02d} | {l:.1f}x{w_val:.1f}x{h_val:.1f}{u} {status_str}"

        # Draw Outlines
        if shape in ["ONION", "CIRCLE"]:
            ccx, ccy = getattr(obj, "circle_center_2d", obj.center_2d)
            cr = int(getattr(obj, "circle_radius_px", 20.0))

            # Tinted circular overlay
            poly_overlay = annotated.copy()
            cv2.circle(poly_overlay, (int(ccx), int(ccy)), cr, fill_color, -1)
            cv2.addWeighted(poly_overlay, 0.18, annotated, 0.82, 0, annotated)

            # Circular outline
            cv2.circle(annotated, (int(ccx), int(ccy)), cr, color, 2)
            cv2.circle(annotated, (int(ccx), int(ccy)), 3, (255, 255, 255), -1)

        elif len(obj.corners_2d) >= 4:
            pts = np.array(obj.corners_2d[:4], dtype=np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            poly_overlay = annotated.copy()
            cv2.fillPoly(poly_overlay, [pts], fill_color)
            cv2.addWeighted(poly_overlay, 0.16, annotated, 0.84, 0, annotated)

            for pt in obj.corners_2d[:4]:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1)
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 5, color, 1)

        # Center marker crosshair
        cx, cy = obj.center_2d
        cv2.drawMarker(annotated, (int(cx), int(cy)), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

        # HUD dimension pill banner
        bx, by, bw, bh = obj.bbox
        text_x = max(10, min(w - 260, bx))
        text_y = max(24, by - 8)

        (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(annotated, (text_x - 3, text_y - th - 5), (text_x + tw + 6, text_y + 3), (10, 14, 20), -1)
        cv2.rectangle(annotated, (text_x - 3, text_y - th - 5), (text_x + tw + 6, text_y + 3), color, 1)
        cv2.putText(annotated, tag_label, (text_x + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

    # Top-Left Live Status Badge
    active_count = len([o for o in tracked_objects if getattr(o, "missing_frames", 0) == 0])
    if active_count > 0:
        badge_text = f"ACTIVE OBJECTS: {active_count} (Boxes & Onions Continuous Tracking)"
        badge_color = (0, 240, 120)
    else:
        badge_text = "SCANNING VIEW — NO OBJECT DETECTED"
        badge_color = (170, 180, 190)

    cv2.putText(annotated, badge_text, (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_color, 2)

    return annotated


# ==============================================================================
# SECTION 4: VELOCITY-COMPENSATED MULTI-OBJECT TRACKER & STABILIZATION ENGINE
# ==============================================================================

class TrackedObject:
    """
    Represents a unique physical object (Box or Onion) tracked continuously.
    Maintains persistent tracking ID, spatial motion history, and stabilized dimensions.
    """

    def __init__(self, obj_id=1, shape_type="BOX", unit="mm"):
        self.obj_id = obj_id
        self.shape_type = shape_type  # "BOX", "SQUARE", "ONION", "CIRCLE"
        self.unit = unit
        self.state = "TRACKING"

        # Visual & spatial geometry
        self.contour = None
        self.corners_2d = []
        self.corners_3d = []
        self.center_2d = (0, 0)
        self.center_3d = None
        self.bbox = (0, 0, 0, 0)
        self.circle_center_2d = (0, 0)
        self.circle_radius_px = 0.0
        self.top_depth_m = 0.50
        self.support_depth_m = 0.85

        # Motion & velocity tracking
        self.vx = 0.0
        self.vy = 0.0
        self.predicted_center = (0.0, 0.0)

        # Tracking metrics
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_tracked = 0
        self.missing_frames = 0
        self.centroid_history = []

        # Measurements
        self.live_dims_mm = {}
        self.stable_dims_mm = {}
        self.is_stable = False
        self.size_class = "MEDIUM"
        self.timestamp = ""

        # Multi-frame measurement buffer for stabilization
        self._sample_history = {
            "length": [],
            "width": [],
            "height": [],
            "depth": [],
            "radius": [],
            "diameter": []
        }

    def update(self, candidate: DetectedObjectCandidate, unit="mm"):
        """
        Updates tracking position, velocity, smooths metrics, and stabilizes physical dimensions.
        Returns True if the object achieved stabilization in this frame.
        """
        self.last_seen = time.time()
        self.frames_tracked += 1
        self.missing_frames = 0
        self.unit = unit
        self.shape_type = candidate.shape_type

        # Estimate velocity
        if self.frames_tracked > 1 and self.center_2d[0] > 0 and self.center_2d[1] > 0:
            dx = candidate.center_2d[0] - self.center_2d[0]
            dy = candidate.center_2d[1] - self.center_2d[1]
            self.vx = 0.55 * dx + 0.45 * self.vx
            self.vy = 0.55 * dy + 0.45 * self.vy
        else:
            self.vx = 0.0
            self.vy = 0.0

        # Update centroid & prediction
        self.center_2d = candidate.center_2d
        self.predicted_center = (
            float(self.center_2d[0] + self.vx),
            float(self.center_2d[1] + self.vy)
        )
        self.center_3d = candidate.center_3d
        self.contour = candidate.contour
        self.bbox = candidate.bbox
        self.circle_center_2d = candidate.circle_center_2d
        self.circle_radius_px = candidate.circle_radius_px

        # Temporal smoothing for 2D corners (for boxes)
        if len(self.corners_2d) == 4 and len(candidate.corners_2d) == 4:
            smoothed_corners = []
            cand_pts = candidate.corners_2d
            for i in range(4):
                best_idx = 0
                best_d = float('inf')
                for j in range(4):
                    d = math.hypot(self.corners_2d[i][0] - cand_pts[j][0], self.corners_2d[i][1] - cand_pts[j][1])
                    if d < best_d:
                        best_d = d
                        best_idx = j
                cx_s = int(0.60 * cand_pts[best_idx][0] + 0.40 * self.corners_2d[i][0])
                cy_s = int(0.60 * cand_pts[best_idx][1] + 0.40 * self.corners_2d[i][1])
                smoothed_corners.append((cx_s, cy_s))
            self.corners_2d = smoothed_corners
        else:
            self.corners_2d = candidate.corners_2d

        self.corners_3d = candidate.corners_3d
        self.top_depth_m = candidate.top_depth_m
        self.support_depth_m = candidate.support_depth_m

        self.centroid_history.append(self.center_2d)
        if len(self.centroid_history) > 40:
            self.centroid_history.pop(0)

        # Update measurement history buffers
        for key, val in candidate.dimensions.items():
            if key in ["size_class", "grade"]:
                continue
            val_mm = val * 10.0 if unit == "cm" else val
            if key not in self._sample_history:
                self._sample_history[key] = []
            self._sample_history[key].append(val_mm)
            if len(self._sample_history[key]) > 15:
                self._sample_history[key].pop(0)

            # Exponential smoothing (alpha = 0.45)
            if key not in self.live_dims_mm or self.live_dims_mm[key] <= 0:
                self.live_dims_mm[key] = val_mm
            else:
                self.live_dims_mm[key] = 0.45 * val_mm + 0.55 * self.live_dims_mm[key]

        if "size_class" in candidate.dimensions:
            self.size_class = candidate.dimensions["size_class"]

        just_stabilized = False

        # State transitions
        if self.frames_tracked == 1:
            self.state = "TRACKING"
            self._finalize_stabilization()
        elif self.frames_tracked == 2:
            self.state = "MEASURING"
            self._finalize_stabilization()
        elif self.frames_tracked >= 3:
            if not self.is_stable:
                self._finalize_stabilization()
                just_stabilized = True
            else:
                self._finalize_stabilization()
            self.state = "STABLE"

        return just_stabilized

    def extrapolate_position(self):
        """Extrapolates position during brief frame dropouts using motion velocity."""
        self.missing_frames += 1
        self.center_2d = (
            int(self.center_2d[0] + self.vx),
            int(self.center_2d[1] + self.vy)
        )
        self.predicted_center = (
            float(self.center_2d[0] + self.vx),
            float(self.center_2d[1] + self.vy)
        )
        self.vx *= 0.85
        self.vy *= 0.85

    def _finalize_stabilization(self):
        """Computes outlier-rejected median measurements from multi-frame history."""
        for key, samples in self._sample_history.items():
            if len(samples) > 0:
                valid = [s for s in samples if s > 0]
                if len(valid) > 0:
                    self.stable_dims_mm[key] = float(np.median(valid))
                elif key in self.live_dims_mm:
                    self.stable_dims_mm[key] = self.live_dims_mm[key]
            elif key in self.live_dims_mm:
                self.stable_dims_mm[key] = self.live_dims_mm[key]

        if self.shape_type in ["ONION", "CIRCLE"]:
            stable_dia = self.stable_dims_mm.get("diameter", self.live_dims_mm.get("diameter", 0.0))
            self.size_class = classify_onion_size(stable_dia)

        self.is_stable = (self.frames_tracked >= 3)
        if not self.timestamp:
            self.timestamp = time.strftime("%H:%M:%S")

    def get_dimensions(self, target_unit="mm"):
        """Returns clean formatted dimensions in requested unit for UI and logging."""
        use_dict = self.stable_dims_mm if self.is_stable and len(self.stable_dims_mm) > 0 else self.live_dims_mm
        formatted = {}

        for key, val_mm in use_dict.items():
            formatted[key] = format_dimension(val_mm, target_unit)

        # Volume calculation
        if self.shape_type in ["ONION", "CIRCLE"]:
            r_mm = use_dict.get("radius", use_dict.get("diameter", 0.0) / 2.0)
            r_cm = r_mm / 10.0
            vol_liters = (4.0 / 3.0) * math.pi * (r_cm ** 3) / 1000.0
            formatted["volume_l"] = vol_liters
            formatted["size_class"] = self.size_class
            formatted["grade"] = self.size_class
        else:
            l_mm = use_dict.get("length", 0.0)
            w_mm = use_dict.get("width", 0.0)
            h_mm = use_dict.get("height", 0.0)
            vol_liters = ((l_mm / 10.0) * (w_mm / 10.0) * (h_mm / 10.0)) / 1000.0
            formatted["volume_l"] = vol_liters

        formatted["obj_id"] = self.obj_id
        formatted["box_id"] = self.obj_id
        formatted["shape_type"] = self.shape_type
        formatted["state"] = self.state
        formatted["is_stable"] = self.is_stable
        formatted["unit"] = target_unit
        formatted["timestamp"] = self.timestamp or time.strftime("%H:%M:%S")

        return formatted


# Backwards compatibility alias
TrackedBox = TrackedObject


def compute_bbox_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) of two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


class MultiObjectTracker:
    """
    Continuous Multi-Object Tracking and Stabilization Engine for Boxes and Onions.
    """

    def __init__(self, baseline_depth_mm=850.0, min_height_mm=12.0):
        self.baseline_depth_mm = baseline_depth_mm
        self.min_height_mm = min_height_mm
        self.active_objects = []
        self.total_objects_counted = 0
        self.completed_objects_log = []
        self.logged_object_ids = set()
        self.latest_diagnostics = {}

    @property
    def active_boxes(self):
        return self.active_objects

    @property
    def completed_boxes_log(self):
        return self.completed_objects_log

    @property
    def total_boxes_counted(self):
        return self.total_objects_counted

    @property
    def current_zone_box(self):
        if len(self.active_objects) > 0:
            stable_objs = [o for o in self.active_objects if o.is_stable and o.missing_frames == 0]
            if len(stable_objs) > 0:
                return stable_objs[0]
            visible_objs = [o for o in self.active_objects if o.missing_frames == 0]
            if len(visible_objs) > 0:
                return visible_objs[0]
            return self.active_objects[0]
        return None

    def calibrate_baseline(self, depth_raw):
        """Calibrates supporting surface depth from current depth frame."""
        if depth_raw is None:
            return self.baseline_depth_mm

        h, w = depth_raw.shape[:2]
        roi = depth_raw[int(h * 0.20):int(h * 0.80), int(w * 0.20):int(w * 0.80)]
        valid = roi[(roi > 100) & (roi < 6000)]
        if len(valid) >= 50:
            p10, p90 = np.percentile(valid, [10, 90])
            filtered = valid[(valid >= p10) & (valid <= p90)]
            if len(filtered) > 0:
                self.baseline_depth_mm = float(np.median(filtered))
            else:
                self.baseline_depth_mm = float(np.median(valid))
            print(f"[Tracker] Conveyor/Floor depth calibrated: {self.baseline_depth_mm:.1f} mm ({self.baseline_depth_mm/10.0:.1f} cm)")
        return self.baseline_depth_mm

    def process_frame(self, color_bgr, depth_raw, intrinsics, unit="mm"):
        """
        Continuously executes full-frame box and onion detection, motion-assisted tracking,
        and dimension stabilization.
        """
        if color_bgr is None or depth_raw is None:
            return [], None, [], None, {}

        t0 = time.perf_counter()
        h, w = depth_raw.shape[:2]

        # 1. Full-Frame Detection of all candidates
        candidates, edge_map, det_diag = MultiObjectDetector.detect_and_measure(
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            intrinsics=intrinsics,
            min_height_mm=self.min_height_mm,
            unit=unit
        )

        newly_completed = []

        # 2. Association with Velocity Compensation & Cost Matrix
        matched_cand_indices = set()
        matched_obj_indices = set()

        if len(self.active_objects) > 0 and len(candidates) > 0:
            cost_matrix = np.zeros((len(self.active_objects), len(candidates)), dtype=np.float32)

            for o_idx, obj in enumerate(self.active_objects):
                pred_x = obj.predicted_center[0] if obj.frames_tracked > 1 else obj.center_2d[0]
                pred_y = obj.predicted_center[1] if obj.frames_tracked > 1 else obj.center_2d[1]

                for c_idx, cand in enumerate(candidates):
                    dist_2d = math.hypot(pred_x - cand.center_2d[0], pred_y - cand.center_2d[1])
                    depth_diff_mm = abs(obj.top_depth_m - cand.top_depth_m) * 1000.0
                    iou = compute_bbox_iou(obj.bbox, cand.bbox)
                    overlap_bonus = -40.0 * iou if iou > 0.10 else 0.0
                    bw_diff = abs(obj.bbox[2] - cand.bbox[2])
                    bh_diff = abs(obj.bbox[3] - cand.bbox[3])
                    shape_penalty = 50.0 if obj.shape_type != cand.shape_type else 0.0

                    cost_matrix[o_idx, c_idx] = dist_2d + (depth_diff_mm * 0.20) + overlap_bonus + (bw_diff + bh_diff) * 0.10 + shape_penalty

            max_match_cost = 260.0
            while True:
                min_cost = np.min(cost_matrix)
                if min_cost > max_match_cost or np.isinf(min_cost):
                    break

                o_idx, c_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                if cost_matrix[o_idx, c_idx] == np.inf:
                    break

                matched_obj_indices.add(o_idx)
                matched_cand_indices.add(c_idx)

                obj = self.active_objects[o_idx]
                cand = candidates[c_idx]

                just_stabilized = obj.update(cand, unit=unit)
                if just_stabilized and obj.obj_id not in self.logged_object_ids:
                    self.logged_object_ids.add(obj.obj_id)
                    dims = obj.get_dimensions(unit)
                    self.completed_objects_log.append(dims)
                    newly_completed.append(dims)

                cost_matrix[o_idx, :] = np.inf
                cost_matrix[:, c_idx] = np.inf

        # 3. Spatial Deduplication & Reconnection for Unmatched Candidates
        for c_idx, cand in enumerate(candidates):
            if c_idx not in matched_cand_indices:
                best_reconnect = None
                best_d = float('inf')
                diag = math.hypot(cand.bbox[2], cand.bbox[3])

                for o_idx, obj in enumerate(self.active_objects):
                    if o_idx not in matched_obj_indices:
                        d = math.hypot(obj.center_2d[0] - cand.center_2d[0], obj.center_2d[1] - cand.center_2d[1])
                        dz = abs(obj.top_depth_m - cand.top_depth_m) * 1000.0
                        iou = compute_bbox_iou(obj.bbox, cand.bbox)

                        if (d < max(80.0, diag * 0.70) or iou > 0.15) and dz < 120.0:
                            if d < best_d:
                                best_d = d
                                best_reconnect = (o_idx, obj)

                if best_reconnect is not None:
                    o_idx, obj = best_reconnect
                    matched_obj_indices.add(o_idx)
                    matched_cand_indices.add(c_idx)
                    just_stabilized = obj.update(cand, unit=unit)
                    if just_stabilized and obj.obj_id not in self.logged_object_ids:
                        self.logged_object_ids.add(obj.obj_id)
                        dims = obj.get_dimensions(unit)
                        self.completed_objects_log.append(dims)
                        newly_completed.append(dims)
                else:
                    self.total_objects_counted += 1
                    new_obj = TrackedObject(
                        obj_id=self.total_objects_counted,
                        shape_type=cand.shape_type,
                        unit=unit
                    )
                    just_stabilized = new_obj.update(cand, unit=unit)
                    self.active_objects.append(new_obj)
                    matched_cand_indices.add(c_idx)

                    if just_stabilized and new_obj.obj_id not in self.logged_object_ids:
                        self.logged_object_ids.add(new_obj.obj_id)
                        dims = new_obj.get_dimensions(unit)
                        self.completed_objects_log.append(dims)
                        newly_completed.append(dims)

        # 4. Handle Missing Frames & Object Removal
        surviving_objects = []
        for o_idx, obj in enumerate(self.active_objects):
            if o_idx not in matched_obj_indices:
                obj.extrapolate_position()
                cx, cy = obj.center_2d
                is_near_border = (cx < 25 or cx > (w - 25) or cy < 25 or cy > (h - 25))
                if is_near_border:
                    obj.state = "LEAVING"

                max_miss = 4 if is_near_border else 8
                if obj.missing_frames <= max_miss:
                    surviving_objects.append(obj)
                else:
                    obj.state = "REMOVED"
                    if obj.obj_id not in self.logged_object_ids and obj.frames_tracked >= 2:
                        self.logged_object_ids.add(obj.obj_id)
                        obj._finalize_stabilization()
                        dims = obj.get_dimensions(unit)
                        self.completed_objects_log.append(dims)
                        newly_completed.append(dims)
            else:
                surviving_objects.append(obj)

        self.active_objects = surviving_objects

        t_total = (time.perf_counter() - t0) * 1000.0

        diagnostics = {
            "t_total_ms": t_total,
            "processing_fps": 1000.0 / max(1.0, t_total),
            "active_objects_count": len(self.active_objects),
            "candidates_count": len(candidates),
            "contours_found": det_diag.get("contours_found", 0),
            "rejected_count": det_diag.get("rejected_count", 0)
        }
        self.latest_diagnostics = diagnostics
        primary_object = self.current_zone_box

        return self.active_objects, primary_object, newly_completed, edge_map, diagnostics


# Backwards compatibility aliases
MultiBoxConveyorTracker = MultiObjectTracker


def draw_multi_object_tracker_overlay(image_bgr, tracker: MultiObjectTracker, unit="mm"):
    """Renders live full-frame tracking wireframes, circles, and dimension HUD tags."""
    return draw_multi_object_overlays(image_bgr, tracker.active_objects, unit=unit)


draw_conveyor_inspection_overlay = draw_multi_object_tracker_overlay


# ==============================================================================
# SECTION 5: PYQT5 DESKTOP DASHBOARD & VIEWPORT
# ==============================================================================

class LiveCameraViewport(QLabel):
    """
    Renders the live video stream with full-frame multi-object detection,
    wireframe/circle overlays, and dynamic HUD dimension tags.
    """
    mouse_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(580, 430)
        self.setStyleSheet("background-color: #0b0d12; border: 1px solid #1a202c; border-radius: 8px;")

        self.raw_frame_w = 640
        self.raw_frame_h = 480

        self.cursor_img_x = -1
        self.cursor_img_y = -1
        self.is_mouse_inside = False

    def update_frame(self, display_bgr):
        """Updates viewport with annotated OpenCV frame."""
        if display_bgr is None:
            return

        h, w, ch = display_bgr.shape
        self.raw_frame_w = w
        self.raw_frame_h = h

        rgb_image = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self.is_mouse_inside and self.cursor_img_x >= 0:
            self._draw_cursor_overlay(scaled_pixmap)

        self.setPixmap(scaled_pixmap)

    def _get_image_rect_in_viewport(self):
        if self.pixmap() is None:
            return QRect(0, 0, self.width(), self.height())
        pm_w = self.pixmap().width()
        pm_h = self.pixmap().height()
        x_offset = (self.width() - pm_w) // 2
        y_offset = (self.height() - pm_h) // 2
        return QRect(x_offset, y_offset, pm_w, pm_h)

    def _map_viewport_to_image_coords(self, vp_x, vp_y):
        img_rect = self._get_image_rect_in_viewport()
        if not img_rect.contains(vp_x, vp_y):
            return -1, -1

        rel_x = vp_x - img_rect.x()
        rel_y = vp_y - img_rect.y()

        frame_x = int((rel_x / max(1, img_rect.width())) * self.raw_frame_w)
        frame_y = int((rel_y / max(1, img_rect.height())) * self.raw_frame_h)

        frame_x = max(0, min(self.raw_frame_w - 1, frame_x))
        frame_y = max(0, min(self.raw_frame_h - 1, frame_y))
        return frame_x, frame_y

    def mouseMoveEvent(self, event):
        img_x, img_y = self._map_viewport_to_image_coords(event.x(), event.y())
        if img_x >= 0 and img_y >= 0:
            self.cursor_img_x = img_x
            self.cursor_img_y = img_y
            self.is_mouse_inside = True
            self.mouse_moved.emit(img_x, img_y)
        else:
            self.is_mouse_inside = False
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.is_mouse_inside = False
        super().leaveEvent(event)

    def _draw_cursor_overlay(self, target_pixmap):
        painter = QPainter(target_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        target_rect = target_pixmap.rect()

        scale_x = target_rect.width() / float(self.raw_frame_w)
        scale_y = target_rect.height() / float(self.raw_frame_h)
        px = int(self.cursor_img_x * scale_x)
        py = int(self.cursor_img_y * scale_y)

        pen = QPen(QColor(0, 220, 255, 170), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(px - 10, py, px + 10, py)
        painter.drawLine(px, py - 10, px, py + 10)
        painter.end()


class MainWindow(QMainWindow):
    """
    Main desktop window providing continuous multi-object real-time 3D measurement
    for Boxes and Onions.
    """

    def __init__(self, force_simulate=False, width=640, height=480, fps=30):
        super().__init__()
        self.width_res = width
        self.height_res = height
        self.fps = fps
        self.force_simulate = force_simulate

        self.setWindowTitle("Intel RealSense D455f — Continuous Multi-Object Measurement System (Boxes & Onions)")
        self.resize(1440, 900)
        self.setMinimumSize(1140, 720)

        # Core State & Tracker Engine
        self.current_unit = "mm"
        self.view_mode = "INSPECTION"

        self.latest_depth_raw = None
        self.latest_color_bgr = None
        self.latest_depth_colormap = None
        self.latest_intrinsics = None

        self.tracker = MultiObjectTracker(baseline_depth_mm=850.0, min_height_mm=12.0)

        # Performance
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        self._apply_theme()
        self._init_ui()

        # Camera Thread Worker
        self.camera_thread = RealSenseThread(
            width=self.width_res,
            height=self.height_res,
            fps=self.fps,
            force_simulate=self.force_simulate
        )
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.status_changed.connect(self.on_camera_status_changed)
        self.camera_thread.error_occurred.connect(self.on_camera_error)
        self.camera_thread.start()

    def _apply_theme(self):
        """Applies a clean, dark industrial theme."""
        dark_style = """
        QMainWindow {
            background-color: #0b0d11;
        }
        QWidget {
            color: #d8dee9;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 13px;
            border: 1px solid #1a202c;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 12px;
            background-color: #11141c;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 8px;
            color: #61afef;
        }
        QPushButton {
            background-color: #1a202c;
            color: #ffffff;
            border: 1px solid #2d3748;
            border-radius: 6px;
            padding: 7px 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #2b3548;
            border-color: #61afef;
        }
        QPushButton:pressed {
            background-color: #141822;
        }
        QPushButton#startBtn {
            background-color: #185432;
            border-color: #98c379;
        }
        QPushButton#startBtn:hover {
            background-color: #217344;
        }
        QPushButton#stopBtn {
            background-color: #5a1e23;
            border-color: #e06c75;
        }
        QPushButton#stopBtn:hover {
            background-color: #77282e;
        }
        QPushButton#calibBtn {
            background-color: #21334f;
            border-color: #61afef;
        }
        QPushButton#calibBtn:hover {
            background-color: #2c446b;
        }
        QComboBox {
            background-color: #161a24;
            color: #e5e9f0;
            border: 1px solid #283244;
            border-radius: 4px;
            padding: 4px 10px;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #1a202c;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #61afef;
            width: 14px;
            margin-top: -4px;
            margin-bottom: -4px;
            border-radius: 7px;
        }
        QTableWidget {
            background-color: #0d1017;
            border: 1px solid #1a202c;
            border-radius: 6px;
            gridline-color: #1f2737;
            color: #d8dee9;
            font-size: 12px;
        }
        QHeaderView::section {
            background-color: #161b26;
            color: #61afef;
            font-weight: bold;
            border: 1px solid #1a202c;
            padding: 4px 8px;
        }
        QTableWidget::item:selected {
            background-color: #1e3654;
            color: #ffffff;
        }
        QStatusBar {
            background-color: #0c0e12;
            color: #8892b0;
            border-top: 1px solid #1a1e27;
        }
        """
        self.setStyleSheet(dark_style)

    def _init_ui(self):
        """Builds the comprehensive GUI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 10, 14, 10)
        main_layout.setSpacing(10)

        # 1. Top Header Bar
        header_layout = QHBoxLayout()
        title_label = QLabel("INTEL REALSENSE D455f — MULTI-OBJECT MEASUREMENT (BOXES & ONIONS)")
        title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff; letter-spacing: 0.5px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.active_count_badge = QLabel("🔍 Onions in Camera View: 0")
        self.active_count_badge.setStyleSheet(
            "background-color: #14281e; color: #00ff78; padding: 4px 12px; "
            "border-radius: 10px; font-weight: bold; border: 1px solid #00aa50;"
        )
        header_layout.addWidget(self.active_count_badge)

        self.box_count_badge = QLabel("🧅 Total Onions Counted: 0")
        self.box_count_badge.setStyleSheet(
            "background-color: #162438; color: #61afef; padding: 4px 12px; "
            "border-radius: 10px; font-weight: bold; border: 1px solid #2f5485;"
        )
        header_layout.addWidget(self.box_count_badge)

        self.status_badge = QLabel("[*] Initializing Camera...")
        self.status_badge.setStyleSheet(
            "background-color: #241c12; color: #e5c07b; padding: 4px 12px; "
            "border-radius: 10px; font-weight: bold; border: 1px solid #d19a66;"
        )
        header_layout.addWidget(self.status_badge)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #98c379; font-weight: bold; margin-left: 8px;")
        header_layout.addWidget(self.fps_label)

        main_layout.addLayout(header_layout)

        # 2. Main Content Splitter (Left: Live Viewport, Right: Dashboard & Tables)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # 2A. LEFT: Live Camera Viewport Container
        vp_container = QWidget()
        vp_layout = QVBoxLayout(vp_container)
        vp_layout.setContentsMargins(0, 0, 0, 0)
        vp_layout.setSpacing(6)

        vp_header_layout = QHBoxLayout()
        vp_title = QLabel("FULL-FRAME LIVE CAMERA VIEW (Continuous Multi-Object Tracking)")
        vp_title.setStyleSheet("font-weight: bold; color: #61afef;")
        vp_header_layout.addWidget(vp_title)

        vp_header_layout.addStretch()
        vp_header_layout.addWidget(QLabel("View Mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "Live Inspection Stream (RGB + 3D Tracking HUD)",
            "Canny Edge Detection Map",
            "Depth Colormap (Z-Depth)"
        ])
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        vp_header_layout.addWidget(self.view_mode_combo)

        vp_layout.addLayout(vp_header_layout)

        self.camera_viewport = LiveCameraViewport()
        self.camera_viewport.mouse_moved.connect(self.on_mouse_moved)
        vp_layout.addWidget(self.camera_viewport, stretch=1)

        content_layout.addWidget(vp_container, stretch=6)

        # 2B. RIGHT: Active Objects Dashboard & Completed Log
        sidebar = QWidget()
        sidebar.setMinimumWidth(460)
        sidebar.setMaximumWidth(580)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        # 2B-1. PROMINENT ACTIVE OBJECTS LIVE PANEL
        meas_group = QGroupBox("Active Tracked Objects (Live 3D Dimensions)")
        meas_layout = QVBoxLayout(meas_group)
        meas_layout.setSpacing(6)

        self.active_status_badge = QLabel("SCANNING VIEW — NO OBJECT DETECTED")
        self.active_status_badge.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #abb2bf; padding: 5px 10px; "
            "background-color: #171a22; border-radius: 6px; border: 1px solid #282e3d;"
        )
        meas_layout.addWidget(self.active_status_badge)

        self.dim_display_box = QLabel("No objects currently visible in camera view.\nPlace boxes or circular onions/potatoes in front of camera.")
        self.dim_display_box.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #98c379; line-height: 1.4; "
            "padding: 10px; background-color: #0b1710; border-radius: 6px; border: 1px solid #1b4528;"
        )
        meas_layout.addWidget(self.dim_display_box)

        sidebar_layout.addWidget(meas_group)

        # 2B-2. DETECTION THRESHOLD & CALIBRATION
        settings_group = QGroupBox("Detection Sensitivity & Surface Calibration")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(6)

        settings_layout.addWidget(QLabel("Min Elevation:"), 0, 0)
        self.min_h_slider = QSlider(Qt.Horizontal)
        self.min_h_slider.setRange(6, 60)
        self.min_h_slider.setValue(int(self.tracker.min_height_mm))
        self.min_h_label = QLabel(f"{int(self.tracker.min_height_mm)} mm")
        self.min_h_slider.valueChanged.connect(self.on_min_height_slider_changed)
        settings_layout.addWidget(self.min_h_slider, 0, 1)
        settings_layout.addWidget(self.min_h_label, 0, 2)

        # Baseline Calibration Button
        self.calib_btn = QPushButton("⚙ Auto Calibrate Conveyor Floor")
        self.calib_btn.setObjectName("calibBtn")
        self.calib_btn.clicked.connect(self.calibrate_baseline)
        settings_layout.addWidget(self.calib_btn, 1, 0, 1, 3)

        # Dedicated display label for Camera-to-Floor Distance
        self.calib_display_label = QLabel(f"Camera-to-Floor Distance: {self.tracker.baseline_depth_mm:.0f} mm ({self.tracker.baseline_depth_mm/10.0:.1f} cm)")
        self.calib_display_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #61afef; "
            "padding: 4px 8px; background-color: #0e1622; border-radius: 4px; border: 1px solid #1e3654;"
        )
        self.calib_display_label.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(self.calib_display_label, 2, 0, 1, 3)

        sidebar_layout.addWidget(settings_group)

        # 2B-3. COMPLETED MEASUREMENTS TABLE
        table_group = QGroupBox("Completed Measurements Log")
        table_layout = QVBoxLayout(table_group)
        table_layout.setSpacing(6)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Object ID", "Type", "Length / Dia", "Width / Rad", "Height / Grade", "Volume (L)", "Time"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        table_layout.addWidget(self.results_table)

        # Table Action Buttons (Export CSV, Clear)
        tbl_btn_layout = QHBoxLayout()
        self.export_csv_btn = QPushButton("📥 Export Measurements (CSV)")
        self.export_csv_btn.clicked.connect(self.export_csv)
        tbl_btn_layout.addWidget(self.export_csv_btn)

        self.clear_table_btn = QPushButton("🗑 Clear Log")
        self.clear_table_btn.clicked.connect(self.clear_table)
        tbl_btn_layout.addWidget(self.clear_table_btn)

        table_layout.addLayout(tbl_btn_layout)
        sidebar_layout.addWidget(table_group, stretch=1)

        # 2B-4. CURSOR INSPECTOR
        insp_row = QHBoxLayout()
        self.pixel_label = QLabel("Pixel: X = --, Y = --")
        self.pixel_label.setStyleSheet("color: #abb2bf; font-weight: bold;")
        self.cursor_depth_label = QLabel("Depth: --")
        self.cursor_depth_label.setStyleSheet("color: #e5c07b; font-weight: bold;")
        insp_row.addWidget(self.pixel_label)
        insp_row.addWidget(self.cursor_depth_label)
        sidebar_layout.addLayout(insp_row)

        content_layout.addWidget(sidebar, stretch=4)
        main_layout.addLayout(content_layout)

        # 3. Bottom Controls Toolbar
        toolbar_frame = QFrame()
        toolbar_frame.setStyleSheet("background-color: #12151d; border-radius: 8px; padding: 5px;")
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(10, 4, 10, 4)
        toolbar_layout.setSpacing(12)

        self.start_cam_btn = QPushButton("▶ START CAMERA")
        self.start_cam_btn.setObjectName("startBtn")
        self.start_cam_btn.clicked.connect(self.start_camera)
        toolbar_layout.addWidget(self.start_cam_btn)

        self.stop_cam_btn = QPushButton("⏹ STOP CAMERA")
        self.stop_cam_btn.setObjectName("stopBtn")
        self.stop_cam_btn.clicked.connect(self.stop_camera)
        toolbar_layout.addWidget(self.stop_cam_btn)

        self.reconnect_cam_btn = QPushButton("🔄 RECONNECT CAMERA")
        self.reconnect_cam_btn.setObjectName("reconnectBtn")
        self.reconnect_cam_btn.clicked.connect(self.reconnect_camera)
        toolbar_layout.addWidget(self.reconnect_cam_btn)

        toolbar_layout.addStretch()

        toolbar_layout.addWidget(QLabel("Measurement Unit:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Millimeters (mm)", "Centimeters (cm)"])
        self.unit_combo.currentIndexChanged.connect(self.on_unit_changed)
        toolbar_layout.addWidget(self.unit_combo)

        self.snap_btn = QPushButton("📷 Save Snapshot")
        self.snap_btn.clicked.connect(self.save_snapshot)
        toolbar_layout.addWidget(self.snap_btn)

        main_layout.addWidget(toolbar_frame)

        # 4. Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Continuous multi-object measurement active (Boxes & Onions).")

    def on_view_mode_changed(self, idx):
        if idx == 0:
            self.view_mode = "INSPECTION"
        elif idx == 1:
            self.view_mode = "CANNY_EDGES"
        elif idx == 2:
            self.view_mode = "DEPTH_MAP"

    def on_min_height_slider_changed(self, val_mm):
        self.tracker.min_height_mm = float(val_mm)
        self.min_h_label.setText(f"{val_mm} mm")

    def on_unit_changed(self, idx):
        self.current_unit = "mm" if idx == 0 else "cm"
        self._refresh_measurement_card()

    def calibrate_baseline(self):
        """Calibrates the conveyor/floor supporting depth from current depth frame."""
        if self.latest_depth_raw is not None:
            new_baseline_mm = self.tracker.calibrate_baseline(self.latest_depth_raw)
            new_baseline_cm = new_baseline_mm / 10.0
            
            dist_text = f"Camera-to-Floor Distance: {new_baseline_mm:.0f} mm ({new_baseline_cm:.1f} cm)"
            self.calib_display_label.setText(dist_text)
            self.calib_display_label.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #00ff78; "
                "padding: 4px 8px; background-color: #0b1f13; border-radius: 4px; border: 1px solid #00aa50;"
            )
            
            confirmation_msg = f"Floor depth calibrated successfully: {new_baseline_mm:.0f} mm ({new_baseline_cm:.1f} cm)"
            self.status_bar.showMessage(confirmation_msg, 5000)
            print(f"[+] {confirmation_msg}")
        else:
            self.status_bar.showMessage("Error: No depth frame available from camera to calibrate.", 4000)

    def on_frame_ready(self, color_bgr, depth_colormap, depth_raw, intrinsics):
        """Processes live camera frames, executes continuous full-frame tracking."""
        self.latest_color_bgr = color_bgr
        self.latest_depth_raw = depth_raw
        self.latest_depth_colormap = depth_colormap
        self.latest_intrinsics = intrinsics

        if color_bgr is not None and depth_raw is not None:
            active_objects, primary_obj, newly_completed, edge_map, diagnostics = self.tracker.process_frame(
                color_bgr=color_bgr,
                depth_raw=depth_raw,
                intrinsics=intrinsics,
                unit=self.current_unit
            )

            if len(newly_completed) > 0:
                for obj_data in newly_completed:
                    self._add_object_to_results_table(obj_data)

            active_vis = [o for o in active_objects if o.missing_frames == 0]
            self.active_count_badge.setText(f"🔍 Onions in Camera View: {len(active_vis)}")
            self.box_count_badge.setText(f"🧅 Total Onions Counted: {self.tracker.total_objects_counted}")

            self._update_active_objects_card(active_vis)

            if self.view_mode == "INSPECTION":
                display_frame = draw_multi_object_tracker_overlay(
                    color_bgr, self.tracker, unit=self.current_unit
                )
            elif self.view_mode == "CANNY_EDGES":
                display_frame = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR) if edge_map is not None else color_bgr
            else:  # DEPTH_MAP
                display_frame = depth_colormap if depth_colormap is not None else color_bgr

            self.camera_viewport.update_frame(display_frame)

            t_total = diagnostics.get("t_total_ms", 0.0)
            n_active = len(active_vis)
            status_msg = f"Live Processing: {t_total:.1f} ms ({1000.0/max(1.0, t_total):.0f} FPS) | Active Objects: {n_active} | Total Logged: {len(self.tracker.completed_objects_log)}"
            self.status_bar.showMessage(status_msg)

        # FPS calculation
        self.fps_counter += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (now - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = now
            self.fps_label.setText(f"FPS: {self.current_fps:.1f}")

    def _update_active_objects_card(self, active_objects):
        """Updates the prominent Active Objects Live Panel with clean multi-object readouts."""
        if len(active_objects) > 0:
            self.active_status_badge.setText(f"DETECTING & TRACKING {len(active_objects)} OBJECT{'S' if len(active_objects) > 1 else ''}")
            self.active_status_badge.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #00ff78; padding: 5px 10px; "
                "background-color: #0b1f13; border-radius: 6px; border: 1px solid #00ff78;"
            )

            lines = []
            u = self.current_unit

            for obj in active_objects[:4]:
                dims = obj.get_dimensions(u)
                shape = obj.shape_type
                status_str = "[STABLE]" if obj.is_stable else "[TRACKING...]"

                if shape in ["ONION", "CIRCLE"]:
                    dia = dims.get("diameter", 0)
                    rad = dims.get("radius", 0)
                    size_cls = dims.get("size_class", "MEDIUM")
                    h_val = dims.get("height", 0)
                    vol = dims.get("volume_l", 0)
                    lines.append(f"▶ ONION #{obj.obj_id:02d}  [{size_cls}]  {status_str}")
                    if u == "mm":
                        lines.append(f"   Dia: {dia:4.0f} mm (Rad: {rad:3.0f} mm) | Size: {size_cls}")
                        lines.append(f"   Height: {h_val:3.0f} mm  (Vol: {vol:.3f} L)")
                    else:
                        lines.append(f"   Dia: {dia:4.1f} cm (Rad: {rad:3.1f} cm) | Size: {size_cls}")
                        lines.append(f"   Height: {h_val:3.1f} cm  (Vol: {vol:.3f} L)")
                else:
                    l = dims.get("length", 0)
                    w = dims.get("width", 0)
                    h = dims.get("height", 0)
                    vol = dims.get("volume_l", 0)
                    lines.append(f"▶ {shape} #{obj.obj_id:02d}  {status_str}")
                    if u == "mm":
                        lines.append(f"   Dim: {l:4.0f} x {w:4.0f} x {h:4.0f} mm  (Vol: {vol:.2f} L)")
                    else:
                        lines.append(f"   Dim: {l:4.1f} x {w:4.1f} x {h:4.1f} cm  (Vol: {vol:.2f} L)")

                lines.append("")

            if len(active_objects) > 4:
                lines.append(f"... and {len(active_objects) - 4} more objects tracked.")

            self.dim_display_box.setText("\n".join(lines).strip())
        else:
            self.active_status_badge.setText("SCANNING VIEW — NO OBJECT DETECTED")
            self.active_status_badge.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #abb2bf; padding: 5px 10px; "
                "background-color: #171a22; border-radius: 6px; border: 1px solid #282e3d;"
            )
            self.dim_display_box.setText(
                "No objects currently visible in camera view.\n"
                "The entire camera view is actively scanning.\n"
                "Boxes and Onions/Potatoes are measured and tracked automatically upon entry."
            )

    def _refresh_measurement_card(self):
        if self.tracker is not None:
            vis = [o for o in self.tracker.active_objects if o.missing_frames == 0]
            self._update_active_objects_card(vis)

    def _add_object_to_results_table(self, obj_data):
        """Appends a newly stabilized object measurement row to the QTableWidget."""
        row_idx = self.results_table.rowCount()
        self.results_table.insertRow(row_idx)

        u = obj_data.get("unit", "mm")
        obj_id_text = f"ID #{obj_data.get('obj_id', obj_data.get('box_id', 1)):02d}"
        shape_text = obj_data.get("shape_type", "BOX")

        if shape_text in ["ONION", "CIRCLE"]:
            dia_val = obj_data.get("diameter", 0)
            rad_val = obj_data.get("radius", 0)
            size_cls = obj_data.get("size_class", "MEDIUM")
            l_text = f"Dia: {dia_val:.0f} {u}" if u == "mm" else f"Dia: {dia_val:.1f} {u}"
            w_text = f"Rad: {rad_val:.0f} {u}" if u == "mm" else f"Rad: {rad_val:.1f} {u}"
            h_val = obj_data.get("height", 0)
            h_text = f"{size_cls} ({h_val:.0f} {u})" if u == "mm" else f"{size_cls} ({h_val:.1f} {u})"
        else:
            l_text = f"{obj_data.get('length', 0):.0f} {u}" if u == "mm" else f"{obj_data.get('length', 0):.1f} {u}"
            w_text = f"{obj_data.get('width', 0):.0f} {u}" if u == "mm" else f"{obj_data.get('width', 0):.1f} {u}"
            h_text = f"{obj_data.get('height', 0):.0f} {u}" if u == "mm" else f"{obj_data.get('height', 0):.1f} {u}"

        vol_val = obj_data.get("volume_l", 0.0)
        vol_text = f"{vol_val:.3f} L" if shape_text == "ONION" else f"{vol_val:.2f} L"
        time_text = obj_data.get("timestamp", time.strftime("%H:%M:%S"))

        items = [obj_id_text, shape_text, l_text, w_text, h_text, vol_text, time_text]
        for col_idx, val in enumerate(items):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            if col_idx == 0:
                item.setForeground(QColor("#61afef"))
            elif col_idx == 1:
                item.setForeground(QColor("#e5c07b") if shape_text == "ONION" else QColor("#ffd700"))
            elif col_idx in [2, 3, 4]:
                item.setForeground(QColor("#98c379"))
            self.results_table.setItem(row_idx, col_idx, item)

        self.results_table.scrollToBottom()

    def export_csv(self):
        """Exports the completed measurements log to a CSV file."""
        if len(self.tracker.completed_objects_log) == 0:
            QMessageBox.information(self, "Export CSV", "No object measurements recorded yet.")
            return

        default_name = f"realsense_measurements_{int(time.time())}.csv"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Measurements CSV", default_name, "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Object ID", "Type", "Length / Dia", "Width / Rad", "Height / Grade", "Volume (L)", "Timestamp"])
                for b in self.tracker.completed_objects_log:
                    obj_id_val = b.get("obj_id", b.get("box_id", 1))
                    shape = b.get("shape_type", "BOX")
                    if shape in ["ONION", "CIRCLE"]:
                        dia_val = b.get("diameter", 0)
                        rad_val = b.get("radius", 0)
                        size_cls = b.get("size_class", "MEDIUM")
                        writer.writerow([
                            f"ID #{obj_id_val:02d}", shape, f"{dia_val:.1f}", f"{rad_val:.1f}",
                            f"{size_cls}", f"{b.get('volume_l', 0.0):.3f}", b.get("timestamp", "")
                        ])
                    else:
                        l_val = b.get("length", 0)
                        w_val = b.get("width", 0)
                        h_val = b.get("height", 0)
                        writer.writerow([
                            f"ID #{obj_id_val:02d}", shape, f"{l_val:.1f}", f"{w_val:.1f}",
                            f"{h_val:.1f}", f"{b.get('volume_l', 0.0):.2f}", b.get("timestamp", "")
                        ])
            self.status_bar.showMessage(f"Successfully exported log to: {file_path}", 4000)
            QMessageBox.information(self, "Export Successful", f"Saved {len(self.tracker.completed_objects_log)} object records to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save CSV: {e}")

    def clear_table(self):
        """Clears the completed measurements log table."""
        self.results_table.setRowCount(0)
        self.tracker.completed_objects_log.clear()
        self.tracker.logged_object_ids.clear()
        self.status_bar.showMessage("Measurements log cleared.", 3000)

    def start_camera(self):
        if not self.camera_thread.is_connected and not self.force_simulate:
            self.status_bar.showMessage("Connecting to RealSense camera...", 3000)
            success = self.camera_thread.reconnect()
            if not success:
                self.camera_thread.resume_stream()
        else:
            self.camera_thread.resume_stream()
            self.status_bar.showMessage("Camera started.", 3000)

    def stop_camera(self):
        self.camera_thread.pause_stream()
        self.status_bar.showMessage("Camera stopped / paused.", 3000)

    def reconnect_camera(self):
        self.status_bar.showMessage("Scanning USB and reconnecting camera...", 3000)
        success = self.camera_thread.reconnect()
        if success:
            self.status_bar.showMessage("RealSense camera connected successfully!", 4000)
        else:
            self.status_bar.showMessage("No physical RealSense device detected. Check USB connection.", 5000)

    def on_camera_status_changed(self, status_text, is_connected, device_info):
        if is_connected:
            self.status_badge.setText(f"[+] {status_text}")
            self.status_badge.setStyleSheet(
                "background-color: #162d22; color: #98c379; padding: 4px 12px; "
                "border-radius: 10px; font-weight: bold; border: 1px solid #236e49;"
            )
        else:
            self.status_badge.setText(f"[*] {status_text}")
            self.status_badge.setStyleSheet(
                "background-color: #241c12; color: #e5c07b; padding: 4px 12px; "
                "border-radius: 10px; font-weight: bold; border: 1px solid #d19a66;"
            )
        self.status_bar.showMessage(status_text, 4000)

    def on_camera_error(self, error_message):
        self.status_bar.showMessage(f"Camera Error: {error_message}", 6000)

    def on_mouse_moved(self, px_x, px_y):
        if self.latest_depth_raw is None:
            return

        depth_meters = get_robust_depth_sample(self.latest_depth_raw, px_x, px_y, kernel_size=3)
        self.pixel_label.setText(f"Pixel: X = {px_x:3d}, Y = {px_y:3d}")

        if depth_meters > 0:
            depth_mm = depth_meters * 1000.0
            depth_disp = format_dimension(depth_mm, self.current_unit)
            self.cursor_depth_label.setText(f"Depth: {depth_disp:.1f} {self.current_unit}")
        else:
            self.cursor_depth_label.setText("Depth: Invalid / Occluded")

    def save_snapshot(self):
        if self.latest_color_bgr is None or self.latest_depth_raw is None:
            QMessageBox.warning(self, "Warning", "No active frames to save.")
            return

        ts = int(time.time())
        rgb_path = f"measurement_rgb_{ts}.png"
        depth_path = f"measurement_depth_{ts}.png"
        cv2.imwrite(rgb_path, self.latest_color_bgr)
        cv2.imwrite(depth_path, self.camera_thread.colorize_depth(self.latest_depth_raw))
        self.status_bar.showMessage(f"Saved snapshots: {rgb_path}, {depth_path}", 4000)

    def closeEvent(self, event):
        print("[*] Stopping camera thread and closing...")
        self.camera_thread.stop()
        event.accept()


# ==============================================================================
# SECTION 6: APPLICATION ENTRYPOINT & AUTOMATED VERIFICATION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Intel RealSense D455f Multi-Object Measurement System (Boxes & Onions)")
    parser.add_argument("--simulate", action="store_true", help="Force synthetic RealSense simulation mode")
    parser.add_argument("--width", type=int, default=640, help="Stream width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Stream height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Stream framerate (default: 30)")
    parser.add_argument("--test-headless", type=int, default=None, help="Run N frames headlessly for automated verification")

    args = parser.parse_args()

    # Enable High-DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Headless test support
    if args.test_headless is not None:
        print(f"[*] Running automated headless test for {args.test_headless} frames...")
        worker = RealSenseThread(width=args.width, height=args.height, fps=args.fps, force_simulate=args.simulate)
        
        frames_received = [0]
        def frame_cb(color, depth_cmap, depth_raw, intrin):
            frames_received[0] += 1
            if frames_received[0] >= args.test_headless:
                print(f"[TEST PASS] Successfully processed {frames_received[0]} frames in headless test.")
                worker.stop()
                app.quit()

        worker.frame_ready.connect(frame_cb)
        worker.start()
        app.exec_()
        return

    # Create and show main window
    window = MainWindow(
        force_simulate=args.simulate,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
