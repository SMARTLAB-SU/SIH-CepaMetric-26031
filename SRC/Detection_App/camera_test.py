#!/usr/bin/env python3
"""
================================================================================
Intel RealSense D455f - Camera & Depth Verification Test (Phase 1 & Phase 2)
================================================================================
Strict Classical Computer Vision + RealSense SDK (NO AI / ML Models).

Features:
  1. Detects Intel RealSense D455f hardware and queries device metadata.
  2. Streams synchronized Color (RGB) and Depth streams at 640x480 @ 30 FPS.
  3. Aligns Depth frame to Color frame (rs.align(rs.stream.color)).
  4. Real-time mouse tracking: Displays Pixel (X, Y) and Depth in millimeters.
  5. Extracts 3D coordinates (X, Y, Z in mm) via 2D->3D deprojection.
  6. Renders side-by-side RGB & Colorized Depth visualization with interactive HUD.
  7. Includes a high-fidelity synthetic simulation mode (--simulate) for offline testing.

Controls:
  - Move Mouse : Inspect pixel (X, Y), Depth (mm), and 3D coordinates (X, Y, Z).
  - Left Click : Print exact coordinates and measurement snapshot to terminal.
  - 'c' Key    : Cycle depth colormap (Jet, Turbo, RealSense Colorizer, Ocean).
  - 'h' Key    : Toggle HUD overlay on/off.
  - 's' Key    : Save snapshot (RGB, Depth map, and metadata log).
  - 'q' / ESC  : Quit.
================================================================================
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

# Attempt to import pyrealsense2
try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False


class CameraDiagnostics:
    """Utility class to query and print detailed RealSense camera diagnostics."""

    @staticmethod
    def print_banner():
        banner = """
================================================================================
      INTEL REALSENSE D455f DEPTH & RGB CAMERA TEST (NO AI / NO ML)
================================================================================
        """
        print(banner)

    @staticmethod
    def query_device_info():
        """Queries all attached RealSense devices and displays metadata."""
        if not PYREALSENSE_AVAILABLE:
            print("[WARN] pyrealsense2 is not installed or available.")
            return None

        ctx = rs.context()
        devices = ctx.query_devices()
        print(f"[*] Scanning USB ports for RealSense devices... Found: {len(devices)}")

        if len(devices) == 0:
            return None

        for idx, dev in enumerate(devices):
            name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Unknown"
            serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "Unknown"
            fw = dev.get_info(rs.camera_info.firmware_version) if dev.supports(rs.camera_info.firmware_version) else "Unknown"
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"
            product_id = dev.get_info(rs.camera_info.product_id) if dev.supports(rs.camera_info.product_id) else "Unknown"
            
            print(f"\n--- RealSense Device #{idx + 1} ---")
            print(f"  Device Name      : {name}")
            print(f"  Serial Number    : {serial}")
            print(f"  Firmware Version : {fw}")
            print(f"  USB Type         : USB {usb_type}")
            print(f"  Product ID       : {product_id}")

            # Check sensors
            sensors = dev.query_sensors()
            print(f"  Sensors ({len(sensors)}):")
            for s in sensors:
                s_name = s.get_info(rs.camera_info.name) if s.supports(rs.camera_info.name) else "Sensor"
                print(f"    - {s_name}")
                if s.is_depth_sensor():
                    depth_scale = s.as_depth_sensor().get_depth_scale()
                    print(f"      * Depth Scale: {depth_scale} m/unit ({depth_scale * 1000.0:.3f} mm/unit)")

        print("--------------------------------------------------------------------------------\n")
        return devices[0]


class SimulatedRealSenseCamera:
    """
    High-fidelity RealSense Simulator for testing without physical hardware.
    Generates aligned RGB and Depth maps with realistic geometric objects.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale = 0.001  # 1 mm per unit

        # Realistic D455f optical intrinsics (FOV ~ 86° x 57°)
        self.fx = 385.0
        self.fy = 385.0
        self.ppx = width / 2.0
        self.ppy = height / 2.0

        # Simulated depth intrinsics object
        class MockIntrinsics:
            def __init__(mock_self, w, h, fx, fy, ppx, ppy):
                mock_self.width = w
                mock_self.height = h
                mock_self.fx = fx
                mock_self.fy = fy
                mock_self.ppx = ppx
                mock_self.ppy = ppy
                mock_self.model = rs.distortion.none if PYREALSENSE_AVAILABLE else "None"
                mock_self.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.intrinsics = MockIntrinsics(width, height, self.fx, self.fy, self.ppx, self.ppy)
        self.start_time = time.time()
        print("[SIM] Simulated RealSense D455f Camera Initialized (640x480 @ 30 FPS).")

    def get_frames(self):
        """Generates synthetic aligned RGB and Depth frames."""
        time.sleep(1.0 / self.fps)
        elapsed = time.time() - self.start_time

        # Background: table surface at depth 850 mm (0.85 m)
        depth_map = np.full((self.height, self.width), 850, dtype=np.uint16)
        color_image = np.full((self.height, self.width, 3), (215, 220, 225), dtype=np.uint8)

        # Draw grid lines on the table
        for y in range(0, self.height, 40):
            cv2.line(color_image, (0, y), (self.width, y), (190, 195, 200), 1)
        for x in range(0, self.width, 40):
            cv2.line(color_image, (x, 0), (x, self.height), (190, 195, 200), 1)

        # 1. Rectangle Object: 140 mm x 80 mm at depth 520 mm
        rect_x, rect_y, rect_w, rect_h = 100, 140, 120, 80
        depth_map[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 520
        cv2.rectangle(color_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (60, 140, 220), -1)
        cv2.rectangle(color_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (20, 80, 180), 2)
        cv2.putText(color_image, "Rectangle", (rect_x + 5, rect_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 80, 180), 1)

        # 2. Square Object: 90 mm x 90 mm at depth 480 mm
        sq_x, sq_y, sq_s = 280, 220, 80
        depth_map[sq_y:sq_y + sq_s, sq_x:sq_x + sq_s] = 480
        cv2.rectangle(color_image, (sq_x, sq_y), (sq_x + sq_s, sq_y + sq_s), (70, 180, 90), -1)
        cv2.rectangle(color_image, (sq_x, sq_y), (sq_x + sq_s, sq_y + sq_s), (30, 120, 40), 2)
        cv2.putText(color_image, "Square", (sq_x + 5, sq_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 120, 40), 1)

        # 3. Circular Object: Diameter 90 mm at depth 450 mm
        circ_cx, circ_cy, circ_r = 480, 160, 45
        cv2.circle(color_image, (circ_cx, circ_cy), circ_r, (180, 70, 160), -1)
        cv2.circle(color_image, (circ_cx, circ_cy), circ_r, (120, 30, 100), 2)
        cv2.putText(color_image, "Circle", (circ_cx - 25, circ_cy - circ_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 30, 100), 1)
        
        # Mask depth for circle
        y_grid, x_grid = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((x_grid - circ_cx) ** 2 + (y_grid - circ_cy) ** 2)
        depth_map[dist_from_center <= circ_r] = 450

        # 4. Triangular Object at depth 500 mm
        tri_pts = np.array([[380, 340], [330, 430], [430, 430]], np.int32)
        cv2.fillPoly(color_image, [tri_pts], (220, 160, 50))
        cv2.polylines(color_image, [tri_pts], isClosed=True, color=(160, 100, 20), thickness=2)
        cv2.putText(color_image, "Triangle", (350, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 100, 20), 1)
        
        tri_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(tri_mask, [tri_pts], 255)
        depth_map[tri_mask == 255] = 500

        # Add slight natural sensor noise (realistic depth variation +- 1-2 mm)
        noise = np.random.normal(0, 1.2, depth_map.shape).astype(np.int16)
        noisy_depth = np.clip(depth_map.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

        # Mock frame object with get_distance method
        class MockDepthFrame:
            def __init__(self, raw_depth):
                self._depth = raw_depth

            def get_distance(self, x, y):
                if 0 <= x < 640 and 0 <= y < 480:
                    val_mm = self._depth[y, x]
                    return float(val_mm) / 1000.0  # Return meters
                return 0.0

        return color_image, noisy_depth, MockDepthFrame(noisy_depth)


class RealSenseCameraStream:
    """
    Manages the Intel RealSense D455f pipeline, streams, alignment, and intrinsics.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = 0.001
        self.depth_intrinsics = None
        self.color_intrinsics = None
        self.is_running = False

    def start(self):
        """Initializes and starts the RealSense pipeline with aligned color/depth streams."""
        if not PYREALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 library is not installed.")

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable Depth and Color streams
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Start streaming pipeline
        print(f"[*] Starting Intel RealSense pipeline ({self.width}x{self.height} @ {self.fps} FPS)...")
        profile = self.pipeline.start(self.config)

        # Get depth sensor scale (e.g. 0.001 for D455f)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[+] Depth Scale: {self.depth_scale:.6f} meters/unit ({self.depth_scale * 1000:.2f} mm/unit)")

        # Create align object to align depth frames to color viewport
        # align_to = rs.stream.color
        self.align = rs.align(rs.stream.color)

        # Obtain optical intrinsics from aligned depth stream
        depth_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

        self.depth_intrinsics = depth_stream_profile.get_intrinsics()
        self.color_intrinsics = color_stream_profile.get_intrinsics()

        print("\n--- Optical Intrinsics (Color / Aligned Depth) ---")
        print(f"  Focal Length (fx, fy)     : ({self.color_intrinsics.fx:.2f}, {self.color_intrinsics.fy:.2f}) px")
        print(f"  Principal Point (ppx, ppy): ({self.color_intrinsics.ppx:.2f}, {self.color_intrinsics.ppy:.2f}) px")
        print(f"  Distortion Model          : {self.color_intrinsics.model}")
        print(f"  Distortion Coeffs         : {self.color_intrinsics.coeffs}")
        print("--------------------------------------------------\n")

        self.is_running = True
        return True

    def get_aligned_frames(self):
        """
        Polls the pipeline, aligns depth to color, and returns numpy arrays and depth frame.
        """
        if not self.is_running:
            return None, None, None

        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames(5000)

        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None, None

        # Update intrinsics from aligned stream if needed
        self.depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image, aligned_depth_frame

    def stop(self):
        """Stops the RealSense streaming pipeline safely."""
        if self.is_running and self.pipeline:
            print("[*] Stopping RealSense pipeline...")
            self.pipeline.stop()
            self.is_running = False


class RealSenseApp:
    """
    Main interactive test application: handles display, mouse events,
    2D-to-3D deprojection, and HUD rendering.
    """

    def __init__(self, simulate=False, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.simulate = simulate

        self.mouse_x = width // 2
        self.mouse_y = height // 2
        self.locked_points = []
        self.show_hud = True
        self.colormap_idx = 0
        self.colormaps = [
            ("COLORMAP_JET", cv2.COLORMAP_JET),
            ("COLORMAP_TURBO", cv2.COLORMAP_TURBO),
            ("COLORMAP_OCEAN", cv2.COLORMAP_OCEAN),
            ("COLORMAP_BONE", cv2.COLORMAP_BONE),
        ]

        self.camera = None
        self.window_name = "Intel RealSense D455f - Camera & Depth Verification Test"

        # Frame rate calculation
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

    def deproject_pixel_to_point(self, intrinsics, pixel_x, pixel_y, depth_meters):
        """
        Converts 2D pixel coordinates and depth distance into 3D real-world coordinates (X, Y, Z).
        Uses RealSense deprojection: rs.rs2_deproject_pixel_to_point().
        """
        if depth_meters <= 0:
            return None

        if PYREALSENSE_AVAILABLE and hasattr(intrinsics, "fx"):
            # Native RealSense SDK deprojection
            try:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [float(pixel_x), float(pixel_y)], float(depth_meters))
                return np.array(point_3d) * 1000.0  # Convert meters to millimeters (X, Y, Z)
            except Exception:
                pass

        # Standard pinhole camera model fallback:
        # X = (x - ppx) * Z / fx
        # Y = (y - ppy) * Z / fy
        # Z = depth
        fx = getattr(intrinsics, "fx", 385.0)
        fy = getattr(intrinsics, "fy", 385.0)
        ppx = getattr(intrinsics, "ppx", self.width / 2.0)
        ppy = getattr(intrinsics, "ppy", self.height / 2.0)

        z_mm = depth_meters * 1000.0
        x_mm = (pixel_x - ppx) * z_mm / fx
        y_mm = (pixel_y - ppy) * z_mm / fy

        return np.array([x_mm, y_mm, z_mm])

    def mouse_callback(self, event, x, y, flags, param):
        """Tracks mouse movement and user clicks on the OpenCV display window."""
        # Handle side-by-side coordinate translation:
        # If clicked on right panel (depth map), map x coordinate back to [0, width)
        actual_x = x % self.width if x < self.width * 2 else min(x, self.width - 1)
        actual_y = min(y, self.height - 1)

        self.mouse_x = max(0, min(self.width - 1, actual_x))
        self.mouse_y = max(0, min(self.height - 1, actual_y))

        if event == cv2.EVENT_LBUTTONDOWN:
            depth_meters = param["get_depth_func"](self.mouse_x, self.mouse_y)
            intrinsics = param["intrinsics"]
            pt3d = self.deproject_pixel_to_point(intrinsics, self.mouse_x, self.mouse_y, depth_meters)

            print(f"\n[CLICKED POINT INSPECTION]")
            print(f"  Pixel Coordinate : (X: {self.mouse_x}, Y: {self.mouse_y})")
            if depth_meters > 0:
                print(f"  Measured Depth   : {depth_meters * 1000.0:.1f} mm  ({depth_meters:.3f} m)")
                if pt3d is not None:
                    print(f"  3D Coordinates   : X = {pt3d[0]:+.1f} mm, Y = {pt3d[1]:+.1f} mm, Z = {pt3d[2]:.1f} mm")
            else:
                print("  Measured Depth   : 0 mm (Invalid / Out of Range / Occluded)")

    def colorize_depth_image(self, depth_raw, max_distance_mm=3000):
        """
        Converts 16-bit depth image (Z16 in mm) into an aesthetic 8-bit colormap.
        Clips max distance to maintain visual clarity and contrast.
        """
        # Clip depth between 100mm and max_distance_mm for optimal visualization
        depth_clipped = np.clip(depth_raw, 100, max_distance_mm)
        # Normalize to 0-255
        depth_scaled = cv2.convertScaleAbs(depth_clipped, alpha=(255.0 / max_distance_mm))
        # Invert so closer objects are warmer/brighter
        depth_inverted = 255 - depth_scaled
        
        colormap_name, colormap_flag = self.colormaps[self.colormap_idx]
        colorized = cv2.applyColorMap(depth_inverted, colormap_flag)

        # Set invalid depth pixels (depth == 0) to black
        colorized[depth_raw == 0] = [0, 0, 0]
        return colorized

    def render_hud(self, canvas, depth_meters, pt3d):
        """Renders information overlay HUD on the display canvas."""
        if not self.show_hud:
            return

        h, w_total = canvas.shape[:2]
        
        # 1. Top Header Bar (Semi-transparent black overlay)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w_total, 54), (15, 15, 20), -1)
        
        # 2. Bottom Status Bar
        cv2.rectangle(overlay, (0, h - 45), (w_total, h), (15, 15, 20), -1)
        
        # 3. Mouse Inspector Card (Top Right)
        card_w, card_h = 360, 105
        card_x, card_y = w_total - card_w - 15, 65
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 30, 35), -1)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 180, 240), 1)

        # Blend overlays
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

        # Title & Stream labels
        cv2.putText(canvas, "Intel RealSense D455f - Depth Measurement System [NO AI/ML]", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        mode_text = "MODE: HARDWARE STREAM" if not self.simulate else "MODE: SIMULATOR"
        cv2.putText(canvas, f"{mode_text} | FPS: {self.current_fps:.1f} | Res: {self.width}x{self.height} | Aligned: True",
                    (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 230, 160), 1)

        # Panel Titles
        cv2.putText(canvas, "[1] RGB Color Stream", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(canvas, f"[2] Aligned Depth Map ({self.colormaps[self.colormap_idx][0]})",
                    (self.width + 20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Crosshairs at mouse position on both RGB and Depth panels
        mx, my = self.mouse_x, self.mouse_y
        for x_offset in [0, self.width]:
            cx = mx + x_offset
            cy = my
            cv2.drawMarker(canvas, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
            cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), 1)

        # Mouse Inspector Card Text
        cv2.putText(canvas, "LIVE PIXEL & DEPTH INSPECTOR", (card_x + 12, card_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 180, 240), 1)
        
        pixel_info = f"Pixel: X = {mx:3d}, Y = {my:3d}"
        cv2.putText(canvas, pixel_info, (card_x + 12, card_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)

        if depth_meters > 0:
            depth_mm = depth_meters * 1000.0
            depth_info = f"Depth: {depth_mm:6.1f} mm  ({depth_meters:.3f} m)"
            cv2.putText(canvas, depth_info, (card_x + 12, card_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if pt3d is not None:
                coord_info = f"3D (XYZ): ({pt3d[0]:+5.1f}, {pt3d[1]:+5.1f}, {pt3d[2]:5.1f}) mm"
                cv2.putText(canvas, coord_info, (card_x + 12, card_y + 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 230, 255), 1)
        else:
            cv2.putText(canvas, "Depth: N/A (Invalid / Occluded)", (card_x + 12, card_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)
            cv2.putText(canvas, "3D (XYZ): No valid depth data", (card_x + 12, card_y + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        # Bottom Controls Guide
        guide = "Controls: [Left Click] Print Details | [c] Change Colormap | [h] Toggle HUD | [s] Snapshot | [q / ESC] Quit"
        cv2.putText(canvas, guide, (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

    def run(self, headless_frames=None):
        """Main execution loop."""
        CameraDiagnostics.print_banner()

        # Check device availability
        detected_device = CameraDiagnostics.query_device_info()

        if not self.simulate and detected_device is None:
            print("[INFO] No physical Intel RealSense device detected on USB.")
            print("  Options:")
            print("  1. Connect your Intel RealSense D455f via USB 3.0 port and re-run.")
            print("  2. Run in simulation mode using: python camera_test.py --simulate\n")
            if headless_frames is None:
                user_choice = input("Would you like to run in SIMULATION mode now? [Y/n]: ").strip().lower()
                if user_choice in ["", "y", "yes"]:
                    self.simulate = True
                else:
                    print("[*] Exiting. Please plug in the camera and run again.")
                    return
            else:
                self.simulate = True

        # Initialize hardware or simulator
        if self.simulate:
            self.camera = SimulatedRealSenseCamera(self.width, self.height, self.fps)
            intrinsics = self.camera.intrinsics
        else:
            self.camera = RealSenseCameraStream(self.width, self.height, self.fps)
            try:
                self.camera.start()
                intrinsics = self.camera.depth_intrinsics
            except Exception as e:
                print(f"[ERROR] Failed to start RealSense camera pipeline: {e}")
                print("[*] Falling back to simulation mode...")
                self.simulate = True
                self.camera = SimulatedRealSenseCamera(self.width, self.height, self.fps)
                intrinsics = self.camera.intrinsics

        # Setup OpenCV display window unless running headless test
        if headless_frames is None:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # Helper callback dict for mouse inspection
        callback_param = {
            "intrinsics": intrinsics,
            "get_depth_func": lambda x, y: 0.0
        }

        frame_count = 0
        print("[*] Streaming started. Move mouse over the display window to inspect depth values.")

        try:
            while True:
                # 1. Fetch aligned frames
                if self.simulate:
                    color_image, depth_raw, depth_frame = self.camera.get_frames()
                    intrinsics = self.camera.intrinsics
                else:
                    color_image, depth_raw, depth_frame = self.camera.get_aligned_frames()
                    intrinsics = self.camera.depth_intrinsics

                if color_image is None or depth_raw is None:
                    continue

                # Update callback function to read from current depth frame
                def get_distance_meters(px, py):
                    if hasattr(depth_frame, "get_distance"):
                        return depth_frame.get_distance(px, py)
                    elif 0 <= py < depth_raw.shape[0] and 0 <= px < depth_raw.shape[1]:
                        return float(depth_raw[py, px]) * 0.001
                    return 0.0

                callback_param["get_depth_func"] = get_distance_meters
                callback_param["intrinsics"] = intrinsics

                if headless_frames is None:
                    cv2.setMouseCallback(self.window_name, self.mouse_callback, callback_param)

                # 2. Colorize depth map
                colorized_depth = self.colorize_depth_image(depth_raw)

                # 3. Stack Color and Depth side-by-side: [RGB | Depth]
                canvas = np.hstack((color_image, colorized_depth))

                # 4. Extract current mouse measurement
                current_depth_m = get_distance_meters(self.mouse_x, self.mouse_y)
                current_pt3d = self.deproject_pixel_to_point(intrinsics, self.mouse_x, self.mouse_y, current_depth_m)

                # 5. FPS update
                frame_count += 1
                self.fps_counter += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (now - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = now

                # 6. Render HUD
                self.render_hud(canvas, current_depth_m, current_pt3d)

                # Headless mode check
                if headless_frames is not None:
                    if frame_count >= headless_frames:
                        print(f"[TEST PASS] Successfully processed {frame_count} frames headlessly.")
                        print(f"  Final Depth at ({self.mouse_x}, {self.mouse_y}): {current_depth_m * 1000.0:.1f} mm")
                        if current_pt3d is not None:
                            print(f"  3D Coordinates (XYZ): {current_pt3d}")
                        break
                    continue

                # 7. Display window
                cv2.imshow(self.window_name, canvas)

                # Handle keyboard inputs
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n[*] Exiting test...")
                    break
                elif key == ord('c'):  # Cycle colormap
                    self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
                    print(f"[*] Colormap switched to: {self.colormaps[self.colormap_idx][0]}")
                elif key == ord('h'):  # Toggle HUD
                    self.show_hud = not self.show_hud
                    print(f"[*] HUD Overlay: {'ON' if self.show_hud else 'OFF'}")
                elif key == ord('s'):  # Save Snapshot
                    ts = int(time.time())
                    rgb_filename = f"snapshot_rgb_{ts}.png"
                    depth_filename = f"snapshot_depth_{ts}.png"
                    cv2.imwrite(rgb_filename, color_image)
                    cv2.imwrite(depth_filename, colorized_depth)
                    print(f"[+] Saved snapshots: '{rgb_filename}' and '{depth_filename}'")

        except KeyboardInterrupt:
            print("\n[*] Stopped by user.")
        finally:
            if not self.simulate and hasattr(self.camera, "stop"):
                self.camera.stop()
            if headless_frames is None:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Intel RealSense D455f Camera & Depth Verification Test (No AI/ML)")
    parser.add_argument("--simulate", action="store_true", help="Run with simulated D455f RGB-D feed")
    parser.add_argument("--width", type=int, default=640, help="Stream width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Stream height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Stream frame rate (default: 30)")
    parser.add_argument("--test-headless", type=int, default=None, help="Run N frames without UI for automated testing")

    args = parser.parse_args()

    app = RealSenseApp(
        simulate=args.simulate,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    app.run(headless_frames=args.test_headless)


if __name__ == "__main__":
    main()
