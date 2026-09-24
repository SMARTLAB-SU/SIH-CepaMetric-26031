"""
================================================================================
Intel RealSense D455f — Continuous Multi-Object Measurement GUI
================================================================================
Strict Classical Computer Vision + RealSense D455f 3D Geometry (NO AI / NO ML).

Features:
  1. Full-Frame Continuous Multi-Object Scanning (No fixed zone restriction).
  2. Multi-Object Tracking with Persistent Sequential IDs (ID 01, ID 02...).
  3. Real-Time Dimensioning for Boxes, Squares, Triangles, and Circles.
  4. Multi-Frame Temporal Stabilization (Outlier-rejected median).
  5. Live Active Objects Panel & Completed Measurement Log with CSV Export.
  6. Multi-Mode Viewport (RGB + 3D HUD, Canny Edge Map, Depth Colormap).
================================================================================
"""

import os
import time
import csv
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QComboBox, QSlider, QStatusBar,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QFileDialog, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect

from measurement import get_robust_depth_sample, format_dimension
from camera import RealSenseThread, CameraDeviceInfo
from box_tracker import (
    MultiObjectTracker,
    draw_multi_object_tracker_overlay
)


class LiveCameraViewport(QLabel):
    """
    Renders the live video stream with full-frame multi-object detection,
    wireframe overlays, corner markers, and dynamic HUD dimension tags.
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

        # Mouse tracking
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

        # Convert OpenCV BGR to QImage RGB
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
    Main desktop window providing continuous multi-object real-time 3D measurement.
    """

    def __init__(self, force_simulate=False, width=640, height=480, fps=30):
        super().__init__()
        self.width_res = width
        self.height_res = height
        self.fps = fps
        self.force_simulate = force_simulate

        self.setWindowTitle("Intel RealSense D455f — Continuous Multi-Object 3D Measurement System")
        self.resize(1440, 900)
        self.setMinimumSize(1140, 720)

        # Core State & Tracker Engine
        self.current_unit = "mm"  # "mm" or "cm"
        self.view_mode = "INSPECTION"  # "INSPECTION", "CANNY_EDGES", "DEPTH_MAP"

        self.latest_depth_raw = None
        self.latest_color_bgr = None
        self.latest_depth_colormap = None
        self.latest_intrinsics = None

        self.tracker = MultiObjectTracker(baseline_depth_mm=850.0, min_height_mm=12.0)

        # Performance / FPS
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # UI & Camera setup
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
        title_label = QLabel("INTEL REALSENSE D455f — CONTINUOUS MULTI-OBJECT MEASUREMENT")
        title_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff; letter-spacing: 0.5px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.active_count_badge = QLabel("🔍 Active Boxes: 0")
        self.active_count_badge.setStyleSheet(
            "background-color: #14281e; color: #00ff78; padding: 4px 12px; "
            "border-radius: 10px; font-weight: bold; border: 1px solid #00aa50;"
        )
        header_layout.addWidget(self.active_count_badge)

        self.box_count_badge = QLabel("📦 Total Counted: 0")
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

        # Viewport Header with View Mode Selector
        vp_header_layout = QHBoxLayout()
        vp_title = QLabel("FULL-FRAME LIVE CAMERA VIEW (Continuous Multi-Box Detection & Tracking)")
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
        meas_group = QGroupBox("Active Tracked Boxes (Live 3D Dimensions)")
        meas_layout = QVBoxLayout(meas_group)
        meas_layout.setSpacing(6)

        self.active_status_badge = QLabel("SCANNING VIEW — NO BOX DETECTED")
        self.active_status_badge.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #abb2bf; padding: 5px 10px; "
            "background-color: #171a22; border-radius: 6px; border: 1px solid #282e3d;"
        )
        meas_layout.addWidget(self.active_status_badge)

        self.dim_display_box = QLabel("No boxes currently visible in camera view.\nPlace cardboard, wooden, square, or rectangular boxes in front of camera.")
        self.dim_display_box.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #98c379; line-height: 1.4; "
            "padding: 10px; background-color: #0b1710; border-radius: 6px; border: 1px solid #1b4528;"
        )
        meas_layout.addWidget(self.dim_display_box)

        sidebar_layout.addWidget(meas_group)

        # 2B-2. DETECTION THRESHOLD & CALIBRATION
        settings_group = QGroupBox("Detection Sensitivity & Surface Calibration")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(6)

        # Min Height Threshold Slider
        settings_layout.addWidget(QLabel("Min Box Height:"), 0, 0)
        self.min_h_slider = QSlider(Qt.Horizontal)
        self.min_h_slider.setRange(8, 60)
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
        self.calib_display_label = QLabel(f"Camera-to-Floor: {self.tracker.baseline_depth_mm:.0f} mm ({self.tracker.baseline_depth_mm/10.0:.1f} cm)")
        self.calib_display_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #61afef; "
            "padding: 4px 8px; background-color: #0e1622; border-radius: 4px; border: 1px solid #1e3654;"
        )
        self.calib_display_label.setAlignment(Qt.AlignCenter)
        settings_layout.addWidget(self.calib_display_label, 2, 0, 1, 3)

        sidebar_layout.addWidget(settings_group)

        # 2B-3. COMPLETED MEASUREMENTS TABLE
        table_group = QGroupBox("Completed Box Measurements Log")
        table_layout = QVBoxLayout(table_group)
        table_layout.setSpacing(6)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Box ID", "Type", "Length", "Width", "Height", "Volume (L)", "Time"
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
        self.reconnect_cam_btn.clicked.connect(self.reconnect_camera)
        toolbar_layout.addWidget(self.reconnect_cam_btn)

        toolbar_layout.addStretch()

        # Unit Selector
        toolbar_layout.addWidget(QLabel("Measurement Unit:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Millimeters (mm)", "Centimeters (cm)"])
        self.unit_combo.currentIndexChanged.connect(self.on_unit_changed)
        toolbar_layout.addWidget(self.unit_combo)

        # Snapshot
        self.snap_btn = QPushButton("📷 Save Snapshot")
        self.snap_btn.clicked.connect(self.save_snapshot)
        toolbar_layout.addWidget(self.snap_btn)

        main_layout.addWidget(toolbar_frame)

        # 4. Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Full-frame continuous multi-object measurement active.")

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
            
            # Format and display measured distance in mm and cm
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

            # Update Table when new objects achieve stabilized measurement
            if len(newly_completed) > 0:
                for obj_data in newly_completed:
                    self._add_object_to_results_table(obj_data)

            # Update badges
            active_vis = [o for o in active_objects if o.missing_frames == 0]
            self.active_count_badge.setText(f"🔍 Active Objects: {len(active_vis)}")
            self.box_count_badge.setText(f"📦 Total Counted: {self.tracker.total_objects_counted}")

            # Update prominent measurement display card
            self._update_active_objects_card(active_vis)

            # Select View Display
            if self.view_mode == "INSPECTION":
                display_frame = draw_multi_object_tracker_overlay(
                    color_bgr, self.tracker, unit=self.current_unit
                )
            elif self.view_mode == "CANNY_EDGES":
                display_frame = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR) if edge_map is not None else color_bgr
            else:  # DEPTH_MAP
                display_frame = depth_colormap if depth_colormap is not None else color_bgr

            self.camera_viewport.update_frame(display_frame)

            # Real-time Diagnostics in Status Bar
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
        """Updates the prominent Active Objects Live Panel with clean multi-box readouts."""
        if len(active_objects) > 0:
            self.active_status_badge.setText(f"DETECTING & TRACKING {len(active_objects)} BOX{'ES' if len(active_objects) > 1 else ''}")
            self.active_status_badge.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #00ff78; padding: 5px 10px; "
                "background-color: #0b1f13; border-radius: 6px; border: 1px solid #00ff78;"
            )

            lines = []
            u = self.current_unit

            for obj in active_objects[:4]:  # Display up to 4 prominent objects
                dims = obj.get_dimensions(u)
                shape = obj.shape_type
                status_str = "[STABLE]" if obj.is_stable else "[TRACKING...]"
                lines.append(f"▶ {shape} #{obj.obj_id:02d}  {status_str}")

                l = dims.get("length", 0)
                w = dims.get("width", 0)
                h = dims.get("height", 0)
                vol = dims.get("volume_l", 0)
                if u == "mm":
                    lines.append(f"   Dim: {l:4.0f} x {w:4.0f} x {h:4.0f} mm  (Vol: {vol:.2f} L)")
                else:
                    lines.append(f"   Dim: {l:4.1f} x {w:4.1f} x {h:4.1f} cm  (Vol: {vol:.2f} L)")

                lines.append("")

            if len(active_objects) > 4:
                lines.append(f"... and {len(active_objects) - 4} more boxes tracked.")

            self.dim_display_box.setText("\n".join(lines).strip())
        else:
            self.active_status_badge.setText("SCANNING VIEW — NO BOX DETECTED")
            self.active_status_badge.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #abb2bf; padding: 5px 10px; "
                "background-color: #171a22; border-radius: 6px; border: 1px solid #282e3d;"
            )
            self.dim_display_box.setText(
                "No boxes currently visible in camera view.\n"
                "The entire camera view is actively scanning.\n"
                "Boxes measure and track automatically upon entry."
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

        l_text = f"{obj_data.get('length', 0):.0f} {u}" if u == "mm" else f"{obj_data.get('length', 0):.1f} {u}"
        w_text = f"{obj_data.get('width', 0):.0f} {u}" if u == "mm" else f"{obj_data.get('width', 0):.1f} {u}"
        h_text = f"{obj_data.get('height', 0):.0f} {u}" if u == "mm" else f"{obj_data.get('height', 0):.1f} {u}"

        vol_val = obj_data.get("volume_l", 0.0)
        vol_text = f"{vol_val:.2f} L" if vol_val > 0 else "--"
        time_text = obj_data.get("timestamp", time.strftime("%H:%M:%S"))

        items = [obj_id_text, shape_text, l_text, w_text, h_text, vol_text, time_text]
        for col_idx, val in enumerate(items):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            if col_idx == 0:
                item.setForeground(QColor("#61afef"))
            elif col_idx == 1:
                item.setForeground(QColor("#ffd700"))
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
                writer.writerow(["Object ID", "Shape", "Length/Dia", "Width/Rad", "Height", "Volume (L)", "Timestamp"])
                for b in self.tracker.completed_objects_log:
                    obj_id_val = b.get("obj_id", b.get("box_id", 1))
                    shape = b.get("shape_type", "BOX")
                    l_val = b.get("length", b.get("diameter", b.get("side1", 0)))
                    w_val = b.get("width", b.get("radius", b.get("side2", 0)))
                    h_val = b.get("height", 0)
                    vol_val = b.get("volume_l", 0.0)
                    writer.writerow([
                        f"ID #{obj_id_val:02d}",
                        shape,
                        f"{l_val:.1f}",
                        f"{w_val:.1f}",
                        f"{h_val:.1f}",
                        f"{vol_val:.2f}",
                        b.get("timestamp", "")
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
