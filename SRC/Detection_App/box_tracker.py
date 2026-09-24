"""
================================================================================
Continuous Multi-Box Tracking & 3D Dimensioning Engine (NO AI / NO ML)
================================================================================
Strict Classical Computer Vision + RealSense D455f 3D Depth Geometry.

Features:
  1. Full-Frame Continuous Scanning: Detects and tracks boxes anywhere in camera view.
  2. Velocity-Compensated Multi-Box Association: Keeps persistent IDs on moving boxes.
  3. Consistent Persistent Sequential IDs (ID 01, ID 02, etc.) without duplication.
  4. Complete Lifecycle: NEW -> TRACKING -> MEASURING -> STABLE -> LEAVING -> REMOVED.
  5. Instant Frame 1 Detection & Temporal Measurement Smoothing (Moving median + EMA).
  6. Corner and Bounding Box Temporal Smoothing to eliminate visual jitter.
  7. Brief Disappearance Grace Period: Reconnects tracks if temporarily occluded.
================================================================================
"""

import math
import time
import numpy as np
import cv2

from measurement import format_dimension
from detection import MultiObjectDetector, DetectedObjectCandidate, draw_multi_object_overlays


class TrackedObject:
    """
    Represents a unique physical box tracked continuously across consecutive video frames.
    Maintains persistent tracking ID, spatial motion history, and stabilized 3D dimensions.
    """

    def __init__(self, obj_id=1, shape_type="BOX", unit="mm"):
        self.obj_id = obj_id
        self.shape_type = shape_type  # "BOX" or "SQUARE"
        self.unit = unit
        self.state = "TRACKING"

        # Visual & spatial geometry
        self.contour = None
        self.corners_2d = []
        self.corners_3d = []
        self.center_2d = (0, 0)
        self.center_3d = None
        self.bbox = (0, 0, 0, 0)
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

        # Live smoothed measurements (raw mm)
        self.live_dims_mm = {}
        # Locked/Stabilized measurements (raw mm)
        self.stable_dims_mm = {}
        self.is_stable = False
        self.timestamp = ""

        # Multi-frame measurement buffer for stabilization (outlier rejection)
        self._sample_history = {
            "length": [],
            "width": [],
            "height": [],
            "depth": []
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

        # Estimate velocity from centroid displacement
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

        # Temporal smoothing for 2D corners (reduces wireframe jitter on moving boxes)
        if len(self.corners_2d) == 4 and len(candidate.corners_2d) == 4:
            smoothed_corners = []
            cand_pts = candidate.corners_2d
            for i in range(4):
                # Match to closest candidate corner
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
        self.bbox = candidate.bbox
        self.top_depth_m = candidate.top_depth_m
        self.support_depth_m = candidate.support_depth_m

        self.centroid_history.append(self.center_2d)
        if len(self.centroid_history) > 40:
            self.centroid_history.pop(0)

        # Update measurement history buffers
        for key, val in candidate.dimensions.items():
            val_mm = val * 10.0 if unit == "cm" else val
            if key not in self._sample_history:
                self._sample_history[key] = []
            self._sample_history[key].append(val_mm)
            if len(self._sample_history[key]) > 15:
                self._sample_history[key].pop(0)

            # Fast exponential smoothing (alpha = 0.45)
            if key not in self.live_dims_mm or self.live_dims_mm[key] <= 0:
                self.live_dims_mm[key] = val_mm
            else:
                self.live_dims_mm[key] = 0.45 * val_mm + 0.55 * self.live_dims_mm[key]

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
        # Decay velocity
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

        self.is_stable = (self.frames_tracked >= 3)
        if not self.timestamp:
            self.timestamp = time.strftime("%H:%M:%S")

    def get_dimensions(self, target_unit="mm"):
        """Returns clean formatted dimensions in requested unit for UI and logging."""
        use_dict = self.stable_dims_mm if self.is_stable and len(self.stable_dims_mm) > 0 else self.live_dims_mm
        formatted = {}

        for key, val_mm in use_dict.items():
            formatted[key] = format_dimension(val_mm, target_unit)

        # Compute volume in Liters
        l_mm = use_dict.get("length", 0.0)
        w_mm = use_dict.get("width", 0.0)
        h_mm = use_dict.get("height", 0.0)
        vol_liters = ((l_mm / 10.0) * (w_mm / 10.0) * (h_mm / 10.0)) / 1000.0
        formatted["volume_l"] = vol_liters

        formatted["obj_id"] = self.obj_id
        formatted["box_id"] = self.obj_id  # compatibility
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
    Continuous Multi-Box Tracking and Stabilization Engine.
    Operates full-frame with velocity-assisted persistent tracking.
    """

    def __init__(self, baseline_depth_mm=850.0, min_height_mm=12.0):
        self.baseline_depth_mm = baseline_depth_mm
        self.min_height_mm = min_height_mm
        self.active_objects = []          # List of active TrackedObject instances
        self.total_objects_counted = 0
        self.completed_objects_log = []   # History of all stabilized / completed objects
        self.logged_object_ids = set()    # Prevent duplicate entries in log
        self.latest_diagnostics = {}

    @property
    def active_boxes(self):
        """Compatibility property."""
        return self.active_objects

    @property
    def completed_boxes_log(self):
        """Compatibility property."""
        return self.completed_objects_log

    @property
    def total_boxes_counted(self):
        """Compatibility property."""
        return self.total_objects_counted

    @property
    def current_zone_box(self):
        """Returns the primary active box (first stable or visible)."""
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
        Continuously executes full-frame box detection, motion-assisted tracking association,
        and dimension stabilization across the entire camera field of view.

        Returns:
            active_objects: list[TrackedObject]
            primary_object: TrackedObject
            newly_completed: list[dict]
            edge_map: np.ndarray
            diagnostics: dict
        """
        if color_bgr is None or depth_raw is None:
            return [], None, [], None, {}

        t0 = time.perf_counter()
        h, w = depth_raw.shape[:2]

        # 1. Full-Frame Detection of all box candidates
        candidates, edge_map, det_diag = MultiObjectDetector.detect_and_measure(
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            intrinsics=intrinsics,
            min_height_mm=self.min_height_mm,
            unit=unit
        )

        newly_completed = []

        # 2. Multi-Metric Non-AI Object Association with Velocity Compensation & IoU
        matched_cand_indices = set()
        matched_obj_indices = set()

        if len(self.active_objects) > 0 and len(candidates) > 0:
            cost_matrix = np.zeros((len(self.active_objects), len(candidates)), dtype=np.float32)

            for o_idx, obj in enumerate(self.active_objects):
                # Use motion-predicted centroid if available
                pred_x = obj.predicted_center[0] if obj.frames_tracked > 1 else obj.center_2d[0]
                pred_y = obj.predicted_center[1] if obj.frames_tracked > 1 else obj.center_2d[1]

                for c_idx, cand in enumerate(candidates):
                    # Motion-compensated 2D centroid distance
                    dist_2d = math.hypot(pred_x - cand.center_2d[0], pred_y - cand.center_2d[1])

                    # 3D Depth Difference (converted to mm)
                    depth_diff_mm = abs(obj.top_depth_m - cand.top_depth_m) * 1000.0

                    # 2D Bounding Box IoU
                    iou = compute_bbox_iou(obj.bbox, cand.bbox)
                    overlap_bonus = -40.0 * iou if iou > 0.10 else 0.0

                    # Bounding Box Size similarity
                    bw_diff = abs(obj.bbox[2] - cand.bbox[2])
                    bh_diff = abs(obj.bbox[3] - cand.bbox[3])

                    cost_matrix[o_idx, c_idx] = dist_2d + (depth_diff_mm * 0.20) + overlap_bonus + (bw_diff + bh_diff) * 0.10

            # Match greedily based on minimum cost
            max_match_cost = 250.0
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
                # Check if there is ANY surviving active object that was not matched this frame
                # but is located at or near this candidate (reconnecting dropped tracks)
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
                    # Truly a NEW physical box in an unoccupied location!
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

        # 4. Handle Missing Frames & Object Removal / Leaving
        surviving_objects = []
        for o_idx, obj in enumerate(self.active_objects):
            if o_idx not in matched_obj_indices:
                obj.extrapolate_position()

                # Check if near frame border (Leaving state)
                cx, cy = obj.center_2d
                is_near_border = (cx < 25 or cx > (w - 25) or cy < 25 or cy > (h - 25))
                if is_near_border:
                    obj.state = "LEAVING"

                # Drop after 4 frames if leaving near border, or 8 frames (~0.25 sec) if inside view
                max_miss = 4 if is_near_border else 8
                if obj.missing_frames <= max_miss:
                    surviving_objects.append(obj)
                else:
                    obj.state = "REMOVED"
                    # Log final measurement if not already logged and was tracked
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


# Backwards compatibility alias
MultiBoxConveyorTracker = MultiObjectTracker


def draw_multi_object_tracker_overlay(image_bgr, tracker: MultiObjectTracker, unit="mm"):
    """
    Renders live full-frame tracking wireframes, corner markers, and dimension HUD tags.
    """
    return draw_multi_object_overlays(image_bgr, tracker.active_objects, unit=unit)


# Backwards compatibility alias
draw_conveyor_inspection_overlay = draw_multi_object_tracker_overlay
