"""
================================================================================
Continuous Multi-Box RealSense D455f 3D Detection & Dimensioning Module (NO AI)
================================================================================
Strict Classical Computer Vision + RealSense D455f 3D Depth Geometry.
Target: Strictly physical BOX-SHAPED objects (Cardboard, Wooden, Square, Rectangular).
Detects: ALL separable boxes (1, 2, 3, 4, 5+ boxes) simultaneously across full FOV.
Rejects: Humans, people, body parts, arms, legs, curved surfaces, and irregular contours.

Uses pure classical OpenCV contour analysis, dual color/depth edge barriers,
multi-depth region segmentation, convexity/solidity filtering, and 3D depth planarity.
================================================================================
"""

import math
import numpy as np
import cv2

from measurement import (
    get_robust_depth_sample,
    deproject_pixel_to_3d_point,
    calculate_3d_distance,
    format_dimension
)


class DetectedObjectCandidate:
    """Represents a validated physical box candidate extracted in the current frame."""

    def __init__(self, obj_id=1, shape_type="BOX", dimensions=None, unit="mm",
                 contour=None, center_2d=(0, 0), center_3d=None,
                 corners_2d=None, corners_3d=None, bbox=(0, 0, 0, 0),
                 top_depth_m=0.5, support_depth_m=0.85):
        self.obj_id = obj_id
        self.shape_type = shape_type  # "BOX" or "SQUARE"
        self.dimensions = dimensions or {}
        self.unit = unit
        self.contour = contour
        self.center_2d = center_2d
        self.center_3d = center_3d
        self.corners_2d = corners_2d or []
        self.corners_3d = corners_3d or []
        self.bbox = bbox  # (x, y, w, h)
        self.top_depth_m = top_depth_m
        self.support_depth_m = support_depth_m

    def get_display_text(self):
        """Returns a clean formatted block for GUI display."""
        u = self.unit
        lines = [f"{self.shape_type} #{self.obj_id}"]

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
    Detects and measures ALL physical box-shaped objects (Cardboard, Wooden, Square, Rectangular)
    simultaneously across the entire camera field of view (conveyor, table, floor, or air).
    Strictly filters out human bodies, limbs, and non-box irregular contours.
    """

    @staticmethod
    def detect_and_measure(color_bgr, depth_raw, intrinsics=None,
                           min_depth_mm=120, max_depth_mm=2600,
                           min_height_mm=10.0,
                           unit="mm", min_contour_area=280):
        """
        Full-frame multi-box detection and 3D physical dimensioning.
        Processes and returns ALL valid box candidates in the frame.

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

        # Depth discontinuity gradient (finds physical height jumps >= 14mm)
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        depth_grad = cv2.morphologyEx(depth_raw, cv2.MORPH_GRADIENT, kernel_3)
        depth_step_edges = ((depth_grad >= 14) & valid_depth_mask).astype(np.uint8) * 255

        # Combined edge map
        combined_edges = cv2.bitwise_or(canny_edges, depth_step_edges)

        # ----------------------------------------------------------------------
        # 2. Multi-Depth Region Segmentation (No single background cutoff)
        # ----------------------------------------------------------------------
        valid_u8 = valid_depth_mask.astype(np.uint8) * 255

        # Edge barriers dilated slightly to separate adjacent/touching boxes
        edge_barriers = cv2.dilate(depth_step_edges, kernel_3, iterations=1)

        # Segment regions by subtracting edge boundaries from valid depth areas
        seg_mask = cv2.subtract(valid_u8, edge_barriers)
        seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel_3, iterations=1)

        # ----------------------------------------------------------------------
        # 3. Find All Contours Across View (Using RETR_LIST to catch all regions)
        # ----------------------------------------------------------------------
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        raw_candidates = []
        rejected_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and whole-frame boundaries
            if area < min_contour_area or area > (w * h * 0.80):
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Skip outer whole-frame background border
            if bx <= 4 and by <= 4 and (bx + bw) >= (w - 4) and (by + bh) >= (h - 4):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            # Minimum area rotated rectangle
            min_rect = cv2.minAreaRect(cnt)
            rect_w, rect_h = min_rect[1]
            rect_area = rect_w * rect_h
            if rect_area <= 0:
                continue

            rectangularity = area / rect_area

            # ------------------------------------------------------------------
            # 4. Box-Specific Geometric Filtering (Rejects Humans & Blobs)
            # ------------------------------------------------------------------
            is_box_geom, approx_pts, geom_reason = MultiObjectDetector._check_box_geometry(
                cnt=cnt,
                perimeter=perimeter,
                area=area,
                rect_w=rect_w,
                rect_h=rect_h,
                rectangularity=rectangularity
            )

            if not is_box_geom:
                rejected_count += 1
                continue

            # Center calculation
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = bx + bw // 2, by + bh // 2

            # ------------------------------------------------------------------
            # 5. Precise Elevation & Supporting Depth Verification
            # ------------------------------------------------------------------
            # Sample interior depth and surrounding outer ring depth
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

            if len(inner_depths) < 20:
                rejected_count += 1
                continue

            top_depth_mm = float(np.median(inner_depths))
            top_depth_m = top_depth_mm / 1000.0

            valid_outer = outer_depths[outer_depths > (top_depth_mm + 8.0)]
            if len(valid_outer) >= 10:
                support_depth_mm = float(np.median(valid_outer))
            else:
                # No deeper supporting surface behind candidate -> background
                support_depth_mm = top_depth_mm

            support_depth_m = support_depth_mm / 1000.0
            height_mm = max(0.0, support_depth_mm - top_depth_mm)

            # Reject if candidate is on the background surface without elevation
            if height_mm < min_height_mm:
                rejected_count += 1
                continue

            # ------------------------------------------------------------------
            # 6. 3D Depth Planarity & Surface Solidity Check (Rejects Humans)
            # ------------------------------------------------------------------
            is_planar, depth_std, depth_reason = MultiObjectDetector._check_depth_planarity(
                depth_raw=depth_raw,
                cnt=cnt,
                bx=bx, by=by, bw=bw, bh=bh,
                top_depth_mm=top_depth_mm
            )

            if not is_planar:
                rejected_count += 1
                continue

            # Deproject center to 3D camera coordinates
            center_3d = deproject_pixel_to_3d_point(intrinsics, cx, cy, top_depth_m, w, h)

            # ------------------------------------------------------------------
            # 7. Extract 3D Box Corners & Compute Real-World Dimensions
            # ------------------------------------------------------------------
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

            # Realistic Physical Dimension Filter (Rejects non-box objects / noise)
            if length_mm < 30.0 or length_mm > 2400.0 or width_mm < 30.0 or height_mm > 2500.0:
                rejected_count += 1
                continue

            # Classify Square vs Box
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

        # ----------------------------------------------------------------------
        # 8. Non-Maximum Suppression with Containment Filtering
        # ----------------------------------------------------------------------
        candidates = MultiObjectDetector._apply_nms(raw_candidates, iou_threshold=0.60, containment_threshold=0.65)

        diagnostics = {
            "contours_found": len(contours),
            "candidates_count": len(candidates),
            "raw_candidates_count": len(raw_candidates),
            "rejected_count": rejected_count
        }

        return candidates, combined_edges, diagnostics

    @staticmethod
    def _check_box_geometry(cnt, perimeter, area, rect_w, rect_h, rectangularity):
        """
        Validates whether a 2D contour conforms to rigid box geometry.
        Rejects human bodies, curved contours, limbs, and irregular shapes.
        """
        # 1. Solidity / Convexity Check
        # Box top/front faces are convex polygons (solidity >= 0.76)
        # Human bodies, arms, and clothing folds have deep concavities (solidity < 0.72)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False, None, "Invalid hull area"

        solidity = area / hull_area
        if solidity < 0.76:
            return False, None, f"Low solidity ({solidity:.2f} < 0.76)"

        # 2. Aspect Ratio Limit (Rejects tall human silhouettes & limbs)
        aspect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1e-3)
        if aspect_ratio > 4.2:
            return False, None, f"Extreme aspect ratio ({aspect_ratio:.2f} > 4.2)"

        # 3. Circularity / Isoperimetric Quotient Check (Rejects round heads, spheres, circles)
        # Perfect square circularity = pi/4 ~ 0.7854. All rectangles have circularity <= 0.7854.
        # Circles/ellipses/heads have circularity 0.84 - 1.00.
        if perimeter > 0:
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity > 0.82:
                return False, None, f"High circularity / round shape ({circularity:.2f} > 0.82)"

        # 4. Rectangularity / Fill Extent
        # Physical box faces match the minimum rotated rectangle with high extent
        if rectangularity < 0.60:
            return False, None, f"Low rectangularity ({rectangularity:.2f} < 0.60)"

        # 5. Polygon Approximation & Corner Count
        # True box faces are 4-sided quadrilaterals with 4 distinct corners
        approx_poly = cv2.approxPolyDP(cnt, 0.026 * perimeter, True)
        if len(approx_poly) != 4:
            approx_poly = cv2.approxPolyDP(cnt, 0.038 * perimeter, True)
            if len(approx_poly) != 4:
                return False, None, f"Non-box vertex count ({len(approx_poly)})"

        # 6. Corner Angle Check for 4-vertex quadrilateral
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
            # In perspective projection, box corners are between 50° and 130°
            if angle_deg < 50.0 or angle_deg > 130.0:
                return False, None, f"Irregular corner angle ({angle_deg:.1f}°)"

        return True, approx_poly, "Valid box geometry"

    @staticmethod
    def _check_depth_planarity(depth_raw, cnt, bx, by, bw, bh, top_depth_mm):
        """
        Verifies that the surface inside the contour is a rigid flat plane (box face)
        and NOT a curved/irregular organic surface (human body, head, arms, clothes).
        """
        roi_depth = depth_raw[by:by + bh, bx:bx + bw]
        roi_mask = np.zeros((bh, bw), dtype=np.uint8)
        shifted_cnt = cnt - np.array([bx, by])
        cv2.drawContours(roi_mask, [shifted_cnt], -1, 255, -1)

        # Depth across the entire object mask
        surf_depths = roi_depth[(roi_mask == 255) & (roi_depth > 0)]

        if len(surf_depths) < 20:
            return False, 0.0, "Insufficient valid surface depth samples"

        std_z = float(np.std(surf_depths))

        # A flat rigid box face has uniform depth with low standard deviation (std <= 36 mm)
        if std_z <= 36.0:
            return True, std_z, "Planar flat surface (low std)"

        # If slightly tilted, check least-squares plane fit residual
        y_idx, x_idx = np.where((roi_mask == 255) & (roi_depth > 0))
        if len(x_idx) > 30:
            z_vals = roi_depth[y_idx, x_idx].astype(np.float32)
            A = np.column_stack([x_idx, y_idx, np.ones_like(x_idx)])
            plane_params, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
            fitted_z = A @ plane_params
            residuals = z_vals - fitted_z
            plane_rms = float(np.sqrt(np.mean(residuals ** 2)))

            # Rigid flat box has plane fit RMS <= 28 mm
            # Human body has RMS >= 45 - 120 mm
            if plane_rms <= 28.0:
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

                # Compute Intersection
                inter_x1 = max(cand_box[0], kept_box[0])
                inter_y1 = max(cand_box[1], kept_box[1])
                inter_x2 = min(cand_box[2], kept_box[2])
                inter_y2 = min(cand_box[3], kept_box[3])

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    iou = inter_area / float(cand_area + kept_area - inter_area)
                    containment = inter_area / float(min(cand_area, kept_area))

                    # If candidate has high IoU OR is largely contained inside a larger box of similar depth
                    depth_diff = abs(cand.top_depth_m - kept_cand.top_depth_m) * 1000.0
                    if iou >= iou_threshold or (containment >= containment_threshold and depth_diff < 40.0):
                        duplicate = True
                        break

            if not duplicate:
                keep.append(cand)

        return keep


def draw_multi_object_overlays(image_bgr, tracked_objects, unit="mm"):
    """
    Renders clean, high-visibility wireframes, corner markers, center crosshairs,
    and dynamic HUD tags on every active tracked box across the complete camera view.
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

        # High-visibility palette for boxes and squares
        color = (0, 240, 120) if is_stable else (0, 220, 255)  # Green (Stable) / Cyan (Tracking)
        fill_color = (0, 180, 80) if is_stable else (0, 160, 200)
        status_str = "[STABLE]" if is_stable else "[TRACKING]"

        l = dims.get("length", 0.0)
        w_val = dims.get("width", 0.0)
        h_val = dims.get("height", 0.0)
        tag_label = f"{shape} #{obj_id:02d} | {l:.0f}x{w_val:.0f}x{h_val:.0f} {u} {status_str}" if u == "mm" else f"{shape} #{obj_id:02d} | {l:.1f}x{w_val:.1f}x{h_val:.1f} {u} {status_str}"

        # Draw rotated rectangle wireframe & tinted polygon
        if len(obj.corners_2d) >= 4:
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

        # High-tech HUD dimension pill banner
        bx, by, bw, bh = obj.bbox
        text_x = max(10, min(w - 240, bx))
        text_y = max(24, by - 8)

        (tw, th), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(annotated, (text_x - 3, text_y - th - 5), (text_x + tw + 6, text_y + 3), (10, 14, 20), -1)
        cv2.rectangle(annotated, (text_x - 3, text_y - th - 5), (text_x + tw + 6, text_y + 3), color, 1)
        cv2.putText(annotated, tag_label, (text_x + 2, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

    # Top-Left Live Status Badge
    active_count = len([o for o in tracked_objects if getattr(o, "missing_frames", 0) == 0])
    if active_count > 0:
        badge_text = f"ACTIVE BOXES: {active_count} (Continuous Multi-Box Tracking)"
        badge_color = (0, 240, 120)
    else:
        badge_text = "SCANNING VIEW — NO BOX DETECTED"
        badge_color = (170, 180, 190)

    cv2.putText(annotated, badge_text, (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_color, 2)

    return annotated
