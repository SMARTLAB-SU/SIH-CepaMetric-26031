"""
================================================================================
3D Spatial Geometry & Dimension Measurement Module (NO AI / NO ML)
================================================================================
Pure classical mathematics, Euclidean distance, neighborhood depth filtering,
and 2D-to-3D optical pinhole deprojection using Intel RealSense intrinsics.
Supports selectable units: Centimeters (cm) and Millimeters (mm).
================================================================================
"""

import math
import numpy as np

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False


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
    if unit.lower() == "cm":
        return val_mm / 10.0
    return val_mm


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
    
    Args:
        point_a_3d: [X1, Y1, Z1] in millimeters.
        point_b_3d: [X2, Y2, Z2] in millimeters.
        
    Returns:
        float: Euclidean distance in millimeters, or None if invalid.
    """
    if point_a_3d is None or point_b_3d is None:
        return None

    p1 = np.asarray(point_a_3d, dtype=np.float64)
    p2 = np.asarray(point_b_3d, dtype=np.float64)

    if len(p1) < 3 or len(p2) < 3:
        return None

    return float(np.linalg.norm(p2 - p1))


def calculate_point_to_line_distance_3d(point_p, line_a, line_b):
    """
    Computes perpendicular distance from 3D point P to line passing through A and B (Triangle Height).
    Formula: ||(P - A) x (P - B)|| / ||B - A||
    """
    if point_p is None or line_a is None or line_b is None:
        return None

    p = np.asarray(point_p, dtype=np.float64)
    a = np.asarray(line_a, dtype=np.float64)
    b = np.asarray(line_b, dtype=np.float64)

    ab = b - a
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-6:
        return 0.0

    cross_prod = np.cross(p - a, p - b)
    distance = np.linalg.norm(cross_prod) / ab_norm
    return float(distance)
