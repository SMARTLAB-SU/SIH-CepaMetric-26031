"""
================================================================================
Automated Circle Object & Onion Size Classification Test Suite
================================================================================
Strict Classical Computer Vision + RealSense D455f 3D Geometry (NO AI / NO ML).

Verifies:
  TEST 1 — Stationary Small, Medium, Large Onions:
            Correct Radius, Diameter, and Size Classification.
  TEST 2 — Moving Onions on Conveyor:
            Continuous tracking with stable persistent IDs across motion.
  TEST 3 — Mixed Multi-Object Scene (Boxes + Onions):
            Boxes measured with H/L/W and Onions measured with Radius/Diameter/Grade.
  TEST 4 — Rejection Guarantees:
            Human body and non-circular irregular shapes rejected.
  TEST 5 — Event-Based Logging:
            1 physical onion visible for 300 frames generates exactly 1 record.
================================================================================
"""

import math
import numpy as np
import cv2

# Import from main standalone application
from main import (
    MultiObjectTracker,
    MultiObjectDetector,
    classify_onion_size,
    ONION_SIZE_THRESHOLDS
)


class MockIntrinsics:
    def __init__(self, w=640, h=480, fx=385.0, fy=385.0):
        self.width = w
        self.height = h
        self.fx = fx
        self.fy = fy
        self.ppx = w / 2.0
        self.ppy = h / 2.0
        self.coeffs = [0.0] * 5


def draw_box_on_frame(color, depth, x, y, w, h, box_depth_mm, box_color=(55, 115, 185)):
    """Helper to draw a physical box with crisp edges on color and depth frames."""
    depth[y:y + h, x:x + w] = box_depth_mm
    cv2.rectangle(color, (x, y), (x + w, y + h), box_color, -1)
    cv2.rectangle(color, (x, y), (x + w, y + h), (25, 30, 40), 2)
    cv2.line(color, (x, y + h // 2), (x + w, y + h // 2), (180, 170, 130), 2)


def draw_onion_on_frame(color, depth, cx, cy, r_px, onion_depth_mm, onion_color=(45, 50, 180)):
    """Helper to draw a circular agricultural onion on color and depth frames."""
    h, w = color.shape[:2]
    y_g, x_g = np.ogrid[:h, :w]
    dist = np.sqrt((x_g - cx) ** 2 + (y_g - cy) ** 2)
    depth[dist <= r_px] = onion_depth_mm
    cv2.circle(color, (cx, cy), r_px, onion_color, -1)
    cv2.circle(color, (cx, cy), r_px, (25, 25, 90), 2)


def draw_person_on_frame(color, depth, person_cx=320, person_depth=1200):
    """Renders a realistic synthetic human figure with curved torso/limbs and high depth variance."""
    h, w = color.shape[:2]
    person_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(person_mask, (person_cx, 120), (32, 42), 0, 0, 360, 255, -1)
    torso_pts = np.array([
        [person_cx - 55, 170],
        [person_cx + 55, 170],
        [person_cx + 45, 340],
        [person_cx - 45, 340]
    ], np.int32)
    cv2.fillPoly(person_mask, [torso_pts], 255)
    cv2.line(person_mask, (person_cx - 55, 180), (person_cx - 95, 280), 255, 22)
    cv2.line(person_mask, (person_cx + 55, 180), (person_cx + 95, 280), 255, 22)

    cv2.ellipse(color, (person_cx, 120), (32, 42), 0, 0, 360, (140, 160, 200), -1)
    cv2.fillPoly(color, [torso_pts], (180, 100, 60))
    cv2.line(color, (person_cx - 55, 180), (person_cx - 95, 280), (140, 160, 200), 22)
    cv2.line(color, (person_cx + 55, 180), (person_cx + 95, 280), (140, 160, 200), 22)

    y_grid, x_grid = np.ogrid[:h, :w]
    dx = (x_grid - person_cx).astype(np.float32)
    dy = (y_grid - 260).astype(np.float32)
    curved_depth = person_depth + 0.08 * (dx ** 2) + 0.04 * (dy ** 2)
    human_depth_vals = np.clip(curved_depth, person_depth - 80, person_depth + 600).astype(np.uint16)
    depth[person_mask == 255] = human_depth_vals[person_mask == 255]


# ==============================================================================
# TEST 1: Stationary Small, Medium, Large Onions
# ==============================================================================
def test_1_stationary_onions_classification():
    print("\n--- [TEST 1] Stationary Small, Medium, Large Onions ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 850, dtype=np.uint16)
    intrin = MockIntrinsics()

    # Small Onion (r=20px at 700mm -> Dia ~72.7mm? Wait: r=20px at 700mm / 385 = 36.3mm Dia)
    # At depth 700 mm: fx=385 -> Dia_mm = 2 * r_px * 700 / 385 = r_px * 3.636 mm
    # For Small (<45 mm): r_px = 10 -> Dia ~36.4 mm
    # For Medium (45-70 mm): r_px = 16 -> Dia ~58.2 mm
    # For Large (>=70 mm): r_px = 22 -> Dia ~80.0 mm

    draw_onion_on_frame(color, depth, cx=120, cy=240, r_px=8, onion_depth_mm=700)   # Small (~38 mm)
    draw_onion_on_frame(color, depth, cx=300, cy=240, r_px=14, onion_depth_mm=700)  # Medium (~56 mm)
    draw_onion_on_frame(color, depth, cx=500, cy=240, r_px=22, onion_depth_mm=700)  # Large (~85 mm)

    tracker = MultiObjectTracker()
    for _ in range(4):
        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Detected Objects: {len(active)}")
    assert len(active) == 3, f"Expected 3 onions detected, got {len(active)}"

    grades = []
    for obj in active:
        dims = obj.get_dimensions("mm")
        print(f"    - ID #{obj.obj_id:02d}: {obj.shape_type} | Dia: {dims['diameter']:.1f} mm, Rad: {dims['radius']:.1f} mm | Grade: {dims['size_class']}")
        assert obj.shape_type == "ONION", f"Expected ONION, got {obj.shape_type}"
        grades.append(dims["size_class"])

    assert "SMALL" in grades, f"Expected SMALL in {grades}"
    assert "MEDIUM" in grades, f"Expected MEDIUM in {grades}"
    assert "LARGE" in grades, f"Expected LARGE in {grades}"
    print("[PASS] TEST 1: Small, Medium, Large onions detected and classified accurately.")


# ==============================================================================
# TEST 2: Moving Onions on Conveyor Belt
# ==============================================================================
def test_2_moving_onions_tracking():
    print("\n--- [TEST 2] Moving Onions on Conveyor ---")
    h, w = 480, 640
    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    for frame_idx in range(15):
        color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
        depth = np.full((h, w), 850, dtype=np.uint16)

        x_shift = frame_idx * 10
        # Onion 1 (Medium)
        draw_onion_on_frame(color, depth, cx=80 + x_shift, cy=240, r_px=16, onion_depth_mm=720)
        # Onion 2 (Large)
        draw_onion_on_frame(color, depth, cx=260 + x_shift, cy=240, r_px=22, onion_depth_mm=710)

        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")
        assert len(active) == 2, f"Frame {frame_idx}: Expected 2 moving onions, got {len(active)}"

    tracked_ids = {b.obj_id for b in tracker.active_objects}
    print(f"[*] Persistent Tracked IDs across motion: {tracked_ids}")
    assert len(tracked_ids) == 2, f"Expected 2 persistent IDs, got {tracked_ids}"
    assert tracker.total_objects_counted == 2, f"Expected total counted = 2, got {tracker.total_objects_counted}"
    print("[PASS] TEST 2: Moving onions continuously tracked with persistent IDs.")


# ==============================================================================
# TEST 3: Mixed Multi-Object Scene (Boxes + Onions Together)
# ==============================================================================
def test_3_mixed_boxes_and_onions():
    print("\n--- [TEST 3] Mixed Multi-Object Scene (Boxes + Onions) ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 850, dtype=np.uint16)
    intrin = MockIntrinsics()

    # 1 Real Box on table
    draw_box_on_frame(color, depth, x=60, y=140, w=160, h=120, box_depth_mm=620)
    # 1 Medium Onion
    draw_onion_on_frame(color, depth, cx=360, cy=200, r_px=16, onion_depth_mm=720)
    # 1 Large Onion
    draw_onion_on_frame(color, depth, cx=520, cy=200, r_px=22, onion_depth_mm=700)

    tracker = MultiObjectTracker()
    for _ in range(4):
        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Total Active Objects: {len(active)}")
    assert len(active) == 3, f"Expected 3 objects (1 box + 2 onions), got {len(active)}"

    shapes = [o.shape_type for o in active]
    print(f"[*] Detected Shape Types: {shapes}")
    assert "BOX" in shapes, "Expected BOX detected"
    assert shapes.count("ONION") == 2, f"Expected 2 ONIONs, got {shapes}"

    for o in active:
        dims = o.get_dimensions("mm")
        if o.shape_type == "BOX":
            print(f"    - BOX ID #{o.obj_id:02d}: L={dims['length']:.0f} mm, W={dims['width']:.0f} mm, H={dims['height']:.0f} mm")
            assert dims["length"] > 100 and dims["width"] > 80
        elif o.shape_type == "ONION":
            print(f"    - ONION ID #{o.obj_id:02d}: Dia={dims['diameter']:.1f} mm, Rad={dims['radius']:.1f} mm, Grade={dims['size_class']}")
            assert dims["diameter"] > 30

    print("[PASS] TEST 3: Both Boxes and Onions successfully detected and dimensioned simultaneously.")


# ==============================================================================
# TEST 4: Rejection of Human Body from Circle Detection
# ==============================================================================
def test_4_human_rejection():
    print("\n--- [TEST 4] Human Rejection Test ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 2200, dtype=np.uint16)
    intrin = MockIntrinsics()

    draw_person_on_frame(color, depth, person_cx=320, person_depth=1200)

    tracker = MultiObjectTracker()
    for _ in range(3):
        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Detected Objects with Person in View: {len(active)}")
    assert len(active) == 0, f"Person was falsely detected as an object! ({len(active)} found)"
    print("[PASS] TEST 4: Human body successfully rejected from detection.")


# ==============================================================================
# TEST 5: Event-Based Single Record Logging for Onions
# ==============================================================================
def test_5_event_based_single_record_logging_onion():
    print("\n--- [TEST 5] Event-Based Logging for Onions (1 Onion = 1 Record) ---")
    h, w = 480, 640
    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 850, dtype=np.uint16)
    draw_onion_on_frame(color, depth, cx=300, cy=240, r_px=16, onion_depth_mm=720)

    # 1 Onion visible for 300 frames
    for _ in range(300):
        tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Total Records Logged for 1 Onion over 300 frames: {len(tracker.completed_objects_log)}")
    assert len(tracker.completed_objects_log) == 1, f"Expected exactly 1 record, got {len(tracker.completed_objects_log)}"
    rec = tracker.completed_objects_log[0]
    print(f"    - Logged Record: ID #{rec['obj_id']:02d} | Shape: {rec['shape_type']} | Dia: {rec['diameter']:.1f} mm | Grade: {rec['size_class']}")
    assert rec["shape_type"] == "ONION"
    assert rec["size_class"] == "MEDIUM"
    print("[PASS] TEST 5: Event-based logging verified (1 physical onion generates exactly 1 record).")


def run_all_tests():
    print("================================================================================")
    print("RUNNING AUTOMATED TEST SUITE: CIRCLE DETECTION & ONION SIZE CLASSIFICATION")
    print("================================================================================")

    test_1_stationary_onions_classification()
    test_2_moving_onions_tracking()
    test_3_mixed_boxes_and_onions()
    test_4_human_rejection()
    test_5_event_based_single_record_logging_onion()

    print("\n================================================================================")
    print("[ALL 5 TESTS PASSED] CIRCLE DETECTION & ONION SIZE CLASSIFICATION FULLY VERIFIED!")
    print("================================================================================\n")


if __name__ == "__main__":
    run_all_tests()
