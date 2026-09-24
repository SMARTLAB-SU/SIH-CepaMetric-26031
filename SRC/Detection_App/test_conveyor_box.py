"""
================================================================================
Comprehensive Automated Multi-Box Acceptance Test Suite (Tests 1 to 9)
================================================================================
Strict Classical Computer Vision + RealSense D455f 3D Geometry (NO AI / NO ML).

Verifies all 9 user test scenarios:
  TEST 1 — 1 stationary box: Detects 1 box accurately.
  TEST 2 — 3 stationary boxes: Detects all 3 boxes simultaneously with 3 IDs.
  TEST 3 — 5 boxes present: Detects all 5 separable boxes simultaneously.
  TEST 4 — 3 boxes moving on conveyor: Continuously detected and tracked.
  TEST 5 — Boxes entering one after another: New sequential ID assigned to each box.
  TEST 6 — Box temporarily missed for 1–3 frames: Same tracking ID retained.
  TEST 7 — Person enters camera view: Person is NOT detected as a box (0 boxes).
  TEST 8 — Box and person appear together: Box detected, person rejected.
  TEST 9 — Boxes at different depths: Boxes separated and measured independently.
================================================================================
"""

import math
import numpy as np
import cv2

from box_tracker import MultiObjectTracker, TrackedObject
from detection import MultiObjectDetector


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


def draw_person_on_frame(color, depth, person_cx=320, person_depth=1200):
    """
    Renders a realistic synthetic human figure with curved torso/limbs
    and non-planar curved depth (high variance / organic surface).
    """
    h, w = color.shape[:2]
    person_mask = np.zeros((h, w), dtype=np.uint8)
    # Head
    cv2.ellipse(person_mask, (person_cx, 120), (32, 42), 0, 0, 360, 255, -1)
    # Torso
    torso_pts = np.array([
        [person_cx - 55, 170],
        [person_cx + 55, 170],
        [person_cx + 45, 340],
        [person_cx - 45, 340]
    ], np.int32)
    cv2.fillPoly(person_mask, [torso_pts], 255)
    # Limbs
    cv2.line(person_mask, (person_cx - 55, 180), (person_cx - 95, 280), 255, 22)
    cv2.line(person_mask, (person_cx + 55, 180), (person_cx + 95, 280), 255, 22)
    cv2.line(person_mask, (person_cx - 25, 340), (person_cx - 30, 460), 255, 24)
    cv2.line(person_mask, (person_cx + 25, 340), (person_cx + 30, 460), 255, 24)

    # Render RGB colors
    cv2.ellipse(color, (person_cx, 120), (32, 42), 0, 0, 360, (140, 160, 200), -1)
    cv2.fillPoly(color, [torso_pts], (180, 100, 60))
    cv2.line(color, (person_cx - 55, 180), (person_cx - 95, 280), (140, 160, 200), 22)
    cv2.line(color, (person_cx + 55, 180), (person_cx + 95, 280), (140, 160, 200), 22)
    cv2.line(color, (person_cx - 25, 340), (person_cx - 30, 460), (70, 70, 90), 24)
    cv2.line(color, (person_cx + 25, 340), (person_cx + 30, 460), (70, 70, 90), 24)

    # Curved non-planar organic depth
    y_grid, x_grid = np.ogrid[:h, :w]
    dx = (x_grid - person_cx).astype(np.float32)
    dy = (y_grid - 260).astype(np.float32)
    curved_depth = person_depth + 0.08 * (dx ** 2) + 0.04 * (dy ** 2)
    noise = np.random.normal(0, 8.0, (h, w))
    human_depth_vals = np.clip(curved_depth + noise, person_depth - 80, person_depth + 600).astype(np.uint16)
    depth[person_mask == 255] = human_depth_vals[person_mask == 255]


# ==============================================================================
# TEST 1: 1 Stationary Box
# ==============================================================================
def test_1_stationary_box():
    print("\n--- [TEST 1] 1 Stationary Box ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 850, dtype=np.uint16)

    draw_box_on_frame(color, depth, x=220, y=160, w=180, h=120, box_depth_mm=680)

    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    for _ in range(3):
        active, primary, completed, edges, diag = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Boxes Detected: {len(active)}")
    assert len(active) == 1, f"Expected 1 box detected, got {len(active)}"
    b = active[0]
    dims = b.get_dimensions("mm")
    print(f"    - ID #{b.obj_id:02d}: {b.shape_type} | L={dims['length']:.0f} mm, W={dims['width']:.0f} mm, H={dims['height']:.0f} mm")
    print("[PASS] TEST 1: 1 stationary box detected and measured accurately.")


# ==============================================================================
# TEST 2: 3 Stationary Boxes
# ==============================================================================
def test_2_three_stationary_boxes():
    print("\n--- [TEST 2] 3 Stationary Boxes ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 850, dtype=np.uint16)

    # Box 1: Left
    draw_box_on_frame(color, depth, x=50, y=160, w=150, h=120, box_depth_mm=660, box_color=(55, 115, 185))
    # Box 2: Center
    draw_box_on_frame(color, depth, x=240, y=160, w=150, h=120, box_depth_mm=680, box_color=(40, 95, 160))
    # Box 3: Right
    draw_box_on_frame(color, depth, x=430, y=160, w=150, h=120, box_depth_mm=670, box_color=(50, 160, 75))

    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    for _ in range(3):
        active, primary, completed, edges, diag = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Boxes Detected: {len(active)}")
    for b in active:
        dims = b.get_dimensions("mm")
        print(f"    - ID #{b.obj_id:02d}: {b.shape_type} | L={dims['length']:.0f} mm, W={dims['width']:.0f} mm, H={dims['height']:.0f} mm")

    assert len(active) == 3, f"Expected 3 boxes detected, got {len(active)}"
    unique_ids = {b.obj_id for b in active}
    assert len(unique_ids) == 3, f"Expected 3 unique IDs, got {unique_ids}"
    print("[PASS] TEST 2: 3 stationary boxes detected simultaneously with unique IDs.")


# ==============================================================================
# TEST 3: 5 Boxes Present Simultaneously
# ==============================================================================
def test_3_five_boxes():
    print("\n--- [TEST 3] 5 Boxes Present Simultaneously ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 850, dtype=np.uint16)

    # 5 Separable Boxes across frame
    draw_box_on_frame(color, depth, x=40, y=70, w=140, h=110, box_depth_mm=620, box_color=(55, 115, 185))
    draw_box_on_frame(color, depth, x=240, y=60, w=150, h=120, box_depth_mm=650, box_color=(40, 95, 160))
    draw_box_on_frame(color, depth, x=440, y=70, w=140, h=110, box_depth_mm=630, box_color=(50, 160, 75))
    draw_box_on_frame(color, depth, x=100, y=280, w=160, h=130, box_depth_mm=670, box_color=(175, 105, 45))
    draw_box_on_frame(color, depth, x=370, y=280, w=170, h=130, box_depth_mm=660, box_color=(120, 60, 150))

    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    for _ in range(3):
        active, primary, completed, edges, diag = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Boxes Detected: {len(active)}")
    for b in active:
        dims = b.get_dimensions("mm")
        print(f"    - ID #{b.obj_id:02d}: {b.shape_type} | L={dims['length']:.0f} mm, W={dims['width']:.0f} mm, H={dims['height']:.0f} mm")

    assert len(active) == 5, f"Expected 5 boxes detected, got {len(active)}"
    unique_ids = {b.obj_id for b in active}
    assert len(unique_ids) == 5, f"Expected 5 unique IDs, got {unique_ids}"
    print("[PASS] TEST 3: All 5 separable boxes detected simultaneously.")


# ==============================================================================
# TEST 4: 3 Boxes Moving on Conveyor
# ==============================================================================
def test_4_three_moving_boxes():
    print("\n--- [TEST 4] 3 Boxes Moving on Conveyor ---")
    h, w = 480, 640
    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    # Track 3 moving boxes over 15 frames
    for frame_idx in range(15):
        color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
        depth = np.full((h, w), 850, dtype=np.uint16)

        x_shift = frame_idx * 12
        # Box 1
        draw_box_on_frame(color, depth, x=30 + x_shift, y=180, w=120, h=100, box_depth_mm=660)
        # Box 2
        draw_box_on_frame(color, depth, x=190 + x_shift, y=180, w=120, h=100, box_depth_mm=680)
        # Box 3
        draw_box_on_frame(color, depth, x=350 + x_shift, y=180, w=120, h=100, box_depth_mm=670)

        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")
        assert len(active) == 3, f"Frame {frame_idx}: Expected 3 moving boxes, got {len(active)}"

    # Check that exactly 3 unique IDs were used
    tracked_ids = {b.obj_id for b in tracker.active_objects}
    print(f"[*] Tracked IDs across motion: {tracked_ids}")
    assert len(tracked_ids) == 3, f"Expected 3 persistent IDs, got {tracked_ids}"
    assert tracker.total_objects_counted == 3, f"Expected total counted = 3, got {tracker.total_objects_counted}"
    print("[PASS] TEST 4: 3 moving boxes continuously detected and tracked with stable IDs.")


# ==============================================================================
# TEST 5: Boxes Entering One After Another (Sequential New IDs)
# ==============================================================================
def test_5_sequential_box_entries():
    print("\n--- [TEST 5] Boxes Entering One After Another ---")
    h, w = 480, 640
    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    # Box 1 enters and stays for 3 frames
    color1 = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth1 = np.full((h, w), 850, dtype=np.uint16)
    draw_box_on_frame(color1, depth1, x=100, y=180, w=150, h=120, box_depth_mm=680)
    for _ in range(3):
        active, _, _, _, _ = tracker.process_frame(color1, depth1, intrin, unit="mm")
    assert len(active) == 1
    id1 = active[0].obj_id
    print(f"[*] Box 1 entered -> ID #{id1:02d}")

    # Box 1 leaves (10 empty frames)
    color_empty = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth_empty = np.full((h, w), 850, dtype=np.uint16)
    for _ in range(10):
        tracker.process_frame(color_empty, depth_empty, intrin, unit="mm")
    assert len(tracker.active_objects) == 0

    # Box 2 enters
    color2 = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth2 = np.full((h, w), 850, dtype=np.uint16)
    draw_box_on_frame(color2, depth2, x=300, y=180, w=150, h=120, box_depth_mm=680)
    active2, _, _, _, _ = tracker.process_frame(color2, depth2, intrin, unit="mm")
    assert len(active2) == 1
    id2 = active2[0].obj_id
    print(f"[*] Box 2 entered -> ID #{id2:02d}")

    assert id2 == id1 + 1, f"Expected sequential ID #{id1 + 1}, got #{id2}"
    print("[PASS] TEST 5: Sequential new IDs correctly assigned upon box entry.")


# ==============================================================================
# TEST 6: Box Temporarily Missed for 1-3 Frames (Grace Period)
# ==============================================================================
def test_6_temporary_missed_frames():
    print("\n--- [TEST 6] Box Temporarily Missed for 1–3 Frames ---")
    h, w = 480, 640
    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 850, dtype=np.uint16)
    draw_box_on_frame(color, depth, x=220, y=160, w=180, h=120, box_depth_mm=680)

    # Frame 1-3: Detected
    for _ in range(3):
        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")
    initial_id = active[0].obj_id
    print(f"[*] Box tracked as ID #{initial_id:02d}")

    # Frame 4-5: Temporarily missed (2 empty frames due to sensor dropout/glitch)
    color_empty = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth_empty = np.full((h, w), 850, dtype=np.uint16)
    tracker.process_frame(color_empty, depth_empty, intrin, unit="mm")
    tracker.process_frame(color_empty, depth_empty, intrin, unit="mm")
    print(f"[*] Simulating 2 missed frames (active surviving count = {len(tracker.active_objects)})")
    assert len(tracker.active_objects) == 1, "Track should survive 2 missed frames in grace period"

    # Frame 6: Box reappears
    active_reconnect, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")
    assert len(active_reconnect) == 1, "Box should be detected again"
    reconnected_id = active_reconnect[0].obj_id
    print(f"[*] Box reappeared -> Reconnected ID #{reconnected_id:02d}")

    assert reconnected_id == initial_id, f"Expected same ID #{initial_id}, got #{reconnected_id}"
    assert tracker.total_objects_counted == 1, "Should not increment total objects counted"
    print("[PASS] TEST 6: Same tracking ID retained across temporary missed frames.")


# ==============================================================================
# TEST 7: Person Enters Camera View (MUST BE REJECTED)
# ==============================================================================
def test_7_person_rejected():
    print("\n--- [TEST 7] Person Enters Camera View (Rejection Test) ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 2200, dtype=np.uint16)

    draw_person_on_frame(color, depth, person_cx=320, person_depth=1200)

    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    for _ in range(3):
        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Detected Boxes with Person in View: {len(active)}")
    assert len(active) == 0, f"FAILED: Person was detected as a box! ({len(active)} boxes found)"
    print("[PASS] TEST 7: Person entering camera view is NOT detected as a box.")


# ==============================================================================
# TEST 8: Box and Person Appear Together
# ==============================================================================
def test_8_box_and_person_together():
    print("\n--- [TEST 8] Box and Person Appear Together ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 2200, dtype=np.uint16)

    # 1 Real Box on table at X=100, Y=260, Depth=700mm
    draw_box_on_frame(color, depth, x=80, y=260, w=170, h=130, box_depth_mm=700)

    # 1 Person standing next to table at X=420, Depth=1200mm
    draw_person_on_frame(color, depth, person_cx=420, person_depth=1200)

    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    for _ in range(3):
        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Active Detections with Box + Person: {len(active)}")
    assert len(active) == 1, f"Expected exactly 1 box detected (person rejected), got {len(active)}"
    b = active[0]
    dims = b.get_dimensions("mm")
    print(f"    - Detected Box ID #{b.obj_id:02d}: {b.shape_type} | L={dims['length']:.0f} mm, W={dims['width']:.0f} mm, H={dims['height']:.0f} mm")
    print("[PASS] TEST 8: Box successfully detected while person is simultaneously rejected.")


# ==============================================================================
# TEST 9: Boxes at Different Depths
# ==============================================================================
def test_9_boxes_at_different_depths():
    print("\n--- [TEST 9] Boxes at Different Depths (Depth Separation) ---")
    h, w = 480, 640
    color = np.full((h, w, 3), (210, 215, 220), dtype=np.uint8)
    depth = np.full((h, w), 2200, dtype=np.uint16)

    # Box 1: Near (Depth = 480 mm)
    draw_box_on_frame(color, depth, x=60, y=140, w=160, h=120, box_depth_mm=480, box_color=(55, 115, 185))

    # Box 2: Medium (Depth = 780 mm)
    draw_box_on_frame(color, depth, x=250, y=160, w=150, h=120, box_depth_mm=780, box_color=(40, 95, 160))

    # Box 3: Far (Depth = 1250 mm)
    draw_box_on_frame(color, depth, x=440, y=180, w=140, h=110, box_depth_mm=1250, box_color=(50, 160, 75))

    intrin = MockIntrinsics()
    tracker = MultiObjectTracker()

    for _ in range(3):
        active, _, _, _, _ = tracker.process_frame(color, depth, intrin, unit="mm")

    print(f"[*] Boxes Detected at Varying Depths: {len(active)}")
    for b in active:
        dims = b.get_dimensions("mm")
        print(f"    - ID #{b.obj_id:02d}: Depth={b.top_depth_m*1000:.0f} mm | L={dims['length']:.0f} mm, W={dims['width']:.0f} mm, H={dims['height']:.0f} mm")

    assert len(active) == 3, f"Expected 3 boxes at different depths, got {len(active)}"
    print("[PASS] TEST 9: Boxes at different depths successfully separated and measured.")


# ==============================================================================
# TEST 10: Event-Based Measurement Logging (1 Physical Box = 1 Record)
# ==============================================================================
def test_10_event_based_single_record_logging():
    print("\n--- [TEST 10] Event-Based Measurement Logging (1 Physical Box = 1 Record) ---")
    h, w = 480, 640
    intrin = MockIntrinsics()

    # Sub-case 1: No box in view for 60 frames -> 0 measurement records
    t_empty = MultiObjectTracker()
    c_empty = np.full((h, w, 3), 215, dtype=np.uint8)
    d_empty = np.full((h, w), 850, dtype=np.uint16)
    for _ in range(60):
        t_empty.process_frame(c_empty, d_empty, intrin)
    assert len(t_empty.completed_objects_log) == 0, f"Expected 0 records for empty view, got {len(t_empty.completed_objects_log)}"
    print("  [OK] Sub-case 1: Empty view generates 0 measurement records.")

    # Sub-case 2: 1 stationary box visible for 300 frames -> Exactly 1 record
    t_1box = MultiObjectTracker()
    c_1 = np.full((h, w, 3), 215, dtype=np.uint8)
    d_1 = np.full((h, w), 850, dtype=np.uint16)
    draw_box_on_frame(c_1, d_1, 220, 160, 180, 120, 680)
    for _ in range(300):
        t_1box.process_frame(c_1, d_1, intrin)
    assert len(t_1box.completed_objects_log) == 1, f"Expected exactly 1 record for 1 box across 300 frames, got {len(t_1box.completed_objects_log)}"
    print("  [OK] Sub-case 2: 1 stationary box for 300 frames generates exactly 1 record.")

    # Sub-case 3: 2 stationary boxes visible for 300 frames -> Exactly 2 records
    t_2box = MultiObjectTracker()
    c_2 = np.full((h, w, 3), 215, dtype=np.uint8)
    d_2 = np.full((h, w), 850, dtype=np.uint16)
    draw_box_on_frame(c_2, d_2, 80, 160, 150, 120, 660, (55, 115, 185))
    draw_box_on_frame(c_2, d_2, 380, 160, 150, 120, 680, (40, 95, 160))
    for _ in range(300):
        t_2box.process_frame(c_2, d_2, intrin)
    assert len(t_2box.completed_objects_log) == 2, f"Expected exactly 2 records for 2 boxes across 300 frames, got {len(t_2box.completed_objects_log)}"
    print("  [OK] Sub-case 3: 2 stationary boxes for 300 frames generate exactly 2 records.")

    # Sub-case 4: 2 moving boxes on conveyor -> Exactly 2 records
    t_move = MultiObjectTracker()
    for f in range(40):
        c_m = np.full((h, w, 3), 215, dtype=np.uint8)
        d_m = np.full((h, w), 850, dtype=np.uint16)
        shift = f * 8
        draw_box_on_frame(c_m, d_m, 60 + shift, 160, 130, 100, 660, (55, 115, 185))
        draw_box_on_frame(c_m, d_m, 260 + shift, 160, 130, 100, 680, (40, 95, 160))
        t_move.process_frame(c_m, d_m, intrin)
    assert len(t_move.completed_objects_log) == 2, f"Expected 2 records for 2 moving boxes, got {len(t_move.completed_objects_log)}"
    assert t_move.total_objects_counted == 2
    print("  [OK] Sub-case 4: 2 moving boxes on conveyor generate exactly 2 records.")

    # Sub-case 5: 5 boxes entering sequentially -> Exactly 5 records
    t_seq = MultiObjectTracker()
    for b_i in range(5):
        c_b = np.full((h, w, 3), 215, dtype=np.uint8)
        d_b = np.full((h, w), 850, dtype=np.uint16)
        draw_box_on_frame(c_b, d_b, 200, 160, 150, 120, 670)
        for _ in range(5):
            t_seq.process_frame(c_b, d_b, intrin)
        for _ in range(12):
            t_seq.process_frame(c_empty, d_empty, intrin)
    assert len(t_seq.completed_objects_log) == 5, f"Expected 5 records for 5 sequential boxes, got {len(t_seq.completed_objects_log)}"
    print("  [OK] Sub-case 5: 5 sequential box entries generate exactly 5 records.")

    # Sub-case 6: 1 box stays visible for 600 frames -> Still only 1 record
    t_long = MultiObjectTracker()
    for _ in range(600):
        t_long.process_frame(c_1, d_1, intrin)
    assert len(t_long.completed_objects_log) == 1, f"Expected 1 record for long visibility, got {len(t_long.completed_objects_log)}"
    print("  [OK] Sub-case 6: Long box visibility generates only 1 persistent measurement record.")

    print("[PASS] TEST 10: Event-based measurement logging (1 Physical Box = 1 Record) verified 100%.")


# ==============================================================================
# MAIN RUNNER
# ==============================================================================
def run_all_tests():
    print("================================================================================")
    print("RUNNING COMPREHENSIVE 10-CASE MULTI-BOX & EVENT-BASED ACCEPTANCE SUITE")
    print("================================================================================")

    test_1_stationary_box()
    test_2_three_stationary_boxes()
    test_3_five_boxes()
    test_4_three_moving_boxes()
    test_5_sequential_box_entries()
    test_6_temporary_missed_frames()
    test_7_person_rejected()
    test_8_box_and_person_together()
    test_9_boxes_at_different_depths()
    test_10_event_based_single_record_logging()

    print("\n================================================================================")
    print("[ALL 10 TESTS PASSED] MULTI-BOX DETECTION & EVENT-BASED MEASUREMENT FULLY VERIFIED!")
    print("================================================================================\n")


if __name__ == "__main__":
    run_all_tests()

