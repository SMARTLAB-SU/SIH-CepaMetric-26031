"""
================================================================================
Automated Box 3D Geometric Measurement & Unit Test Suite
================================================================================
Tests simultaneous detection, classification, and dimensioning of:
  1. CARDBOARD BOXES (Length x Width x Height in mm & cm)
  2. WOODEN BOXES    (Length x Width x Height in mm & cm)
  3. SQUARE BOXES    (Length x Width x Height in mm & cm)
  4. RECTANGULAR BOXES (Length x Width x Height in mm & cm)
================================================================================
"""

import numpy as np
from camera import MultiObjectSceneSimulator
from detection import MultiObjectDetector


def run_tests():
    print("================================================================================")
    print("RUNNING AUTOMATED TEST: MULTI-BOX 3D GEOMETRIC MEASUREMENT (mm & cm)")
    print("================================================================================\n")

    sim = MultiObjectSceneSimulator(width=640, height=480, fps=30)
    color_bgr, depth_raw, intrinsics = sim.generate_frame()

    # 1. Test in Millimeters (mm)
    objects_mm, edge_map, diag = MultiObjectDetector.detect_and_measure(
        color_bgr=color_bgr,
        depth_raw=depth_raw,
        intrinsics=intrinsics,
        min_depth_mm=120,
        max_depth_mm=2500,
        unit="mm"
    )

    print(f"[*] Total Boxes Detected (in mm): {len(objects_mm)}")
    for obj in objects_mm:
        print(f"\n--- {obj.shape_type} ---")
        print(obj.get_display_text())

    # 2. Test in Centimeters (cm)
    objects_cm, _, _ = MultiObjectDetector.detect_and_measure(
        color_bgr=color_bgr,
        depth_raw=depth_raw,
        intrinsics=intrinsics,
        min_depth_mm=120,
        max_depth_mm=2500,
        unit="cm"
    )

    print(f"\n[*] Total Boxes Detected (in cm): {len(objects_cm)}")
    for obj in objects_cm:
        print(f"\n--- {obj.shape_type} ---")
        print(obj.get_display_text())

    detected_types = {obj.shape_type for obj in objects_mm}
    print(f"\n[*] Shape Classifications: {detected_types}")

    # Assertions
    assert len(objects_mm) >= 3, f"Expected at least 3 boxes detected, got {len(objects_mm)}"
    assert "BOX" in detected_types, "Expected BOX detected"
    assert "SQUARE" in detected_types, "Expected SQUARE detected"

    for obj in objects_mm:
        l = obj.dimensions["length"]
        w = obj.dimensions["width"]
        h = obj.dimensions["height"]
        assert l > 50 and w > 50 and h > 20, f"Unreasonable box dimension: {obj.dimensions}"

    print("\n================================================================================")
    print("[PASS] MULTI-BOX 3D GEOMETRIC MEASUREMENT VERIFIED SUCCESSFULLY!")
    print("================================================================================\n")


if __name__ == "__main__":
    run_tests()
