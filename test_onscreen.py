#!/usr/bin/env python3
"""
Test on-screen calibration instructions
"""
import cv2
import numpy as np
import sys
sys.path.append('.')

from real_kinect_sandbox import display_width, display_height

def test_onscreen_instructions():
    print("🧪 Testing On-Screen Calibration Instructions")
    print("=" * 60)
    
    # Test 1: Sandbox calibration screen
    print("1. Testing sandbox calibration instructions...")
    test_screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # Simulate corner selection step
    current_step = 2  # Testing third corner
    corner_labels = ["TL", "TR", "BR", "BL"]
    corner_instructions = [
        "Click TOP-LEFT corner",
        "Click TOP-RIGHT corner", 
        "Click BOTTOM-RIGHT corner",
        "Click BOTTOM-LEFT corner"
    ]
    
    # Main instruction
    cv2.putText(test_screen, corner_instructions[current_step], 
               (display_width // 2 - 150, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Progress indicator
    cv2.putText(test_screen, f"Corner {current_step + 1} of 4", 
               (display_width // 2 - 80, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Corner status list
    for i in range(4):
        if i < current_step:
            # Completed corner
            cv2.putText(test_screen, f"✓ {corner_labels[i]}", 
                       (30, 150 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif i == current_step:
            # Current corner
            cv2.putText(test_screen, f"→ {corner_labels[i]}", 
                       (30, 150 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Pending corner
            cv2.putText(test_screen, f"  {corner_labels[i]}", 
                       (30, 150 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Control instructions
    cv2.putText(test_screen, "Click to select | C: Clear | ESC: Cancel | ENTER: Confirm", 
               (10, display_height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Sandbox Calibration Test', test_screen)
    print("Showing sandbox calibration test - Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Test 2: Projection alignment screen
    print("2. Testing projection alignment instructions...")
    align_screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # Main instruction
    cv2.putText(align_screen, "PROJECTION ALIGNMENT", 
               (display_width // 2 - 150, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    # Current corner indicator
    selected_corner = 1  # Testing second corner
    cv2.putText(align_screen, f"Adjusting: {corner_labels[selected_corner]} corner", 
               (display_width // 2 - 120, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Corner status
    for i in range(4):
        if i == selected_corner:
            # Current corner
            cv2.putText(align_screen, f"→ {corner_labels[i]}", 
                       (30, 140 + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Other corners
            cv2.putText(align_screen, f"  {corner_labels[i]}", 
                       (30, 140 + i * 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    # Control instructions
    cv2.putText(align_screen, "Arrow Keys: Move corner | TAB: Next corner | ENTER: Confirm | ESC: Cancel", 
               (10, display_height - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Projection Alignment Test', align_screen)
    print("Showing projection alignment test - Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Test 3: Depth calibration screen
    print("3. Testing depth calibration instructions...")
    depth_screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    step_idx = 1  # Testing second step
    boundary_name = "Brown → Green"
    description = "close to middle boundary"
    current_depth = 850
    
    # Step indicator
    cv2.putText(depth_screen, f"DEPTH CALIBRATION - Step {step_idx + 1} of 3", 
               (display_width // 2 - 180, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    # Current boundary
    cv2.putText(depth_screen, boundary_name, 
               (display_width // 2 - 100, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(depth_screen, f"Place surface at {description}", 
               (display_width // 2 - 150, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Current values
    cv2.putText(depth_screen, f"Raw depth: {current_depth}", (10, display_height - 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(depth_screen, f"Threshold: 800", (10, display_height - 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Status
    status_text = "VALID"
    status_color = (0, 255, 0)
    cv2.putText(depth_screen, f"Status: {status_text}", (10, display_height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Controls
    cv2.putText(depth_screen, "C: Capture | Arrows: Adjust | Q: Quit", 
               (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Depth Calibration Test', depth_screen)
    print("Showing depth calibration test - Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ All on-screen instruction tests complete!")

if __name__ == "__main__":
    test_onscreen_instructions()