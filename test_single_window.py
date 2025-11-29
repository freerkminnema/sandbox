#!/usr/bin/env python3
"""
Test single window calibration
"""
import cv2
import numpy as np
import sys
sys.path.append('.')

from real_kinect_sandbox import (
    display_width, display_height,
    create_alignment_pattern, create_elevation_colors,
    create_simulated_terrain
)

def test_single_window():
    print("🧪 Testing Single Window Calibration")
    print("=" * 50)
    
    # Test 1: Show that all modes use same window name
    print("1. Testing window consistency...")
    
    # Create test pattern
    pattern = create_alignment_pattern(display_width, display_height)
    
    # Show in main window
    cv2.imshow('AR Sandbox - Contour Lines', pattern)
    print("✅ Showing alignment pattern in main window")
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # Test 2: Show sandbox calibration screen
    print("2. Testing sandbox calibration screen...")
    
    # Create test background
    depth_data = create_simulated_terrain(display_width, display_height)
    colored = create_elevation_colors(depth_data)
    
    # Add calibration overlay
    current_step = 2
    corner_labels = ["TL", "TR", "BR", "BL"]
    corner_instructions = [
        "Click TOP-LEFT corner",
        "Click TOP-RIGHT corner", 
        "Click BOTTOM-RIGHT corner",
        "Click BOTTOM-LEFT corner"
    ]
    
    # Main instruction
    cv2.putText(colored, corner_instructions[current_step], 
               (display_width // 2 - 150, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Progress indicator
    cv2.putText(colored, f"Corner {current_step + 1} of 4", 
               (display_width // 2 - 80, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Corner status
    for i in range(4):
        if i < current_step:
            cv2.putText(colored, f"✓ {corner_labels[i]}", 
                       (30, 150 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif i == current_step:
            cv2.putText(colored, f"→ {corner_labels[i]}", 
                       (30, 150 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(colored, f"  {corner_labels[i]}", 
                       (30, 150 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    # Control instructions
    cv2.putText(colored, "Click to select | C: Clear | ESC: Cancel | ENTER: Confirm", 
               (10, display_height - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('AR Sandbox - Contour Lines', colored)
    print("✅ Showing sandbox calibration in main window")
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # Test 3: Show projection alignment screen
    print("3. Testing projection alignment screen...")
    
    # Create test corners
    test_corners = [
        [display_width // 4, display_height // 4],
        [3 * display_width // 4, display_height // 4],
        [3 * display_width // 4, 3 * display_height // 4],
        [display_width // 4, 3 * display_height // 4]
    ]
    
    display = pattern.copy()
    corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    corner_labels = ["TL", "TR", "BR", "BL"]
    selected_corner = 1
    
    # Draw corner markers
    for i, corner in enumerate(test_corners):
        color = corner_colors[i]
        thickness = 3 if i == selected_corner else 2
        radius = 15 if i == selected_corner else 10
        
        cv2.circle(display, tuple(corner), radius, color, thickness)
        cv2.putText(display, corner_labels[i], (corner[0] + 20, corner[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw connecting lines
        next_i = (i + 1) % 4
        cv2.line(display, tuple(corner), tuple(test_corners[next_i]), color, 2)
    
    # On-screen instructions
    cv2.putText(display, "PROJECTION ALIGNMENT", 
               (display_width // 2 - 150, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    cv2.putText(display, f"Adjusting: {corner_labels[selected_corner]} corner", 
               (display_width // 2 - 120, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Control instructions
    cv2.putText(display, "Arrow Keys: Move corner | TAB: Next corner | ENTER: Confirm | ESC: Cancel", 
               (10, display_height - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('AR Sandbox - Contour Lines', display)
    print("✅ Showing projection alignment in main window")
    print("Press any key to finish test...")
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("✅ Single window test complete!")

if __name__ == "__main__":
    test_single_window()