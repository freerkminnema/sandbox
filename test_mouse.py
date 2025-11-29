#!/usr/bin/env python3
"""
Test mouse clicking in calibration
"""
import cv2
import numpy as np
import sys
sys.path.append('.')

from real_kinect_sandbox import display_width, display_height

def test_mouse_clicking():
    print("🧪 Testing Mouse Clicking in Calibration")
    print("=" * 50)
    
    # Create test screen
    test_screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # Add instructions
    cv2.putText(test_screen, "MOUSE CLICK TEST", 
               (display_width // 2 - 120, display_height // 3),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    cv2.putText(test_screen, "Click anywhere to test", 
               (display_width // 2 - 140, display_height // 3 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(test_screen, "Press ESC to exit", 
               (display_width // 2 - 100, display_height // 3 + 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    click_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append((x, y))
            print(f"✅ Click detected at: ({x}, {y})")
            
            # Draw click indicator
            cv2.circle(test_screen, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(test_screen, f"{len(click_points)}", (x + 15, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Set mouse callback BEFORE showing window
    cv2.namedWindow('AR Sandbox - Contour Lines', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('AR Sandbox - Contour Lines', mouse_callback)
    
    print("Showing mouse test window...")
    print("Try clicking anywhere in the window")
    print("Clicks should be detected and green circles should appear")
    
    while True:
        cv2.imshow('AR Sandbox - Contour Lines', test_screen)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()
    print(f"✅ Test complete! Total clicks detected: {len(click_points)}")

if __name__ == "__main__":
    test_mouse_clicking()