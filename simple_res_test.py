#!/usr/bin/env python3
"""
Simple resolution test
"""
import cv2
import numpy as np

# Test different resolutions
resolutions = [
    (1280, 720),   # 720p
    (1920, 1080),  # 1080p
    (800, 600),    # Default
]

for width, height in resolutions:
    print(f"Testing {width}x{height}...")
    
    # Create test pattern
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add resolution text
    cv2.putText(pattern, f"{width}x{height}", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Add corner markers
    cv2.circle(pattern, (0, 0), 20, (255, 0, 0), -1)      # TL - Red
    cv2.circle(pattern, (width-1, 0), 20, (0, 255, 0), -1)  # TR - Green  
    cv2.circle(pattern, (width-1, height-1), 20, (0, 0, 255), -1)  # BR - Blue
    cv2.circle(pattern, (0, height-1), 20, (255, 255, 0), -1)  # BL - Yellow
    
    # Show briefly
    cv2.imshow(f'Resolution Test - {width}x{height}', pattern)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

print("✅ Resolution test complete!")