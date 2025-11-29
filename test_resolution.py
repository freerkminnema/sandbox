#!/usr/bin/env python3
"""
Test resolution detection system
"""
import sys
sys.path.append('.')

from real_kinect_sandbox import detect_display_resolution, display_width, display_height

def test_resolution_detection():
    print("🧪 Testing Resolution Detection System")
    print("=" * 50)
    
    print("1. Testing auto-detection...")
    detect_display_resolution()
    print(f"   Detected resolution: {display_width}x{display_height}")
    
    print("\n2. Testing alignment pattern creation...")
    from real_kinect_sandbox import create_alignment_pattern
    pattern = create_alignment_pattern(display_width, display_height)
    print(f"   Pattern created: {pattern.shape}")
    
    print("\n3. Displaying pattern...")
    import cv2
    cv2.imshow(f'Resolution Test - {display_width}x{display_height}', pattern)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Resolution detection test complete!")

if __name__ == "__main__":
    test_resolution_detection()