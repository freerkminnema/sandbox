#!/usr/bin/env python3
"""
Test projection alignment system
"""
import cv2
import numpy as np

# Import functions
from real_kinect_sandbox import (
    create_alignment_pattern,
    create_simulated_terrain,
    load_calibration
)

def test_alignment_pattern():
    print("🧪 Testing Projection Alignment Pattern")
    print("=" * 50)
    
    # Load calibration
    load_calibration()
    
    # Create test pattern
    print("Creating alignment pattern...")
    pattern = create_alignment_pattern(640, 480)
    
    # Show the pattern
    print("Displaying alignment pattern...")
    print("This is what you'll see during projection alignment:")
    print("- Yellow crosshairs at corners")
    print("- Magenta circle at center")
    print("- Grid pattern for reference")
    print("- White border")
    
    cv2.imshow('Alignment Pattern Test', cv2.resize(pattern, (800, 600)))
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Alignment pattern test complete!")

if __name__ == "__main__":
    test_alignment_pattern()