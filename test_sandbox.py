#!/usr/bin/env python3
"""
End-to-end test of the enhanced calibration system
"""
import cv2
import numpy as np
import time

# Import the main functions
from real_kinect_sandbox import (
    load_calibration, save_calibration, 
    sandbox_corners, sandbox_rotation,
    create_elevation_colors, apply_sandbox_transformation,
    create_simulated_terrain
)

def test_sandbox_transformation():
    print("🧪 Testing Sandbox Transformation System")
    print("=" * 50)
    
    # Load calibration
    print("1. Loading calibration...")
    load_calibration()
    
    # Create test terrain
    print("2. Creating test terrain...")
    depth_data = create_simulated_terrain()
    colors = create_elevation_colors(depth_data)
    
    print(f"   Original image size: {colors.shape}")
    
    # Apply transformation
    print("3. Applying sandbox transformation...")
    transformed = apply_sandbox_transformation(colors)
    
    print(f"   Transformed image size: {transformed.shape}")
    
    # Show results
    print("4. Displaying results...")
    cv2.imshow('Original', cv2.resize(colors, (400, 300)))
    cv2.imshow('Transformed', cv2.resize(transformed, (400, 300)))
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✅ Test completed!")

if __name__ == "__main__":
    test_sandbox_transformation()