#!/usr/bin/env python3
"""
Test script to verify Kinect v1 connection and depth data capture
"""
import freenect
import numpy as np
import cv2

def test_kinect_connection():
    """Test if Kinect is connected and can capture depth data"""
    print("Testing Kinect v1 connection...")
    
    try:
        # Try to get depth data directly
        print("Attempting to capture depth data...")
        depth, timestamp = freenect.sync_get_depth()
        
        if depth is not None:
            print(f"Successfully captured depth frame: {depth.shape}")
            print(f"Depth range: {depth.min()} to {depth.max()}")
            
            # Normalize depth for visualization
            depth_normalized = cv2.convertScaleAbs(depth, alpha=255/depth.max())
            
            # Save depth visualization
            cv2.imwrite('depth_test.png', depth_normalized)
            print("Depth visualization saved as 'depth_test.png'")
            
            return True
        else:
            print("Failed to capture depth data")
            return False
            
    except Exception as e:
        print(f"Error testing Kinect connection: {e}")
        print("This usually means:")
        print("1. Kinect is not connected")
        print("2. Kinect is not powered (check power light)")
        print("3. USB drivers are not working")
        return False

def test_rgb_capture():
    """Test RGB camera capture"""
    print("\nTesting RGB camera capture...")
    
    try:
        rgb, timestamp = freenect.sync_get_video()
        
        if rgb is not None:
            print(f"Successfully captured RGB frame: {rgb.shape}")
            cv2.imwrite('rgb_test.png', rgb)
            print("RGB image saved as 'rgb_test.png'")
            return True
        else:
            print("Failed to capture RGB data")
            return False
            
    except Exception as e:
        print(f"Error testing RGB capture: {e}")
        return False

if __name__ == "__main__":
    print("=== Kinect v1 Test Script ===")
    print("Make sure your Kinect v1 is:")
    print("1. Connected to both USB and power")
    print("2. Power light is ON (green)")
    print("3. USB cable is properly connected")
    print()
    
    depth_success = test_kinect_connection()
    rgb_success = test_rgb_capture()
    
    if depth_success:
        print("\n✅ Depth camera working!")
    else:
        print("\n❌ Depth camera failed")
        
    if rgb_success:
        print("✅ RGB camera working!")
    else:
        print("❌ RGB camera failed")
    
    if depth_success or rgb_success:
        print("\n🎉 Kinect is partially or fully functional!")
    else:
        print("\n💥 Kinect connection failed - check hardware and drivers")