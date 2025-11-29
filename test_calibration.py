#!/usr/bin/env python3
"""
Test the calibration system
"""
import json
import os

# Import calibration functions from main script
import sys
sys.path.append('.')
from real_kinect_sandbox import save_calibration, load_calibration, sandbox_corners, sandbox_rotation

def test_calibration():
    print("🧪 Testing Calibration System")
    print("=" * 40)
    
    # Test saving calibration
    print("1. Testing save_calibration()...")
    global sandbox_corners, sandbox_rotation
    sandbox_corners = [(100, 100), (500, 100), (500, 400), (100, 400)]
    sandbox_rotation = 90
    
    if save_calibration():
        print("✅ Save successful")
    else:
        print("❌ Save failed")
        return
    
    # Test loading calibration
    print("\n2. Testing load_calibration()...")
    if load_calibration():
        print("✅ Load successful")
        print(f"   Corners: {sandbox_corners}")
        print(f"   Rotation: {sandbox_rotation}")
    else:
        print("❌ Load failed")
    
    print("\n3. Checking calibration file...")
    if os.path.exists("calibration.json"):
        with open("calibration.json", 'r') as f:
            data = json.load(f)
        print("✅ File exists and is valid JSON")
        print(f"   Contents: {json.dumps(data, indent=2)}")
    else:
        print("❌ File not found")

if __name__ == "__main__":
    test_calibration()