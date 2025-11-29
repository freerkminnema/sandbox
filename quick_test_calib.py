#!/usr/bin/env python3
"""
Quick test of the main application startup
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

# Import and test just the calibration loading
from real_kinect_sandbox import load_calibration, sandbox_corners, sandbox_rotation

print("Testing calibration loading...")
load_calibration()

print(f"Final values:")
print(f"  sandbox_corners: {sandbox_corners}")
print(f"  sandbox_rotation: {sandbox_rotation}")

# Check if calibration file exists
if os.path.exists("calibration.json"):
    print("✅ Calibration file exists")
else:
    print("❌ No calibration file")