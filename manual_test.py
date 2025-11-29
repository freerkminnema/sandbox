#!/usr/bin/env python3
"""
Manual test of calibration system
"""
import json
import os

# Test manual save
calib = {
    'sandbox_corners': [[100, 100], [500, 100], [500, 400], [100, 400]],
    'rotation': 90,
    'depth_thresholds': {
        'white_brown': 400,
        'brown_green': 800,
        'green_blue': 1200
    }
}

print("Saving test calibration...")
with open("calibration.json", 'w') as f:
    json.dump(calib, f, indent=2)

print("Reading back...")
with open("calibration.json", 'r') as f:
    loaded = json.load(f)

print("Loaded calibration:")
print(json.dumps(loaded, indent=2))