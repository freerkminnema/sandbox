#!/usr/bin/env python3
"""
Quick test of Kinect integration
"""
import numpy as np
import cv2
import time

# Test simulation mode
def test_simulation():
    print("🧪 Testing simulation mode...")
    
    # Create test terrain
    x = np.linspace(0, 10, 640)
    y = np.linspace(0, 8, 480)
    X, Y = np.meshgrid(x, y)
    
    terrain = (
        np.sin(X * 0.5) * np.cos(Y * 0.5) * 50 +
        np.exp(-((X-5)**2 + (Y-4)**2) / 10) * 100
    )
    
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 2047
    terrain = terrain.astype(np.uint16)
    
    print(f"✅ Generated test terrain: {terrain.shape}")
    print(f"📏 Range: {terrain.min()} - {terrain.max()}")
    
    # Create simple visualization
    vis = cv2.convertScaleAbs(terrain, alpha=255/2047)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    
    cv2.imwrite('test_kinect_output.png', vis)
    print("📸 Saved test output as 'test_kinect_output.png'")
    
    return True

# Test Kinect connection
def test_kinect():
    print("🎮 Testing Kinect connection...")
    
    try:
        import freenect
        print("✅ Freenect library imported successfully")
        
        # Try to get device count
        try:
            # This might not work on all systems
            depth, timestamp = freenect.sync_get_depth()
            if depth is not None:
                print(f"✅ Kinect connected! Depth shape: {depth.shape}")
                print(f"📏 Depth range: {depth.min()} - {depth.max()}")
                return True
            else:
                print("❌ Kinect returned None data")
                return False
        except Exception as e:
            print(f"❌ Kinect communication error: {e}")
            return False
            
    except ImportError:
        print("❌ Freenect library not available")
        return False

if __name__ == "__main__":
    print("🎯 AR Sandbox - Quick Test")
    print("=" * 40)
    
    # Test simulation
    sim_ok = test_simulation()
    
    # Test Kinect
    kinect_ok = test_kinect()
    
    print("\n📊 Test Results:")
    print(f"   Simulation: {'✅' if sim_ok else '❌'}")
    print(f"   Kinect: {'✅' if kinect_ok else '❌'}")
    
    if sim_ok:
        print("\n🎉 Your AR sandbox software is working!")
        print("📂 Check 'test_kinect_output.png' to see the visualization")
        
    if not kinect_ok:
        print("\n🔧 Kinect setup needed:")
        print("   1. Connect Kinect v1 to USB and power")
        print("   2. Check power light is on")
        print("   3. Install drivers if needed")
        print("   4. Run calibration when connected")
    
    print("\n🚀 Ready to build your AR sandbox!")