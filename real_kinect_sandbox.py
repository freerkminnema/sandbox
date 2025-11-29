#!/usr/bin/env python3
"""
Real Kinect v1 AR Sandbox Integration
Connects to actual Kinect hardware for real-time contour line projection
"""
import numpy as np
import cv2
import sys
import time

# Add freenect to path
try:
    import freenect
    KINECT_AVAILABLE = True
    print("✅ Kinect library loaded successfully")
except ImportError:
    KINECT_AVAILABLE = False
    print("❌ Kinect library not available - using simulation mode")

def get_kinect_depth():
    """Get depth data from Kinect or simulation"""
    global freenect
    if KINECT_AVAILABLE:
        try:
            depth, _ = freenect.sync_get_depth()
            if depth is not None:
                return depth
        except Exception as e:
            print(f"Kinect error: {e}, falling back to simulation")
    
    # Fallback to simulation
    return create_simulated_terrain()

def create_simulated_terrain(width=640, height=480):
    """Create simulated terrain data"""
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 8, height)
    X, Y = np.meshgrid(x, y)
    
    # Dynamic terrain that changes over time
    t = time.time() * 0.1
    terrain = (
        np.sin(X * 0.5 + t) * np.cos(Y * 0.5) * 50 +
        np.exp(-((X-5)**2 + (Y-4)**2) / 10) * 100 +
        np.exp(-((X-2)**2 + (Y-6)**2) / 8) * 80 +
        np.random.normal(0, 5, (height, width))
    )
    
    # Normalize to Kinect depth range (0-2047 for Kinect v1)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 2047
    return terrain.astype(np.uint16)

def process_depth_to_contours(depth_data):
    """Convert depth data to contour line visualization"""
    # Convert to 8-bit for processing
    depth_8bit = cv2.convertScaleAbs(depth_data, alpha=255/2047)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(depth_8bit, (5, 5), 0)
    
    # Create contour lines
    contours_img = np.zeros_like(blurred)
    
    # Generate multiple contour levels
    levels = 20
    for i in range(levels):
        level = i * 255 // levels
        # Create binary mask
        _, mask = cv2.threshold(blurred, level, 255, cv2.THRESH_BINARY)
        
        # Find and draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, -1, 255, 1)
    
    return contours_img

def create_elevation_colors(depth_data):
    """Create elevation-based color mapping"""
    # Normalize to 0-255 range
    depth_8bit = cv2.convertScaleAbs(depth_data, alpha=255/2047)
    
    # Create color image
    colored = np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)
    
    # Apply terrain-like colormap
    # Low areas: blue (water)
    low_mask = depth_8bit < 85
    colored[low_mask] = [0, 100, 200]
    
    # Mid areas: green (land)  
    mid_mask = (depth_8bit >= 85) & (depth_8bit < 170)
    colored[mid_mask] = [34, 139, 34]
    
    # High areas: brown (mountains)
    high_mask = depth_8bit >= 170
    colored[high_mask] = [139, 69, 19]
    
    return colored

def create_ar_overlay(depth_data):
    """Create complete AR overlay with contours and colors"""
    # Generate components
    contours = process_depth_to_contours(depth_data)
    colors = create_elevation_colors(depth_data)
    
    # Combine colors and contours
    overlay = cv2.addWeighted(colors, 0.7, cv2.cvtColor(contours, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    return overlay

def run_realtime_sandbox():
    """Run real-time AR sandbox with Kinect"""
    print("🚀 Starting AR Sandbox...")
    print("Controls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save current frame")
    print("  Press 'c' to toggle contours only")
    print("  Press 'e' to toggle elevation colors only")
    print("  Press 'f' to toggle fullscreen")
    
    if not KINECT_AVAILABLE:
        print("⚠️  Running in simulation mode - connect Kinect for real data")
    
    # Display mode
    mode = 'combined'  # 'combined', 'contours', 'colors'
    fullscreen = False
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    # Create window
    cv2.namedWindow('AR Sandbox - Contour Lines', cv2.WINDOW_NORMAL)

    try:
        while True:
            # Get depth data
            depth_data = get_kinect_depth()
            
            # Create AR overlay
            if mode == 'combined':
                display_img = create_ar_overlay(depth_data)
            elif mode == 'contours':
                display_img = cv2.cvtColor(process_depth_to_contours(depth_data), cv2.COLOR_GRAY2BGR)
            else:  # colors
                display_img = create_elevation_colors(depth_data)
            
            # Resize for display
            display_img = cv2.resize(display_img, (800, 600))
            
            # Calculate FPS
            if frame_count > 0:
                fps = frame_count / (time.time() - start_time + 0.001)
            
            # Add info text
            cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            mode_text = f"Mode: {mode} | {'Kinect' if KINECT_AVAILABLE else 'Simulation'}"
            cv2.putText(display_img, mode_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show image
            cv2.imshow('AR Sandbox - Contour Lines', display_img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"ar_sandbox_capture_{int(time.time())}.png"
                cv2.imwrite(filename, display_img)
                print(f"📸 Saved: {filename}")
            elif key == ord('c'):
                mode = 'contours' if mode != 'contours' else 'combined'
                print(f"🎨 Switched to {mode} mode")
            elif key == ord('e'):
                mode = 'colors' if mode != 'colors' else 'combined'
                print(f"🎨 Switched to {mode} mode")
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty('AR Sandbox - Contour Lines', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                print(f"🖥️  Switched to {'fullscreen' if fullscreen else 'windowed'} mode")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n👋 Stopping AR Sandbox...")
    
    cv2.destroyAllWindows()
    print(f"📊 Processed {frame_count} frames at {fps:.1f} FPS")

def calibrate_kinect():
    """Simple Kinect calibration routine"""
    global freenect
    print("🔧 Kinect Calibration")
    print("Place a flat surface at different distances and press 'c' to capture each position")
    print("Press 'q' when done")
    
    if not KINECT_AVAILABLE:
        print("❌ Kinect not available - skipping calibration")
        return
    
    calibrations = []
    
    try:
        while len(calibrations) < 5:
            depth, _ = freenect.sync_get_depth()
            if depth is not None:
                # Show depth image
                depth_vis = cv2.convertScaleAbs(depth, alpha=255/2047)
                cv2.imshow('Kinect Calibration - Press c to capture, q to quit', depth_vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    calibrations.append(depth.mean())
                    print(f"📏 Calibration point {len(calibrations)}: {depth.mean():.1f}")
                elif key == ord('q'):
                    break
                    
    except Exception as e:
        print(f"Calibration error: {e}")
    
    cv2.destroyAllWindows()
    
    if len(calibrations) > 1:
        print(f"✅ Calibration complete: {len(calibrations)} points captured")
        print(f"📊 Depth range: {min(calibrations):.1f} - {max(calibrations):.1f}")

if __name__ == "__main__":
    print("🎯 AR Sandbox - Real Kinect Integration")
    print("=" * 50)
    
    # Run calibration first
    calibrate_kinect()
    
    # Start real-time sandbox
    run_realtime_sandbox()