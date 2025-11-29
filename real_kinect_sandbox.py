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

# Global calibration thresholds (0-255 range)
WHITE_BROWN_THRESHOLD = 64    # Very close to close boundary
BROWN_GREEN_THRESHOLD = 128   # Close to middle boundary  
GREEN_BLUE_THRESHOLD = 192     # Middle to far boundary

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
    """Create elevation-based color mapping using calibrated thresholds"""
    global WHITE_BROWN_THRESHOLD, BROWN_GREEN_THRESHOLD, GREEN_BLUE_THRESHOLD
    
    # Normalize to 0-255 range
    depth_8bit = cv2.convertScaleAbs(depth_data, alpha=255/2047)
    
    # Create color image
    colored = np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)
    
    # Apply calibrated color mapping
    # Very close objects: white
    very_close_mask = depth_8bit < WHITE_BROWN_THRESHOLD
    colored[very_close_mask] = [255, 255, 255]
    
    # Close objects: brown
    close_mask = (depth_8bit >= WHITE_BROWN_THRESHOLD) & (depth_8bit < BROWN_GREEN_THRESHOLD)
    colored[close_mask] = [139, 69, 19]
    
    # Mid areas: green
    mid_mask = (depth_8bit >= BROWN_GREEN_THRESHOLD) & (depth_8bit < GREEN_BLUE_THRESHOLD)
    colored[mid_mask] = [34, 139, 34]
    
    # Far areas: blue
    far_mask = depth_8bit >= GREEN_BLUE_THRESHOLD
    colored[far_mask] = [0, 100, 200]
    
    return colored

def create_elevation_colors_with_thresholds(depth_data, white_brown_thresh, brown_green_thresh, green_blue_thresh):
    """Create elevation-based color mapping using custom thresholds"""
    # Normalize to 0-255 range
    depth_8bit = cv2.convertScaleAbs(depth_data, alpha=255/2047)
    
    # Create color image
    colored = np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)
    
    # Apply custom color mapping
    # Very close objects: white
    very_close_mask = depth_8bit < white_brown_thresh
    colored[very_close_mask] = [255, 255, 255]
    
    # Close objects: brown
    close_mask = (depth_8bit >= white_brown_thresh) & (depth_8bit < brown_green_thresh)
    colored[close_mask] = [139, 69, 19]
    
    # Mid areas: green
    mid_mask = (depth_8bit >= brown_green_thresh) & (depth_8bit < green_blue_thresh)
    colored[mid_mask] = [34, 139, 34]
    
    # Far areas: blue
    far_mask = depth_8bit >= green_blue_thresh
    colored[far_mask] = [0, 100, 200]
    
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
    """Calibrate color boundaries for depth mapping"""
    global freenect, WHITE_BROWN_THRESHOLD, BROWN_GREEN_THRESHOLD, GREEN_BLUE_THRESHOLD
    print("🔧 Kinect Color Boundary Calibration")
    print("This will set the exact distances where colors change:")
    print("  1. White → Brown (very close to close)")
    print("  2. Brown → Green (close to middle)")  
    print("  3. Green → Blue (middle to far)")
    print("\nInstructions:")
    print("- Place a flat surface at the desired boundary distance")
    print("- Press 'c' to capture the depth value")
    print("- Press 'q' to quit calibration")
    print("- Colors update in real-time as you adjust thresholds")
    
    if not KINECT_AVAILABLE:
        print("❌ Kinect not available - using default thresholds")
        return
    
    calibration_steps = [
        ("White → Brown", "very close to close boundary"),
        ("Brown → Green", "close to middle boundary"),
        ("Green → Blue", "middle to far boundary")
    ]
    
    captured_depths = []
    temp_thresholds = [WHITE_BROWN_THRESHOLD, BROWN_GREEN_THRESHOLD, GREEN_BLUE_THRESHOLD]
    
    try:
        for step_idx, (boundary_name, description) in enumerate(calibration_steps):
            print(f"\n📍 Step {step_idx + 1}: {boundary_name}")
            print(f"   Position surface at {description}")
            print(f"   Press 'c' to capture, 'q' to quit")
            print(f"   Use arrow keys to adjust threshold in real-time")
            
            captured = False
            while not captured:
                depth, _ = freenect.sync_get_depth()
                if depth is not None:
                    # Create color visualization with current thresholds
                    colored = create_elevation_colors_with_thresholds(depth, temp_thresholds[0], temp_thresholds[1], temp_thresholds[2])
                    
                    # Highlight center calibration region
                    height, width = depth.shape
                    center_x, center_y = width // 2, height // 2
                    region_size = min(width, height) // 8
                    
                    x1 = max(0, center_x - region_size)
                    x2 = min(width, center_x + region_size)
                    y1 = max(0, center_y - region_size)
                    y2 = min(height, center_y + region_size)
                    
                    # Draw rectangle around calibration region
                    cv2.rectangle(colored, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    
                    # Calculate current depth in calibration region
                    center_region = depth[y1:y2, x1:x2]
                    current_depth = center_region.mean()
                    current_depth_8bit = cv2.convertScaleAbs(np.array([[current_depth]]), alpha=255/2047)[0,0]
                    
                    # Show calibration info
                    cv2.putText(colored, f"Calibrating: {boundary_name}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(colored, f"Current depth: {current_depth_8bit}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(colored, f"Threshold: {temp_thresholds[step_idx]}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show depth value next to calibration square
                    cv2.putText(colored, f"{current_depth_8bit}", (x2 + 10, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    cv2.imshow(f'Calibration Step {step_idx + 1}: {boundary_name}', colored)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        # Capture center region depth
                        center_region = depth[y1:y2, x1:x2]
                        avg_depth = center_region.mean()
                        captured_depths.append(avg_depth)
                        print(f"✅ Captured center region: {avg_depth:.1f} (raw Kinect depth)")
                        captured = True
                    elif key == ord('q'):
                        print("❌ Calibration cancelled")
                        cv2.destroyAllWindows()
                        return
                    elif key == 81:  # Left arrow
                        temp_thresholds[step_idx] = max(0, temp_thresholds[step_idx] - 5)
                        print(f"Threshold: {temp_thresholds[step_idx]}")
                    elif key == 83:  # Right arrow
                        temp_thresholds[step_idx] = min(255, temp_thresholds[step_idx] + 5)
                        print(f"Threshold: {temp_thresholds[step_idx]}")
                        
    except Exception as e:
        print(f"Calibration error: {e}")
        cv2.destroyAllWindows()
        return
    
    cv2.destroyAllWindows()
    
    if len(captured_depths) == 3:
        # Convert raw Kinect depth values to 0-255 range
        min_depth = min(captured_depths)
        max_depth = max(captured_depths)
        depth_range = max_depth - min_depth
        
        if depth_range > 0:
            # Map captured depths to 0-255 range
            white_brown_raw = captured_depths[0]
            brown_green_raw = captured_depths[1] 
            green_blue_raw = captured_depths[2]
            
            # Convert to 0-255 scale
            WHITE_BROWN_THRESHOLD = int(((white_brown_raw - min_depth) / depth_range) * 255)
            BROWN_GREEN_THRESHOLD = int(((brown_green_raw - min_depth) / depth_range) * 255)
            GREEN_BLUE_THRESHOLD = int(((green_blue_raw - min_depth) / depth_range) * 255)
            
            print(f"\n✅ Calibration complete!")
            print(f"📊 New thresholds (0-255 range):")
            print(f"   White→Brown: {WHITE_BROWN_THRESHOLD}")
            print(f"   Brown→Green: {BROWN_GREEN_THRESHOLD}")
            print(f"   Green→Blue: {GREEN_BLUE_THRESHOLD}")
        else:
            print("⚠️  Insufficient depth range - using default thresholds")
    else:
        print("⚠️  Incomplete calibration - using default thresholds")

if __name__ == "__main__":
    print("🎯 AR Sandbox - Real Kinect Integration")
    print("=" * 50)
    
    # Run calibration first
    calibrate_kinect()
    
    # Start real-time sandbox
    run_realtime_sandbox()