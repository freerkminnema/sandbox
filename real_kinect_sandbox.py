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

# Global calibration thresholds (raw Kinect range 0-2047)
WHITE_BROWN_THRESHOLD = 400    # Very close to close boundary
BROWN_GREEN_THRESHOLD = 800   # Close to middle boundary
GREEN_BLUE_THRESHOLD = 1200   # Middle to far boundary

def get_kinect_depth():
    """Get depth data from Kinect or simulation"""
    global freenect
    if KINECT_AVAILABLE:
        try:
            depth, _ = freenect.sync_get_depth()
            if depth is not None:
                # Filter out invalid depth values (too close or too far)
                # Kinect v1: 0 = too close/invalid, 2047 = too far/saturated
                valid_mask = (depth > 0) & (depth < 2047)
                filtered_depth = depth.copy()
                filtered_depth[~valid_mask] = 0  # Set invalid values to 0
                return filtered_depth
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
    # Convert to 8-bit for processing using simple scaling
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

    # Create color image
    colored = np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)

    # Apply color mapping using raw depth values (lower values = closer)
    # Very close objects: white (lowest raw values)
    very_close_mask = (depth_data > 0) & (depth_data < WHITE_BROWN_THRESHOLD)
    colored[very_close_mask] = [255, 255, 255]

    # Close objects: brown (low-mid raw values)
    close_mask = (depth_data >= WHITE_BROWN_THRESHOLD) & (depth_data < BROWN_GREEN_THRESHOLD)
    colored[close_mask] = [19, 69, 139]  # BGR format for brown

    # Mid areas: green (mid-high raw values)
    mid_mask = (depth_data >= BROWN_GREEN_THRESHOLD) & (depth_data < GREEN_BLUE_THRESHOLD)
    colored[mid_mask] = [34, 139, 34]  # Green works the same in RGB/BGR

    # Far areas: blue (highest raw values)
    far_mask = (depth_data >= GREEN_BLUE_THRESHOLD) & (depth_data < 2047)
    colored[far_mask] = [200, 100, 0]  # BGR format for blue

    return colored

def create_elevation_colors_with_thresholds(depth_data, white_brown_thresh, brown_green_thresh, green_blue_thresh):
    """Create elevation-based color mapping using custom thresholds"""
    # Create color image
    colored = np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.uint8)

    # Apply color mapping using raw depth values (lower values = closer)
    # Very close objects: white (lowest raw values)
    very_close_mask = (depth_data > 0) & (depth_data < white_brown_thresh)
    colored[very_close_mask] = [255, 255, 255]

    # Close objects: brown (low-mid raw values)
    close_mask = (depth_data >= white_brown_thresh) & (depth_data < brown_green_thresh)
    colored[close_mask] = [19, 69, 139]  # BGR format for brown

    # Mid areas: green (mid-high raw values)
    mid_mask = (depth_data >= brown_green_thresh) & (depth_data < green_blue_thresh)
    colored[mid_mask] = [34, 139, 34]  # Green works the same in RGB/BGR

    # Far areas: blue (highest raw values)
    far_mask = (depth_data >= green_blue_thresh) & (depth_data < 2047)
    colored[far_mask] = [200, 100, 0]  # BGR format for blue

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
    print("  Press 'd' to toggle debug info")
    print("  Press 'f' to toggle fullscreen")
    
    if not KINECT_AVAILABLE:
        print("⚠️  Running in simulation mode - connect Kinect for real data")
    
    # Display mode
    mode = 'combined'  # 'combined', 'contours', 'colors'
    fullscreen = False
    debug_mode = False
    
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
            
            # Debug info
            if debug_mode:
                # Get center depth value
                height, width = depth_data.shape
                center_x, center_y = width // 2, height // 2
                center_depth = depth_data[center_y, center_x]
                
                # Determine color band
                if center_depth > 0 and center_depth < WHITE_BROWN_THRESHOLD:
                    color_band = "White"
                elif center_depth >= WHITE_BROWN_THRESHOLD and center_depth < BROWN_GREEN_THRESHOLD:
                    color_band = "Brown"
                elif center_depth >= BROWN_GREEN_THRESHOLD and center_depth < GREEN_BLUE_THRESHOLD:
                    color_band = "Green"
                elif center_depth >= GREEN_BLUE_THRESHOLD and center_depth < 2047:
                    color_band = "Blue"
                else:
                    color_band = "Invalid"
                
                # Debug text
                debug_text1 = f"Center Depth: {center_depth}"
                debug_text2 = f"Color Band: {color_band}"
                debug_text3 = f"Thresholds: W:{WHITE_BROWN_THRESHOLD} B:{BROWN_GREEN_THRESHOLD} G:{GREEN_BLUE_THRESHOLD}"
                
                cv2.putText(display_img, debug_text1, (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_img, debug_text2, (10, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_img, debug_text3, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw crosshair at center
                cv2.drawMarker(display_img, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                
                # Show depth value at center
                cv2.putText(display_img, f"{center_depth}", (center_x + 25, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"🔍 Debug mode {'enabled' if debug_mode else 'disabled'}")
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
    print("\nNote: Lower raw depth values = closer objects (0-2047 range)")
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
            print(f"   Debug: Raw (0-2047) | Status")
            
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
                    
                    # Calculate current depth at absolute center point
                    center_x = width // 2
                    center_y = height // 2
                    current_depth = depth[center_y, center_x]

                    # Check if depth is invalid
                    is_invalid = current_depth == 0 or current_depth == 2047

                    # Draw rectangle around calibration region (red if invalid, yellow if valid)
                    rect_color = (0, 0, 255) if is_invalid else (255, 255, 0)  # Red for invalid, yellow for valid
                    cv2.rectangle(colored, (x1, y1), (x2, y2), rect_color, 3)

                    # Debug output to console
                    status = "VALID" if current_depth > 0 and current_depth < 2047 else "INVALID"
                    print(f"\rRaw: {current_depth:4d} | Status: {status}", end="", flush=True)

                    # Show calibration info
                    cv2.putText(colored, f"Calibrating: {boundary_name}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(colored, f"Raw depth: {current_depth}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(colored, f"Threshold: {temp_thresholds[step_idx]}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add color sample squares in top right corner
                    square_size = 40
                    margin = 10
                    start_x = width - (square_size + margin) * 4
                    start_y = margin
                    
                    # White square
                    cv2.rectangle(colored, (start_x, start_y), 
                                 (start_x + square_size, start_y + square_size), 
                                 (255, 255, 255), -1)
                    cv2.putText(colored, "W", (start_x + 12, start_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Brown square
                    cv2.rectangle(colored, (start_x + (square_size + margin), start_y), 
                                 (start_x + (square_size + margin) * 2, start_y + square_size), 
                                 (19, 69, 139), -1)  # BGR format for brown
                    cv2.putText(colored, "B", (start_x + (square_size + margin) + 12, start_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Green square
                    cv2.rectangle(colored, (start_x + (square_size + margin) * 2, start_y), 
                                 (start_x + (square_size + margin) * 3, start_y + square_size), 
                                 (34, 139, 34), -1)  # Green works the same in RGB/BGR
                    cv2.putText(colored, "G", (start_x + (square_size + margin) * 2 + 12, start_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Blue square
                    cv2.rectangle(colored, (start_x + (square_size + margin) * 3, start_y), 
                                 (start_x + (square_size + margin) * 4, start_y + square_size), 
                                 (200, 100, 0), -1)  # BGR format for blue
                    cv2.putText(colored, "L", (start_x + (square_size + margin) * 3 + 12, start_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # Draw crosshair at absolute center point
                    crosshair_color = (0, 0, 255) if is_invalid else (255, 0, 255)  # Red for invalid, magenta for valid
                    cv2.drawMarker(colored, (center_x, center_y), crosshair_color, cv2.MARKER_CROSS, 20, 3)

                    # Show depth value or invalid distance text
                    if is_invalid:
                        cv2.putText(colored, "INVALID DISTANCE", (center_x + 25, center_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(colored, f"{current_depth}", (center_x + 25, center_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    cv2.imshow(f'Calibration Step {step_idx + 1}: {boundary_name}', colored)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        # Capture absolute center point depth
                        center_x = width // 2
                        center_y = height // 2
                        center_depth = depth[center_y, center_x]

                        captured_depths.append(center_depth)
                        print(f"\n✅ Captured center point: Raw={center_depth}")
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
        # Store raw depth values directly as thresholds
        WHITE_BROWN_THRESHOLD = captured_depths[0]
        BROWN_GREEN_THRESHOLD = captured_depths[1]
        GREEN_BLUE_THRESHOLD = captured_depths[2]
        
        print(f"\n✅ Calibration complete!")
        print(f"📊 New thresholds (raw 0-2047 range):")
        print(f"   White→Brown: {WHITE_BROWN_THRESHOLD}")
        print(f"   Brown→Green: {BROWN_GREEN_THRESHOLD}")
        print(f"   Green→Blue: {GREEN_BLUE_THRESHOLD}")
    else:
        print("⚠️  Incomplete calibration - using default thresholds")

if __name__ == "__main__":
    print("🎯 AR Sandbox - Real Kinect Integration")
    print("=" * 50)
    
    # Run calibration first
    calibrate_kinect()
    
    # Start real-time sandbox
    run_realtime_sandbox()