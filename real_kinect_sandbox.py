#!/usr/bin/env python3
"""
Real Kinect v1 AR Sandbox Integration
Connects to actual Kinect hardware for real-time contour line projection
"""
import numpy as np
import cv2
import sys
import time
import json
import os

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

# Sandbox calibration variables
sandbox_corners = None  # Will store 4 corner points: [top-left, top-right, bottom-right, bottom-left]
sandbox_rotation = 0    # Current rotation: 0, 90, 180, 270 degrees
sandbox_mask = None     # Pre-computed mask for sandbox area
calibration_file = "calibration.json"

# Display resolution variables
display_width = 1280    # Default to 720p
display_height = 720    # Default to 720p
fullscreen_mode = False # Track current display mode

def detect_display_resolution():
    """Auto-detect optimal display resolution"""
    global display_width, display_height
    
    # Try to detect primary display resolution
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Common projector resolutions
        common_resolutions = [
            (1280, 720),   # 720p (HD)
            (1920, 1080),  # 1080p (Full HD)
            (1024, 768),   # XGA
            (800, 600),     # SVGA
            (1280, 1024),  # SXGA
        ]
        
        # Find best match (prefer 720p for projectors)
        for width, height in common_resolutions:
            if width <= screen_width and height <= screen_height:
                display_width, display_height = width, height
                if width == 1280 and height == 720:
                    break  # Prefer 720p for projectors
        
        return True
        
    except ImportError:
        display_width, display_height = 1280, 720
        return False
    except Exception as e:
        display_width, display_height = 1280, 720
        return False

def load_calibration():
    """Load calibration settings from file"""
    global sandbox_corners, sandbox_rotation, WHITE_BROWN_THRESHOLD, BROWN_GREEN_THRESHOLD, GREEN_BLUE_THRESHOLD, display_width, display_height
    
    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, 'r') as f:
                calib = json.load(f)
            
            corners = calib.get('sandbox_corners', None)
            if corners is not None:
                sandbox_corners = [(int(x), int(y)) for x, y in corners]
            else:
                sandbox_corners = None
                
            sandbox_rotation = int(calib.get('rotation', 0))
            
            thresholds = calib.get('depth_thresholds', {})
            WHITE_BROWN_THRESHOLD = int(thresholds.get('white_brown', 400))
            BROWN_GREEN_THRESHOLD = int(thresholds.get('brown_green', 800))
            GREEN_BLUE_THRESHOLD = int(thresholds.get('green_blue', 1200))
            
            print(f"✅ Calibration loaded from {calibration_file}")
            if sandbox_corners is not None:
                print(f"   Sandbox corners: {sandbox_corners}")
                print(f"   Rotation: {sandbox_rotation}°")
            print(f"   Depth thresholds: W→B:{WHITE_BROWN_THRESHOLD}, B→G:{BROWN_GREEN_THRESHOLD}, G→L:{GREEN_BLUE_THRESHOLD}")
            return True
        except Exception as e:
            print(f"⚠️  Error loading calibration: {e}")
            return False
    else:
        print("📝 No calibration file found - using defaults")
        return False

def save_calibration():
    """Save current calibration settings to file"""
    global sandbox_corners, sandbox_rotation, WHITE_BROWN_THRESHOLD, BROWN_GREEN_THRESHOLD, GREEN_BLUE_THRESHOLD
    
    # Convert numpy types to regular Python types for JSON serialization
    corners_to_save = None
    if sandbox_corners is not None:
        corners_to_save = [[int(x), int(y)] for x, y in sandbox_corners]
    
    calib = {
        'sandbox_corners': corners_to_save,
        'rotation': int(sandbox_rotation),
        'depth_thresholds': {
            'white_brown': int(WHITE_BROWN_THRESHOLD),
            'brown_green': int(BROWN_GREEN_THRESHOLD),
            'green_blue': int(GREEN_BLUE_THRESHOLD)
        }
    }
    
    try:
        with open(calibration_file, 'w') as f:
            json.dump(calib, f, indent=2)
        print(f"💾 Calibration saved to {calibration_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving calibration: {e}")
        return False

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

def create_sandbox_mask(corners, image_size):
    """Create a binary mask for the sandbox area"""
    global sandbox_mask
    
    if corners is None or len(corners) != 4:
        return None
    
    mask = np.zeros(image_size[:2], dtype=np.uint8)
    
    # Convert corners to proper format
    pts = np.array(corners, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Fill the polygon
    cv2.fillPoly(mask, [pts], 255)
    
    sandbox_mask = mask
    return mask

def apply_sandbox_transformation(image):
    """Apply sandbox calibration transformation to image"""
    global sandbox_corners, sandbox_rotation, sandbox_mask
    
    if sandbox_corners is None:
        return image
    
    # Apply rotation if needed
    if sandbox_rotation != 0:
        if sandbox_rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif sandbox_rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif sandbox_rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Create or use existing mask
    if sandbox_mask is None:
        create_sandbox_mask(sandbox_corners, image.shape)
    
    # Apply mask - set everything outside sandbox to black
    if sandbox_mask is not None:
        # Resize mask to match image if needed
        if sandbox_mask.shape[:2] != image.shape[:2]:
            sandbox_mask_resized = cv2.resize(sandbox_mask, (image.shape[1], image.shape[0]))
        else:
            sandbox_mask_resized = sandbox_mask
        
        # Create 3-channel mask for color image
        if len(image.shape) == 3:
            mask_3d = cv2.cvtColor(sandbox_mask_resized, cv2.COLOR_GRAY2BGR)
            image = cv2.bitwise_and(image, mask_3d)
        else:
            image = cv2.bitwise_and(image, sandbox_mask_resized)
    
    return image

def calibrate_sandbox():
    """Interactive sandbox dimension calibration"""
    global sandbox_corners
    
    # Get current depth data for background
    depth_data = get_kinect_depth()
    if depth_data is None:
        print("❌ Cannot get depth data for calibration")
        return False
    
    # Create color visualization
    colored = create_elevation_colors(depth_data)
    colored = cv2.resize(colored, (display_width, display_height))
    
    corners = []
    corner_labels = ["TL", "TR", "BR", "BL"]
    corner_instructions = [
        "Click TOP-LEFT corner",
        "Click TOP-RIGHT corner", 
        "Click BOTTOM-RIGHT corner",
        "Click BOTTOM-LEFT corner"
    ]
    
    # Mouse callback for corner selection (defined once)
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            print(f"✅ Corner {len(corners)}: ({x}, {y})")
    
    # Set mouse callback once before the loop
    cv2.setMouseCallback('AR Sandbox - Contour Lines', mouse_callback)
    
    while True:
        # Create copy for drawing
        display = colored.copy()
        
        # Draw on-screen instructions
        current_step = len(corners)
        if current_step < 4:
            # Main instruction
            cv2.putText(display, corner_instructions[current_step], 
                       (display_width // 2 - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Progress indicator
            cv2.putText(display, f"Corner {current_step + 1} of 4", 
                       (display_width // 2 - 80, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Corner status list
            for i in range(4):
                if i < current_step:
                    # Completed corner
                    cv2.putText(display, f"✓ {corner_labels[i]}", 
                               (30, 150 + i * 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif i == current_step:
                    # Current corner
                    cv2.putText(display, f"→ {corner_labels[i]}", 
                               (30, 150 + i * 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Pending corner
                    cv2.putText(display, f"  {corner_labels[i]}", 
                               (30, 150 + i * 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        else:
            # All corners selected
            cv2.putText(display, "Press ENTER to confirm or ESC to cancel", 
                       (display_width // 2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Control instructions
        cv2.putText(display, "Click to select | C: Clear | ESC: Cancel | ENTER: Confirm", 
                   (10, display_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw existing corners
        for i, corner in enumerate(corners):
            cv2.circle(display, corner, 8, (0, 255, 255), -1)
            cv2.putText(display, corner_labels[i], (corner[0] + 10, corner[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw lines between corners
        if len(corners) >= 2:
            for i in range(len(corners)):
                next_i = (i + 1) % len(corners)
                cv2.line(display, corners[i], corners[next_i], (0, 255, 0), 2)
        
        cv2.imshow('AR Sandbox - Contour Lines', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("❌ Calibration cancelled")
            return False
        elif key == ord('c'):  # Clear
            corners = []
            print("🗑️  Corners cleared")
        elif key == 13 and len(corners) == 4:  # Enter
            # Scale corners back to original image size
            height, width = depth_data.shape
            scale_x = width / display_width
            scale_y = height / display_height
            
            sandbox_corners = [
                (int(corners[0][0] * scale_x), int(corners[0][1] * scale_y)),  # Top-left
                (int(corners[1][0] * scale_x), int(corners[1][1] * scale_y)),  # Top-right
                (int(corners[2][0] * scale_x), int(corners[2][1] * scale_y)),  # Bottom-right
                (int(corners[3][0] * scale_x), int(corners[3][1] * scale_y))   # Bottom-left
            ]
            
            print(f"✅ Sandbox corners calibrated: {sandbox_corners}")
            return True
    
    cv2.destroyAllWindows()
    return False

def create_alignment_pattern(width, height):
    """Create a distinctive alignment pattern for projection calibration"""
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add grid pattern (subtle) - scale with resolution
    grid_spacing = max(40, min(width, height) // 16)
    grid_color = (50, 50, 50)  # Dark gray grid
    for i in range(0, width, grid_spacing):
        cv2.line(pattern, (i, 0), (i, height), grid_color, 1)
    for i in range(0, height, grid_spacing):
        cv2.line(pattern, (0, i), (width, i), grid_color, 1)
    
    # Add corner crosshairs (very visible) - scale with resolution
    crosshair_color = (0, 255, 255)  # Yellow crosshairs
    crosshair_size = max(30, min(width, height) // 20)
    
    # Corner positions (will be updated during calibration)
    corner_positions = [
        (width // 4, height // 4),      # Top-left
        (3 * width // 4, height // 4),   # Top-right
        (3 * width // 4, 3 * height // 4), # Bottom-right
        (width // 4, 3 * height // 4)    # Bottom-left
    ]
    
    for x, y in corner_positions:
        # Draw crosshair
        cv2.line(pattern, (x - crosshair_size, y), (x + crosshair_size, y), crosshair_color, 3)
        cv2.line(pattern, (x, y - crosshair_size), (x, y + crosshair_size), crosshair_color, 3)
        cv2.circle(pattern, (x, y), 5, crosshair_color, -1)
    
    # Add center marker - scale with resolution
    center_x, center_y = width // 2, height // 2
    center_radius = max(10, min(width, height) // 60)
    cv2.circle(pattern, (center_x, center_y), center_radius, (255, 0, 255), 2)  # Magenta center
    
    # Add border - scale with resolution
    border_margin = max(10, min(width, height) // 64)
    cv2.rectangle(pattern, (border_margin, border_margin), 
                 (width - border_margin, height - border_margin), (255, 255, 255), 2)
    
    # Add resolution info
    font_scale = max(0.5, min(width, height) / 1000)
    cv2.putText(pattern, f"{width}x{height}", (10, height - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return pattern

def calibrate_projection_alignment():
    """Calibrate projection to align with physical sandbox"""
    global sandbox_corners
    
    print("🎯 Projection Alignment Calibration")
    print("This ensures your projected overlay matches the physical sandbox")
    print("\nInstructions:")
    print("1. Project this calibration pattern onto your sandbox")
    print("2. Adjust until corner markers align with your sandbox corners")
    print("3. Use arrow keys to fine-tune each corner")
    print("4. Press TAB to switch between corners")
    print("5. Press ENTER when satisfied with alignment")
    print("\nControls:")
    print("  Arrow Keys: Move selected corner")
    print("  TAB: Next corner")
    print("  SHIFT+TAB: Previous corner")
    print("  ENTER: Confirm alignment")
    print("  ESC: Cancel")
    
    if sandbox_corners is None:
        print("❌ No sandbox corners defined. Please calibrate sandbox dimensions first.")
        return False
    
    # Get depth data to know the original resolution for scaling
    depth_data = get_kinect_depth()
    if depth_data is None:
        print("❌ Cannot get depth data for alignment")
        return False
    
    # Create calibration pattern at DISPLAY resolution (not depth resolution)
    # This ensures the full screen is used for projection alignment
    pattern = create_alignment_pattern(display_width, display_height)
    
    # Add corner markers
    corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
    corner_labels = ["TL", "TR", "BR", "BL"]
    
    # Working copy of corners - scale from depth resolution to display resolution
    depth_height, depth_width = depth_data.shape
    scale_x = display_width / depth_width
    scale_y = display_height / depth_height
    
    working_corners = [
        [int(corner[0] * scale_x), int(corner[1] * scale_y)] 
        for corner in sandbox_corners
    ]
    selected_corner = 0
    
    while True:
        # Create display image - start fresh from pattern each frame
        display = pattern.copy()
        
        # Draw corner markers on the FULL screen pattern
        for i, corner in enumerate(working_corners):
            color = corner_colors[i]
            label = corner_labels[i]
            
            # Highlight selected corner
            thickness = 3 if i == selected_corner else 2
            radius = 15 if i == selected_corner else 10
            
            cv2.circle(display, tuple(corner), radius, color, thickness)
            cv2.putText(display, label, (corner[0] + 20, corner[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw connecting lines
            next_i = (i + 1) % 4
            cv2.line(display, tuple(corner), tuple(working_corners[next_i]), color, 2)
        
        # Add on-screen instructions
        # Main instruction
        cv2.putText(display, "PROJECTION ALIGNMENT", 
                   (display_width // 2 - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # Current corner indicator
        cv2.putText(display, f"Adjusting: {corner_labels[selected_corner]} corner", 
                   (display_width // 2 - 120, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Corner status
        for i in range(4):
            if i == selected_corner:
                # Current corner
                cv2.putText(display, f"→ {corner_labels[i]}", 
                           (30, 140 + i * 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Other corners
                cv2.putText(display, f"  {corner_labels[i]}", 
                           (30, 140 + i * 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Control instructions
        cv2.putText(display, "Arrow Keys: Move corner | TAB: Next corner | ENTER: Confirm | ESC: Cancel", 
                   (10, display_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # DO NOT apply sandbox transformation during projection alignment!
        # We need to see the FULL screen to align the projector properly
        # The pattern is already at display resolution, so just show it directly
        cv2.imshow('AR Sandbox - Contour Lines', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("❌ Projection alignment cancelled")
            return False
        elif key == 13:  # ENTER
            # Scale corners back to original depth resolution and save
            sandbox_corners = [
                (int(corner[0] / scale_x), int(corner[1] / scale_y)) 
                for corner in working_corners
            ]
            print(f"✅ Projection alignment saved: {sandbox_corners}")
            return True
        elif key == 9:  # TAB
            selected_corner = (selected_corner + 1) % 4
            print(f"📍 Selected corner: {corner_labels[selected_corner]}")
        elif key == 256 + 9:  # SHIFT+TAB (approximate)
            selected_corner = (selected_corner - 1) % 4
            print(f"📍 Selected corner: {corner_labels[selected_corner]}")
        elif key == 82:  # UP arrow
            working_corners[selected_corner][1] = max(0, working_corners[selected_corner][1] - 5)
        elif key == 84:  # DOWN arrow
            working_corners[selected_corner][1] = min(display_height, working_corners[selected_corner][1] + 5)
        elif key == 81:  # LEFT arrow
            working_corners[selected_corner][0] = max(0, working_corners[selected_corner][0] - 5)
        elif key == 83:  # RIGHT arrow
            working_corners[selected_corner][0] = min(display_width, working_corners[selected_corner][0] + 5)
    
    return False



def run_unified_calibration():
    """Run complete calibration system (sandbox + depth thresholds + projection alignment)"""
    # Show calibration start screen
    start_screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    cv2.putText(start_screen, "UNIFIED CALIBRATION MODE", 
               (display_width // 2 - 200, display_height // 3),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    cv2.putText(start_screen, "This will guide you through 3 steps:", 
               (display_width // 2 - 180, display_height // 3 + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(start_screen, "1. Sandbox Dimensions - Click 4 corners", 
               (display_width // 2 - 200, display_height // 3 + 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.putText(start_screen, "2. Projection Alignment - Align with physical sandbox", 
               (display_width // 2 - 250, display_height // 3 + 160),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.putText(start_screen, "3. Depth Thresholds - Set color boundaries", 
               (display_width // 2 - 220, display_height // 3 + 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.putText(start_screen, "Press any key to begin...", 
               (display_width // 2 - 120, display_height // 2 + 120),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    cv2.imshow('AR Sandbox - Contour Lines', start_screen)
    cv2.waitKey(0)
    
    # Step 1: Sandbox dimension calibration
    if not calibrate_sandbox():
        print("⚠️  Sandbox calibration skipped")
        return False
    
    # Step 2: Projection alignment
    if not calibrate_projection_alignment():
        print("⚠️  Projection alignment skipped")
    
    # Step 3: Depth threshold calibration
    calibrate_kinect()
    
    # Save calibration
    save_calibration()
    
    # Show completion screen
    complete_screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    cv2.putText(complete_screen, "CALIBRATION COMPLETE!", 
               (display_width // 2 - 180, display_height // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.putText(complete_screen, "Your sandbox is ready to use", 
               (display_width // 2 - 160, display_height // 2 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(complete_screen, "Press any key to continue...", 
               (display_width // 2 - 140, display_height // 2 + 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    cv2.imshow('AR Sandbox - Contour Lines', complete_screen)
    cv2.waitKey(0)
    
    return True

def create_ar_overlay(depth_data):
    """Create complete AR overlay with contours and colors"""
    # Generate components
    contours = process_depth_to_contours(depth_data)
    colors = create_elevation_colors(depth_data)
    
    # Combine colors and contours
    overlay = cv2.addWeighted(colors, 0.7, cv2.cvtColor(contours, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    # Apply sandbox transformation if calibrated
    overlay = apply_sandbox_transformation(overlay)
    
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
    print("  Press 'm' to enter full calibration mode")
    print("  Press 'a' for quick projection alignment")
    print("  Press 'r' to rotate projection (90° increments)")
    print(f"  Auto-detected resolution: {display_width}x{display_height}")
    
    if not KINECT_AVAILABLE:
        print("⚠️  Running in simulation mode - connect Kinect for real data")
    
    # Detect display resolution
    detect_display_resolution()
    
    # Load calibration on startup (may override detected resolution)
    load_calibration()
    
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
                contours_img = cv2.cvtColor(process_depth_to_contours(depth_data), cv2.COLOR_GRAY2BGR)
                contours_img = apply_sandbox_transformation(contours_img)
                display_img = contours_img
            else:  # colors
                colors_img = create_elevation_colors(depth_data)
                colors_img = apply_sandbox_transformation(colors_img)
                display_img = colors_img
            
            # Resize for display
            display_img = cv2.resize(display_img, (display_width, display_height))
            
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
                global fullscreen_mode
                fullscreen = not fullscreen
                fullscreen_mode = fullscreen
                cv2.setWindowProperty('AR Sandbox - Contour Lines', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                print(f"🖥️  Switched to {'fullscreen' if fullscreen else 'windowed'} mode")
            elif key == ord('m'):
                print("🔧 Entering calibration mode...")
                run_unified_calibration()
                # Recreate mask after calibration
                if sandbox_corners is not None:
                    depth_data = get_kinect_depth()
                    if depth_data is not None:
                        create_sandbox_mask(sandbox_corners, depth_data.shape)
            elif key == ord('r'):
                global sandbox_rotation
                sandbox_rotation = (sandbox_rotation + 90) % 360
                print(f"🔄 Rotation: {sandbox_rotation}°")
                save_calibration()  # Save rotation change
            elif key == ord('a'):
                print("🎯 Quick projection alignment...")
                if calibrate_projection_alignment():
                    save_calibration()
                    # Recreate mask after alignment
                    if sandbox_corners is not None:
                        depth_data = get_kinect_depth()
                        if depth_data is not None:
                            create_sandbox_mask(sandbox_corners, depth_data.shape)
            
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

                    # Show calibration info with better on-screen instructions
                    # Step indicator
                    cv2.putText(colored, f"DEPTH CALIBRATION - Step {step_idx + 1} of 3", 
                               (width // 2 - 180, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    
                    # Current boundary
                    cv2.putText(colored, boundary_name, 
                               (width // 2 - 100, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Instructions
                    cv2.putText(colored, f"Place surface at {description}", 
                               (width // 2 - 150, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    
                    # Current values
                    cv2.putText(colored, f"Raw depth: {current_depth}", (10, height - 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(colored, f"Threshold: {temp_thresholds[step_idx]}", (10, height - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Status
                    status_text = "VALID" if current_depth > 0 and current_depth < 2047 else "INVALID"
                    status_color = (0, 255, 0) if current_depth > 0 and current_depth < 2047 else (0, 0, 255)
                    cv2.putText(colored, f"Status: {status_text}", (10, height - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Controls
                    cv2.putText(colored, "C: Capture | Arrows: Adjust | Q: Quit", 
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
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
                    
                    cv2.imshow('AR Sandbox - Contour Lines', colored)
                    
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
    
    # Start real-time sandbox (calibration loaded automatically)
    run_realtime_sandbox()