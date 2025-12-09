#!/usr/bin/env python3
"""
Real Kinect v1 AR Sandbox Integration
Connects to actual Kinect hardware for real-time contour line projection
"""

import json
import os
import sys
import time

import cv2
import numpy as np

# Add freenect to path
try:
    import freenect

    KINECT_AVAILABLE = True
    print("✅ Kinect library loaded successfully")
except ImportError:
    KINECT_AVAILABLE = False
    print("❌ Kinect library not available - using simulation mode")

# Global calibration thresholds (raw Kinect range 0-2047)
WHITE_BROWN_THRESHOLD = 400  # Very close to close boundary
BROWN_GREEN_THRESHOLD = 800  # Close to middle boundary
GREEN_BLUE_THRESHOLD = 1200  # Middle to far boundary

# Simplified calibration variables
sandbox_rotation = 0  # Current rotation: 0, 90, 180, 270 degrees
mirror_flip = False  # Vertical flip for mirror-based projection
sensor_scale = 1.0  # Scale factor for sensor view
sensor_offset_x = 0  # X offset in pixels
sensor_offset_y = 0  # Y offset in pixels
mask_corners = None  # 4 corners defining the table boundary for masking
calibration_file = "calibration.json"

# Display resolution variables
display_width = 1280  # Default to 720p
display_height = 720  # Default to 720p
fullscreen_mode = False  # Track current display mode


class Fish:
    def __init__(self, width, height, fish_type="fish", immediate=False):
        self.width = width
        self.height = height
        self.type = fish_type
        self.respawn_timer = 0

        if fish_type == "whale":
            self.size = 5  # Reduced to 20% of 25
            self.color = (255, 200, 200)  # Light Blue/Grey (BGR)
            self.base_speed = 0.2  # Was 0.8

            if immediate:
                self.respawn_timer = 0
                self.respawn()
            else:
                # Start with a random delay so it appears unexpectedly
                self.respawn_timer = np.random.randint(50, 200)
                self.active = False
                self.x = 0
                self.y = 0
                self.vx = 0
                self.vy = 0
        elif fish_type == "mermaid":
            self.size = 8
            self.color = (180, 105, 255)  # Hot Pink/Purple (BGR)
            self.base_speed = 0.5  # Was 2.0

            if immediate:
                self.respawn_timer = 0
                self.respawn()
            else:
                # Rare appearance
                self.respawn_timer = np.random.randint(100, 500)
                self.active = False
                self.x = 0
                self.y = 0
                self.vx = 0
                self.vy = 0
        elif fish_type == "car":
            self.size = 6
            self.color = (0, 0, 200)  # Red (BGR)
            self.base_speed = 0.3

            if immediate:
                self.respawn_timer = 0
                self.respawn()
            else:
                self.respawn_timer = np.random.randint(50, 150)
                self.active = False
                self.x = 0
                self.y = 0
                self.vx = 0
                self.vy = 0
        elif fish_type == "dolphin":
            self.size = 7
            self.color = (180, 180, 180)  # Gray (BGR)
            self.base_speed = 0.6

            if immediate:
                self.respawn_timer = 0
                self.respawn()
            else:
                self.respawn_timer = np.random.randint(50, 200)
                self.active = False
                self.x = 0
                self.y = 0
                self.vx = 0
                self.vy = 0
        else:  # 'fish'
            self.size = 2  # Even smaller fish
            self.color = (0, 165, 255)  # Orange
            self.base_speed = 0.375  # Was 1.5
            self.active = False
            self.respawn()

    def respawn(self):
        self.x = np.random.randint(0, self.width)
        self.y = np.random.randint(0, self.height)
        self.vx = np.random.uniform(-self.base_speed, self.base_speed)
        self.vy = np.random.uniform(-self.base_speed, self.base_speed)
        self.active = False

    def update(self, depth_data, water_threshold):
        # Handle respawn timer (for whale/mermaid/car)
        if self.respawn_timer > 0:
            self.respawn_timer -= 1
            if self.respawn_timer == 0:
                self.respawn()
            return

        # Proposed new position
        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Check boundaries
        hit_boundary = False
        if new_x < 0 or new_x >= self.width:
            hit_boundary = True
            self.vx *= -1
            new_x = max(0, min(new_x, self.width - 1))
        if new_y < 0 or new_y >= self.height:
            hit_boundary = True
            self.vy *= -1
            new_y = max(0, min(new_y, self.height - 1))

        # Check if new position is valid terrain
        ix, iy = int(new_x), int(new_y)
        ix = max(0, min(ix, self.width - 1))
        iy = max(0, min(iy, self.height - 1))

        current_depth = depth_data[iy, ix]

        # Different terrain requirements for different entity types
        if self.type == "car":
            # Cars drive on brown and green (mid-level terrain)
            is_valid = (
                current_depth >= WHITE_BROWN_THRESHOLD
                and current_depth < GREEN_BLUE_THRESHOLD
            )
        else:
            # Fish, whales, mermaids swim in water (blue/far areas)
            is_valid = current_depth >= water_threshold and current_depth < 2047

        if is_valid:
            self.active = True
            self.x = new_x
            self.y = new_y

            # Movement logic
            if self.type == "whale":
                # Whale moves smoothly, rarely changes direction
                if np.random.random() < 0.005:
                    self.vx += np.random.uniform(-0.05, 0.05)
                    self.vy += np.random.uniform(-0.05, 0.05)
            elif self.type == "mermaid":
                # Mermaid moves fast and graceful
                if np.random.random() < 0.05:
                    self.vx += np.random.uniform(-0.125, 0.125)
                    self.vy += np.random.uniform(-0.125, 0.125)
            elif self.type == "car":
                # Cars move straighter, occasional small adjustments
                if np.random.random() < 0.02:
                    self.vx += np.random.uniform(-0.1, 0.1)
                    self.vy += np.random.uniform(-0.1, 0.1)
            elif self.type == "dolphin":
                # Dolphins are fast and agile
                if np.random.random() < 0.08:
                    self.vx += np.random.uniform(-0.2, 0.2)
                    self.vy += np.random.uniform(-0.2, 0.2)
            else:
                # Fish move erratically
                if np.random.random() < 0.2:
                    self.vx += np.random.uniform(-0.25, 0.25)
                    self.vy += np.random.uniform(-0.25, 0.25)

            # Cap velocity
            speed = np.sqrt(self.vx**2 + self.vy**2)
            if self.type == "whale":
                max_speed = 0.375
            elif self.type == "mermaid":
                max_speed = 0.75
            elif self.type == "car":
                max_speed = 0.5
            elif self.type == "dolphin":
                max_speed = 0.9  # Fast!
            else:
                max_speed = 0.625

            min_speed = 0.125

            if speed > max_speed:
                scale = max_speed / speed
                self.vx *= scale
                self.vy *= scale
            elif speed < min_speed:
                if speed > 0:
                    scale = min_speed / speed
                    self.vx *= scale
                    self.vy *= scale

        else:
            # Hit land or boundaries of water
            if self.active:
                # Bounce back (for all types now)
                self.vx *= -1
                self.vy *= -1
                # Move slightly away from collision
                self.x += self.vx
                self.y += self.vy
            else:
                # Respawn if stuck on land (trying to spawn)
                self.respawn()

    def draw(self, img):
        if not self.active:
            return

        pt1 = (int(self.x), int(self.y))

        if self.type == "car":
            # Draw Car: rectangle body with wheels
            if self.vx != 0 or self.vy != 0:
                angle = np.arctan2(self.vy, self.vx)
            else:
                angle = 0

            # Car body (rectangle)
            length = self.size * 2
            width = self.size

            # Calculate corners of car body
            front_x = int(self.x + np.cos(angle) * length / 2)
            front_y = int(self.y + np.sin(angle) * length / 2)
            back_x = int(self.x - np.cos(angle) * length / 2)
            back_y = int(self.y - np.sin(angle) * length / 2)

            # Perpendicular for width
            perp_angle = angle + np.pi / 2
            w1_x = int(np.cos(perp_angle) * width / 2)
            w1_y = int(np.sin(perp_angle) * width / 2)

            # Four corners
            corners = np.array(
                [
                    [front_x + w1_x, front_y + w1_y],
                    [front_x - w1_x, front_y - w1_y],
                    [back_x - w1_x, back_y - w1_y],
                    [back_x + w1_x, back_y + w1_y],
                ],
                np.int32,
            )

            # Draw car body
            cv2.fillPoly(img, [corners], self.color)

            # Draw wheels (small black circles)
            wheel_size = 2
            wheel_offset = length / 3
            for sign in [-1, 1]:
                wheel_x = int(
                    self.x
                    + sign * np.cos(angle) * wheel_offset
                    + np.cos(perp_angle) * width / 2
                )
                wheel_y = int(
                    self.y
                    + sign * np.sin(angle) * wheel_offset
                    + np.sin(perp_angle) * width / 2
                )
                cv2.circle(img, (wheel_x, wheel_y), wheel_size, (0, 0, 0), -1)

                wheel_x = int(
                    self.x
                    + sign * np.cos(angle) * wheel_offset
                    - np.cos(perp_angle) * width / 2
                )
                wheel_y = int(
                    self.y
                    + sign * np.sin(angle) * wheel_offset
                    - np.sin(perp_angle) * width / 2
                )
                cv2.circle(img, (wheel_x, wheel_y), wheel_size, (0, 0, 0), -1)

        elif self.type == "dolphin":
            # Draw Dolphin: curved body with dorsal fin
            if self.vx != 0 or self.vy != 0:
                angle = np.arctan2(self.vy, self.vx)
            else:
                angle = 0

            # Main body (ellipse)
            axes = (int(self.size * 1.5), int(self.size * 0.8))
            cv2.ellipse(img, pt1, axes, np.degrees(angle), 0, 360, self.color, -1)

            # Dorsal fin (small triangle on top)
            fin_offset_x = int(self.x - np.cos(angle) * self.size * 0.3)
            fin_offset_y = int(self.y - np.sin(angle) * self.size * 0.3)
            perp_angle = angle + np.pi / 2
            fin_tip_x = int(fin_offset_x + np.cos(perp_angle) * self.size * 0.8)
            fin_tip_y = int(fin_offset_y + np.sin(perp_angle) * self.size * 0.8)
            fin_base1_x = int(fin_offset_x + np.cos(angle) * self.size * 0.3)
            fin_base1_y = int(fin_offset_y + np.sin(angle) * self.size * 0.3)
            fin_base2_x = int(fin_offset_x - np.cos(angle) * self.size * 0.3)
            fin_base2_y = int(fin_offset_y - np.sin(angle) * self.size * 0.3)

            fin_pts = np.array(
                [
                    [fin_tip_x, fin_tip_y],
                    [fin_base1_x, fin_base1_y],
                    [fin_base2_x, fin_base2_y],
                ],
                np.int32,
            )
            cv2.fillPoly(img, [fin_pts], self.color)

            # Tail (wider triangle at back)
            tail_center_x = int(self.x - np.cos(angle) * self.size * 1.2)
            tail_center_y = int(self.y - np.sin(angle) * self.size * 1.2)
            tail_width = self.size * 1.0
            t1_x = int(tail_center_x + np.cos(perp_angle) * tail_width)
            t1_y = int(tail_center_y + np.sin(perp_angle) * tail_width)
            t2_x = int(tail_center_x - np.cos(perp_angle) * tail_width)
            t2_y = int(tail_center_y - np.sin(perp_angle) * tail_width)

            tail_pts = np.array(
                [[tail_center_x, tail_center_y], [t1_x, t1_y], [t2_x, t2_y]], np.int32
            )
            cv2.fillPoly(img, [tail_pts], self.color)

        elif self.type == "mermaid":
            # Draw Mermaid: Purple head, Green tail
            # Head
            cv2.circle(img, pt1, self.size // 2, (200, 200, 255), -1)  # Pale face

            # Tail direction
            if self.vx != 0 or self.vy != 0:
                angle = np.arctan2(self.vy, self.vx)

                # Body/Tail
                tail_len = self.size * 2.0
                tail_x = int(self.x - np.cos(angle) * tail_len)
                tail_y = int(self.y - np.sin(angle) * tail_len)

                # Green tail triangle
                tail_width = self.size
                t1_x = int(tail_x + np.cos(angle + np.pi / 2) * tail_width)
                t1_y = int(tail_y + np.sin(angle + np.pi / 2) * tail_width)
                t2_x = int(tail_x + np.cos(angle - np.pi / 2) * tail_width)
                t2_y = int(tail_y + np.sin(angle - np.pi / 2) * tail_width)

                pts = np.array([pt1, (t1_x, t1_y), (t2_x, t2_y)], np.int32)
                cv2.fillPoly(img, [pts], (100, 255, 100))  # Green tail

        else:
            # Draw body (Fish or Whale)
            cv2.circle(img, pt1, self.size, self.color, -1)

            # Draw tail
            if self.vx != 0 or self.vy != 0:
                angle = np.arctan2(self.vy, self.vx)
                tail_len = self.size * 1.5

                tail_x = int(self.x - np.cos(angle) * tail_len)
                tail_y = int(self.y - np.sin(angle) * tail_len)

                tail_width = self.size
                t1_x = int(tail_x + np.cos(angle + np.pi / 2) * tail_width)
                t1_y = int(tail_y + np.sin(angle + np.pi / 2) * tail_width)
                t2_x = int(tail_x + np.cos(angle - np.pi / 2) * tail_width)
                t2_y = int(tail_y + np.sin(angle - np.pi / 2) * tail_width)

                pts = np.array([pt1, (t1_x, t1_y), (t2_x, t2_y)], np.int32)
                cv2.fillPoly(img, [pts], self.color)

                # Whale blowhole effect (simple)
                if self.type == "whale" and np.random.random() < 0.05:
                    cv2.circle(img, pt1, self.size // 3, (255, 255, 255), -1)


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
            (1280, 720),  # 720p (HD)
            (1920, 1080),  # 1080p (Full HD)
            (1024, 768),  # XGA
            (800, 600),  # SVGA
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
    global \
        sandbox_rotation, \
        mirror_flip, \
        sensor_scale, \
        sensor_offset_x, \
        sensor_offset_y, \
        mask_corners
    global WHITE_BROWN_THRESHOLD, BROWN_GREEN_THRESHOLD, GREEN_BLUE_THRESHOLD

    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, "r") as f:
                calib = json.load(f)

            sandbox_rotation = int(calib.get("rotation", 0))
            mirror_flip = bool(calib.get("mirror_flip", False))
            sensor_scale = float(calib.get("sensor_scale", 1.0))
            sensor_offset_x = int(calib.get("sensor_offset_x", 0))
            sensor_offset_y = int(calib.get("sensor_offset_y", 0))

            corners = calib.get("mask_corners", None)
            if corners is not None:
                mask_corners = [(int(x), int(y)) for x, y in corners]
            else:
                mask_corners = None

            thresholds = calib.get("depth_thresholds", {})
            WHITE_BROWN_THRESHOLD = int(thresholds.get("white_brown", 400))
            BROWN_GREEN_THRESHOLD = int(thresholds.get("brown_green", 800))
            GREEN_BLUE_THRESHOLD = int(thresholds.get("green_blue", 1200))

            print(f"✅ Calibration loaded from {calibration_file}")
            print(f"   Rotation: {sandbox_rotation}°")
            print(f"   Mirror flip: {'enabled' if mirror_flip else 'disabled'}")
            print(f"   Sensor scale: {sensor_scale:.2f}")
            print(f"   Sensor offset: ({sensor_offset_x}, {sensor_offset_y})")
            if mask_corners is not None:
                print(f"   Mask corners: {mask_corners}")
            print(
                f"   Depth thresholds: W→B:{WHITE_BROWN_THRESHOLD}, B→G:{BROWN_GREEN_THRESHOLD}, G→L:{GREEN_BLUE_THRESHOLD}"
            )
            return True
        except Exception as e:
            print(f"⚠️  Error loading calibration: {e}")
            return False
    else:
        print("📝 No calibration file found - using defaults")
        return False


def save_calibration():
    """Save current calibration settings to file"""
    global \
        sandbox_rotation, \
        mirror_flip, \
        sensor_scale, \
        sensor_offset_x, \
        sensor_offset_y, \
        mask_corners
    global WHITE_BROWN_THRESHOLD, BROWN_GREEN_THRESHOLD, GREEN_BLUE_THRESHOLD

    # Convert mask corners to list format for JSON
    corners_to_save = None
    if mask_corners is not None:
        corners_to_save = [[int(x), int(y)] for x, y in mask_corners]

    calib = {
        "rotation": int(sandbox_rotation),
        "mirror_flip": bool(mirror_flip),
        "sensor_scale": float(sensor_scale),
        "sensor_offset_x": int(sensor_offset_x),
        "sensor_offset_y": int(sensor_offset_y),
        "mask_corners": corners_to_save,
        "depth_thresholds": {
            "white_brown": int(WHITE_BROWN_THRESHOLD),
            "brown_green": int(BROWN_GREEN_THRESHOLD),
            "green_blue": int(GREEN_BLUE_THRESHOLD),
        },
    }

    try:
        with open(calibration_file, "w") as f:
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
            pass  # Silent failure

    # Return None to indicate failure
    return None


def create_simulated_terrain(width=640, height=480):
    """Create simulated terrain data"""
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 8, height)
    X, Y = np.meshgrid(x, y)

    # Dynamic terrain that changes over time
    t = time.time() * 0.1
    terrain = (
        np.sin(X * 0.5 + t) * np.cos(Y * 0.5) * 50
        + np.exp(-((X - 5) ** 2 + (Y - 4) ** 2) / 10) * 100
        + np.exp(-((X - 2) ** 2 + (Y - 6) ** 2) / 8) * 80
        + np.random.normal(0, 5, (height, width))
    )

    # Normalize to Kinect depth range (0-2047 for Kinect v1)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 2047
    return terrain.astype(np.uint16)


def process_depth_to_contours(depth_data):
    """Convert depth data to contour line visualization"""
    # Convert to 8-bit for processing using simple scaling
    depth_8bit = cv2.convertScaleAbs(depth_data, alpha=255 / 2047)

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
    close_mask = (depth_data >= WHITE_BROWN_THRESHOLD) & (
        depth_data < BROWN_GREEN_THRESHOLD
    )
    colored[close_mask] = [19, 69, 139]  # BGR format for brown

    # Mid areas: green (mid-high raw values)
    mid_mask = (depth_data >= BROWN_GREEN_THRESHOLD) & (
        depth_data < GREEN_BLUE_THRESHOLD
    )
    colored[mid_mask] = [34, 139, 34]  # Green works the same in RGB/BGR

    # Far areas: blue (highest raw values)
    far_mask = (depth_data >= GREEN_BLUE_THRESHOLD) & (depth_data < 2047)
    colored[far_mask] = [200, 100, 0]  # BGR format for blue

    return colored


def create_elevation_colors_with_thresholds(
    depth_data, white_brown_thresh, brown_green_thresh, green_blue_thresh
):
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


def apply_transformation(image):
    """Apply simplified transformation: rotate, flip, scale, translate, and mask"""
    global \
        sandbox_rotation, \
        mirror_flip, \
        sensor_scale, \
        sensor_offset_x, \
        sensor_offset_y, \
        mask_corners

    # Step 1: Apply rotation to sensor image first (before scaling/positioning)
    if sandbox_rotation != 0:
        if sandbox_rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif sandbox_rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif sandbox_rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Step 2: Apply mirror flip to sensor image
    if mirror_flip:
        image = cv2.flip(image, 0)  # Vertical flip

    # Step 3: Scale the image
    height, width = image.shape[:2]
    if sensor_scale != 1.0:
        new_width = int(width * sensor_scale)
        new_height = int(height * sensor_scale)
        image = cv2.resize(image, (new_width, new_height))

    # Step 4: Create canvas at display resolution
    canvas = np.zeros(
        (display_height, display_width, 3)
        if len(image.shape) == 3
        else (display_height, display_width),
        dtype=np.uint8,
    )

    # Step 5: Calculate position with offset (centered by default)
    img_h, img_w = image.shape[:2]
    x_pos = (display_width - img_w) // 2 + sensor_offset_x
    y_pos = (display_height - img_h) // 2 + sensor_offset_y

    # Step 6: Place image on canvas (with bounds checking)
    # Calculate source and destination regions
    src_x1 = max(0, -x_pos)
    src_y1 = max(0, -y_pos)
    src_x2 = min(img_w, display_width - x_pos)
    src_y2 = min(img_h, display_height - y_pos)

    dst_x1 = max(0, x_pos)
    dst_y1 = max(0, y_pos)
    dst_x2 = min(display_width, x_pos + img_w)
    dst_y2 = min(display_height, y_pos + img_h)

    # Copy the visible portion
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    # Step 7: Apply mask if defined
    if mask_corners is not None and len(mask_corners) == 4:
        mask = np.zeros((display_height, display_width), dtype=np.uint8)
        pts = np.array(mask_corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # Apply mask
        if len(canvas.shape) == 3:
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            canvas = cv2.bitwise_and(canvas, mask_3d)
        else:
            canvas = cv2.bitwise_and(canvas, mask)

    return canvas


def calibrate_sensor_alignment():
    """Interactive sensor alignment calibration with real-time preview"""
    global \
        sandbox_rotation, \
        mirror_flip, \
        sensor_scale, \
        sensor_offset_x, \
        sensor_offset_y, \
        mask_corners

    print("🎯 Sensor Alignment Calibration")
    print("Align the sensor view with your physical sandbox")
    print("\nControls:")
    print("  R: Rotate 90° clockwise")
    print("  M: Toggle mirror flip")
    print("  +/-: Scale up/down (0.05 increments)")
    print("  Arrow keys: Move position (10px increments)")
    print("  Shift+Arrow: Fine movement (1px increments)")
    print("  ENTER: Confirm and continue")
    print("  ESC: Cancel")

    # Store original values in case of cancel
    orig_rotation = sandbox_rotation
    orig_mirror = mirror_flip
    orig_scale = sensor_scale
    orig_offset_x = sensor_offset_x
    orig_offset_y = sensor_offset_y
    orig_mask = mask_corners

    # Clear mask during sensor alignment so user can see full view
    mask_corners = None
    print("ℹ️  Mask temporarily cleared for alignment")

    while True:
        # Get live depth data
        depth_data = get_kinect_depth()
        if depth_data is None:
            print("❌ Cannot get depth data")
            return False

        # Create color visualization
        colored = create_elevation_colors(depth_data)

        # Apply current transformation settings
        display = apply_transformation(colored)

        # Add on-screen info
        cv2.putText(
            display,
            "SENSOR ALIGNMENT",
            (display_width // 2 - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

        # Show current settings
        y_offset = 80
        settings_text = [
            f"Rotation: {sandbox_rotation}°",
            f"Mirror: {'ON' if mirror_flip else 'OFF'}",
            f"Scale: {sensor_scale:.2f}x",
            f"Offset: ({sensor_offset_x}, {sensor_offset_y})",
        ]

        for i, text in enumerate(settings_text):
            cv2.putText(
                display,
                text,
                (30, y_offset + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Controls help
        cv2.putText(
            display,
            "R: Rotate | M: Mirror | +/-: Scale | Arrows: Move | ENTER: Confirm | ESC: Cancel",
            (10, display_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("AR Sandbox - Contour Lines", display)

        # Get key input
        key_full = cv2.waitKey(1)
        key = key_full & 0xFF

        if key == 27:  # ESC - Cancel
            # Restore original values
            sandbox_rotation = orig_rotation
            mirror_flip = orig_mirror
            sensor_scale = orig_scale
            sensor_offset_x = orig_offset_x
            sensor_offset_y = orig_offset_y
            mask_corners = orig_mask  # Restore original mask
            print("❌ Sensor alignment cancelled")
            return False

        elif key == ord("s"):  # Skip
            # Restore original values
            sandbox_rotation = orig_rotation
            mirror_flip = orig_mirror
            sensor_scale = orig_scale
            sensor_offset_x = orig_offset_x
            sensor_offset_y = orig_offset_y
            mask_corners = orig_mask  # Restore original mask
            print("⏭️  Skipping sensor alignment (keeping previous values)")
            return True

        elif key == 13:  # ENTER - Confirm
            mask_corners = (
                orig_mask  # Restore mask (will be updated in next step if not skipped)
            )
            print("✅ Sensor alignment confirmed")
            print(f"   Rotation: {sandbox_rotation}°")
            print(f"   Mirror: {'ON' if mirror_flip else 'OFF'}")
            print(f"   Scale: {sensor_scale:.2f}x")
            print(f"   Offset: ({sensor_offset_x}, {sensor_offset_y})")
            return True

        elif key == ord("r") or key == ord("R"):  # Rotate
            sandbox_rotation = (sandbox_rotation + 90) % 360
            print(f"🔄 Rotation: {sandbox_rotation}°")

        elif key == ord("m") or key == ord("M"):  # Mirror
            mirror_flip = not mirror_flip
            print(f"🪞 Mirror: {'ON' if mirror_flip else 'OFF'}")

        elif key == ord("+") or key == ord("="):  # Scale up
            sensor_scale = min(3.0, sensor_scale + 0.05)
            print(f"🔍 Scale: {sensor_scale:.2f}x")

        elif key == ord("-") or key == ord("_"):  # Scale down
            sensor_scale = max(0.1, sensor_scale - 0.05)
            print(f"🔍 Scale: {sensor_scale:.2f}x")

        # Arrow keys for position
        # Check for shift modifier (fine movement)
        shift_pressed = (key_full & 0xFF00) != 0
        step = 1 if shift_pressed else 10

        # Arrow key detection - try multiple methods for cross-platform compatibility
        # UP arrow
        if key_full == 2490368 or key_full == 65362 or key == 82 or key == 0:
            sensor_offset_y -= step
            print(f"⬆️  Offset: ({sensor_offset_x}, {sensor_offset_y})")
        # DOWN arrow
        elif key_full == 2621440 or key_full == 65364 or key == 84 or key == 1:
            sensor_offset_y += step
            print(f"⬇️  Offset: ({sensor_offset_x}, {sensor_offset_y})")
        # LEFT arrow
        elif key_full == 2424832 or key_full == 65361 or key == 81 or key == 2:
            sensor_offset_x -= step
            print(f"⬅️  Offset: ({sensor_offset_x}, {sensor_offset_y})")
        # RIGHT arrow
        elif key_full == 2555904 or key_full == 65363 or key == 83 or key == 3:
            sensor_offset_x += step
            print(f"➡️  Offset: ({sensor_offset_x}, {sensor_offset_y})")
        # Debug: print key codes if not recognized
        elif key != 255 and key != 0:
            print(f"Debug: key={key}, key_full={key_full}")

    return False


def calibrate_mask():
    """Interactive mask calibration - define table boundary"""
    global mask_corners

    print("🎯 Table Mask Calibration")
    print("Click 4 corners to define the table boundary")
    print("Everything outside will be masked (black)")
    print("\nControls:")
    print("  Click: Select corner")
    print("  C: Clear all corners")
    print("  ENTER: Confirm")
    print("  ESC: Cancel")

    # Get live depth data for background
    depth_data = get_kinect_depth()
    if depth_data is None:
        print("❌ Cannot get depth data")
        return False

    # Create color visualization and apply transformation
    colored = create_elevation_colors(depth_data)
    base_display = apply_transformation(colored)

    corners = []
    corner_labels = ["TL", "TR", "BR", "BL"]

    # Mouse callback for corner selection
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            print(f"✅ Corner {len(corners)}: ({x}, {y})")

    cv2.setMouseCallback("AR Sandbox - Contour Lines", mouse_callback)

    while True:
        # Refresh display with live data
        depth_data = get_kinect_depth()
        if depth_data is not None:
            colored = create_elevation_colors(depth_data)
            display = apply_transformation(colored)
        else:
            display = base_display.copy()

        # Add title
        cv2.putText(
            display,
            "TABLE MASK CALIBRATION",
            (display_width // 2 - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

        # Show instructions
        if len(corners) < 4:
            cv2.putText(
                display,
                f"Click corner {len(corners) + 1} of 4",
                (display_width // 2 - 100, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                display,
                "Press ENTER to confirm or ESC to cancel",
                (display_width // 2 - 250, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        # Draw existing corners
        for i, corner in enumerate(corners):
            cv2.circle(display, corner, 10, (0, 255, 255), -1)
            cv2.putText(
                display,
                corner_labels[i],
                (corner[0] + 15, corner[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        # Draw lines between corners
        if len(corners) >= 2:
            for i in range(len(corners)):
                next_i = (i + 1) % len(corners)
                if next_i < len(corners):
                    cv2.line(display, corners[i], corners[next_i], (0, 255, 0), 2)

        # Fill polygon if 4 corners selected (preview)
        if len(corners) == 4:
            overlay = display.copy()
            pts = np.array(corners, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # Controls help
        cv2.putText(
            display,
            "Click: Select | C: Clear | S: Skip | ENTER: Confirm | ESC: Cancel",
            (10, display_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("AR Sandbox - Contour Lines", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("❌ Mask calibration cancelled")
            return False
        elif key == ord("c") or key == ord("C"):  # Clear
            corners = []
            print("🗑️  Corners cleared")
        elif key == ord("s") or key == ord("S"):  # Skip
            print("⏭️  Skipping mask calibration (keeping previous mask)")
            return True
        elif key == 13 and len(corners) == 4:  # ENTER
            mask_corners = corners.copy()
            print(f"✅ Mask corners set: {mask_corners}")
            return True

    return False


def calibrate_projection_alignment():
    """Calibrate projection to align with physical sandbox"""
    global sandbox_corners, projector_corners

    print("🎯 Projection Alignment Calibration")
    print("This ensures your projected overlay matches the physical sandbox")
    print("\nInstructions:")
    print(
        "1. Move the corner markers until they align with your PHYSICAL sandbox corners"
    )
    print("2. The live sensor view will warp in real-time to follow")
    print("3. Use arrow keys to fine-tune each corner")
    print("4. Press TAB to switch between corners")
    print("5. Press ENTER when satisfied with alignment")

    if sandbox_corners is None:
        print(
            "❌ No sandbox corners defined. Please calibrate sandbox dimensions first."
        )
        return False

    # Initialize working corners from existing projector_corners or default to full screen
    if projector_corners is not None:
        working_corners = [list(c) for c in projector_corners]
    else:
        working_corners = [
            [0, 0],
            [display_width - 1, 0],
            [display_width - 1, display_height - 1],
            [0, display_height - 1],
        ]

    corner_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]  # Red, Green, Blue, Yellow
    corner_labels = ["TL", "TR", "BR", "BL"]
    selected_corner = 0

    while True:
        # Get live depth data for background
        depth_data_live = get_kinect_depth()

        if depth_data_live is not None:
            # Create color visualization from live sensor data (Depth Resolution)
            colored_live = create_elevation_colors(depth_data_live)

            # Warp it to the current working_corners (Display Resolution)
            src_points = np.float32(sandbox_corners)
            dst_points = np.float32(working_corners)

            try:
                perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                warped_live = cv2.warpPerspective(
                    colored_live, perspective_matrix, (display_width, display_height)
                )
                display = warped_live
            except Exception as e:
                print(f"Warp error: {e}")
                display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        else:
            display = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Draw corner markers on top of the live sensor view
        for i, corner in enumerate(working_corners):
            color = corner_colors[i]
            label = corner_labels[i]

            # Highlight selected corner
            thickness = 3 if i == selected_corner else 2
            radius = 15 if i == selected_corner else 10

            cv2.circle(display, tuple(corner), radius, color, thickness)
            cv2.putText(
                display,
                label,
                (corner[0] + 20, corner[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Draw connecting lines
            next_i = (i + 1) % 4
            cv2.line(display, tuple(corner), tuple(working_corners[next_i]), color, 2)

        # Add on-screen instructions with semi-transparent background for better readability
        # Main instruction
        cv2.putText(
            display,
            "PROJECTION ALIGNMENT",
            (display_width // 2 - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
        )

        # Mirror warning
        if mirror_flip:
            cv2.putText(
                display,
                "(MIRROR FLIP ACTIVE - Controls Inverted)",
                (display_width // 2 - 200, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Current corner indicator
        cv2.putText(
            display,
            f"Adjusting: {corner_labels[selected_corner]} corner",
            (display_width // 2 - 120, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Corner status
        for i in range(4):
            if i == selected_corner:
                # Current corner
                cv2.putText(
                    display,
                    f"→ {corner_labels[i]}",
                    (30, 140 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                # Other corners
                cv2.putText(
                    display,
                    f"  {corner_labels[i]}",
                    (30, 140 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (150, 150, 150),
                    2,
                )

        # Control instructions
        cv2.putText(
            display,
            "Arrow Keys: Move corner | TAB: Next corner | ENTER: Confirm | ESC: Cancel",
            (10, display_height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("AR Sandbox - Contour Lines", display)

        # Get full key code (not masked) for special keys
        key_full = cv2.waitKey(1)
        key = key_full & 0xFF

        if key == 27:  # ESC
            print("❌ Projection alignment cancelled")
            return False
        elif key == 13:  # ENTER
            # Save corners
            projector_corners = [tuple(c) for c in working_corners]
            print(f"✅ Projection alignment saved: {projector_corners}")
            return True
        elif key == 9:  # TAB
            selected_corner = (selected_corner + 1) % 4
        elif key == 256 + 9:  # SHIFT+TAB (approximate)
            selected_corner = (selected_corner - 1) % 4
        # Arrow keys - handle both masked and full key codes for cross-platform support
        # MIRROR AWARE CONTROLS: Invert UP/DOWN if mirror flip is on
        elif (
            key == 82 or key_full == 65362 or key_full == 2490368 or key == 0
        ):  # UP arrow
            step = -5 if not mirror_flip else 5  # Invert if mirrored
            working_corners[selected_corner][1] = max(
                0, min(display_height, working_corners[selected_corner][1] + step)
            )
        elif (
            key == 84 or key_full == 65364 or key_full == 2621440 or key == 1
        ):  # DOWN arrow
            step = 5 if not mirror_flip else -5  # Invert if mirrored
            working_corners[selected_corner][1] = max(
                0, min(display_height, working_corners[selected_corner][1] + step)
            )
        elif (
            key == 81 or key_full == 65361 or key_full == 2424832 or key == 2
        ):  # LEFT arrow
            working_corners[selected_corner][0] = max(
                0, working_corners[selected_corner][0] - 5
            )
        elif (
            key == 83 or key_full == 65363 or key_full == 2555904 or key == 3
        ):  # RIGHT arrow
            working_corners[selected_corner][0] = min(
                display_width, working_corners[selected_corner][0] + 5
            )

    return False


def run_unified_calibration():
    """Run complete simplified calibration system"""
    # Show calibration start screen
    start_screen = np.zeros((display_height, display_width, 3), dtype=np.uint8)

    cv2.putText(
        start_screen,
        "SIMPLIFIED CALIBRATION MODE",
        (display_width // 2 - 250, display_height // 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3,
    )

    cv2.putText(
        start_screen,
        "This will guide you through 3 steps:",
        (display_width // 2 - 180, display_height // 3 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        start_screen,
        "1. Sensor Alignment - Rotate, mirror, scale, position",
        (display_width // 2 - 250, display_height // 3 + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )

    cv2.putText(
        start_screen,
        "2. Table Mask - Define projection boundary",
        (display_width // 2 - 220, display_height // 3 + 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )

    cv2.putText(
        start_screen,
        "3. Depth Thresholds - Set color boundaries",
        (display_width // 2 - 220, display_height // 3 + 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )

    cv2.putText(
        start_screen,
        "Press 'S' to skip any step during calibration",
        (display_width // 2 - 220, display_height // 3 + 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        start_screen,
        "Press any key to begin...",
        (display_width // 2 - 120, display_height // 2 + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow("AR Sandbox - Contour Lines", start_screen)
    cv2.waitKey(0)

    # Step 1: Sensor alignment
    if not calibrate_sensor_alignment():
        print("⚠️  Sensor alignment skipped")
        return False

    # Step 2: Table mask
    if not calibrate_mask():
        print("⚠️  Table mask skipped")

    # Step 3: Depth threshold calibration
    calibrate_kinect()

    # Save calibration
    save_calibration()

    # Don't show completion screen - just return to live view
    print("✅ Calibration complete! Returning to live view...")

    return True


def create_ar_overlay(depth_data, fishes=None):
    """Create complete AR overlay with contours and colors"""
    # Generate components
    contours = process_depth_to_contours(depth_data)
    colors = create_elevation_colors(depth_data)

    # Combine colors and contours
    overlay = cv2.addWeighted(
        colors, 0.7, cv2.cvtColor(contours, cv2.COLOR_GRAY2BGR), 0.3, 0
    )

    # Draw fishes if provided (before transformation so they align with terrain)
    if fishes:
        for fish in fishes:
            fish.draw(overlay)

    # Apply transformation
    overlay = apply_transformation(overlay)

    return overlay


def run_realtime_sandbox():
    """Run real-time AR sandbox with Kinect"""
    global mirror_flip, sandbox_rotation, fullscreen_mode

    print("🚀 Starting AR Sandbox...")
    print("Controls:")
    print("  Press 'q' to quit")
    print("  Press 's' to save current frame")
    print("  Press 'c' to toggle contours only")
    print("  Press 'e' to toggle elevation colors only")
    print("  Press 'd' to toggle debug info")
    print("  Press 'f' to toggle fullscreen")
    print("  Press 'v' to toggle mirror flip (for mirror projection)")
    print("  Press 'm' to enter full calibration mode")
    print("  Press 'r' to rotate projection (90° increments)")
    print("  Press '1' to spawn a Whale 🐋")
    print("  Press '2' to spawn a Mermaid 🧜‍♀️")
    print("  Press '3' to spawn a Car 🚗")
    print("  Press '4' to spawn a Dolphin 🐬")
    print(f"  Auto-detected resolution: {display_width}x{display_height}")

    if not KINECT_AVAILABLE:
        print("⚠️  Running in simulation mode - connect Kinect for real data")

    # Detect display resolution
    detect_display_resolution()

    # Load calibration on startup (may override detected resolution)
    load_calibration()

    # Display mode
    mode = "combined"  # 'combined', 'contours', 'colors'
    fullscreen = False
    debug_mode = False

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    # Create window
    cv2.namedWindow("AR Sandbox - Contour Lines", cv2.WINDOW_NORMAL)

    # Kinect retry flag
    kinect_should_try = True

    # Initialize fishes
    # We need depth dimensions, usually 640x480 for Kinect v1
    # We'll initialize them and they will find water or respawn
    # Initialize fishes, one whale, one mermaid, some cars, and some dolphins
    fishes = [Fish(640, 480, fish_type="fish") for _ in range(15)]
    fishes.append(Fish(640, 480, fish_type="whale"))
    fishes.append(Fish(640, 480, fish_type="mermaid"))
    # Add 3 cars
    for _ in range(3):
        fishes.append(Fish(640, 480, fish_type="car"))
    # Add 4 dolphins
    for _ in range(4):
        fishes.append(Fish(640, 480, fish_type="dolphin"))

    try:
        while True:
            # Get depth data
            if kinect_should_try:
                depth_data = get_kinect_depth()
                kinect_failed = depth_data is None
                if kinect_failed:
                    depth_data = create_simulated_terrain()
                    kinect_should_try = False  # Stop trying until user retries
                # If successful, keep kinect_should_try = True to continue using Kinect
            else:
                depth_data = create_simulated_terrain()
                kinect_failed = True

            # Update fishes
            if mode == "combined" or mode == "colors":
                for fish in fishes:
                    fish.update(depth_data, GREEN_BLUE_THRESHOLD)

            # Create AR overlay
            if mode == "combined":
                display_img = create_ar_overlay(depth_data, fishes)
            elif mode == "contours":
                contours_img = cv2.cvtColor(
                    process_depth_to_contours(depth_data), cv2.COLOR_GRAY2BGR
                )
                contours_img = apply_transformation(contours_img)
                display_img = contours_img
            else:  # colors
                colors_img = create_elevation_colors(depth_data)
                colors_img = apply_transformation(colors_img)
                display_img = colors_img

            # Resize for display
            display_img = cv2.resize(display_img, (display_width, display_height))

            # Add error overlay if Kinect failed
            if kinect_failed:
                cv2.putText(
                    display_img,
                    "Kinect not connected - Press 'k' to retry",
                    (10, display_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Calculate FPS
            if frame_count > 0:
                fps = frame_count / (time.time() - start_time + 0.001)

            # Debug info and HUD
            if debug_mode:
                # FPS
                cv2.putText(
                    display_img,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                # Mode
                mode_text = (
                    f"Mode: {mode} | {'Kinect' if not kinect_failed else 'Simulation'}"
                )
                cv2.putText(
                    display_img,
                    mode_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Mirror status
                if mirror_flip:
                    cv2.putText(
                        display_img,
                        "Mirror: ON",
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                # Get center depth value
                height, width = depth_data.shape
                center_x, center_y = width // 2, height // 2
                center_depth = depth_data[center_y, center_x]

                # Determine color band
                if center_depth > 0 and center_depth < WHITE_BROWN_THRESHOLD:
                    color_band = "White"
                elif (
                    center_depth >= WHITE_BROWN_THRESHOLD
                    and center_depth < BROWN_GREEN_THRESHOLD
                ):
                    color_band = "Brown"
                elif (
                    center_depth >= BROWN_GREEN_THRESHOLD
                    and center_depth < GREEN_BLUE_THRESHOLD
                ):
                    color_band = "Green"
                elif center_depth >= GREEN_BLUE_THRESHOLD and center_depth < 2047:
                    color_band = "Blue"
                else:
                    color_band = "Invalid"

                # Debug text - adjust position if mirror is on
                debug_y_start = 140 if mirror_flip else 100
                debug_text1 = f"Center Depth: {center_depth}"
                debug_text2 = f"Color Band: {color_band}"
                debug_text3 = f"Thresholds: W:{WHITE_BROWN_THRESHOLD} B:{BROWN_GREEN_THRESHOLD} G:{GREEN_BLUE_THRESHOLD}"

                cv2.putText(
                    display_img,
                    debug_text1,
                    (10, debug_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_img,
                    debug_text2,
                    (10, debug_y_start + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display_img,
                    debug_text3,
                    (10, debug_y_start + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

                # Creature counts
                fish_count = sum(1 for f in fishes if f.type == "fish")
                whale_count = sum(1 for f in fishes if f.type == "whale")
                mermaid_count = sum(1 for f in fishes if f.type == "mermaid")
                car_count = sum(1 for f in fishes if f.type == "car")
                dolphin_count = sum(1 for f in fishes if f.type == "dolphin")
                active_fish = sum(1 for f in fishes if f.type == "fish" and f.active)
                active_whale = sum(1 for f in fishes if f.type == "whale" and f.active)
                active_mermaid = sum(
                    1 for f in fishes if f.type == "mermaid" and f.active
                )
                active_car = sum(1 for f in fishes if f.type == "car" and f.active)
                active_dolphin = sum(
                    1 for f in fishes if f.type == "dolphin" and f.active
                )

                creature_text = f"Creatures (Active/Total): Fish:{active_fish}/{fish_count} Whales:{active_whale}/{whale_count} Mermaids:{active_mermaid}/{mermaid_count} Cars:{active_car}/{car_count} Dolphins:{active_dolphin}/{dolphin_count}"
                cv2.putText(
                    display_img,
                    creature_text,
                    (10, debug_y_start + 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 200, 200),
                    2,
                )

                # Draw crosshair at center
                cv2.drawMarker(
                    display_img,
                    (center_x, center_y),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

                # Show depth value at center
                cv2.putText(
                    display_img,
                    f"{center_depth}",
                    (center_x + 25, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # Show image
            cv2.imshow("Fremin", display_img)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"ar_sandbox_capture_{int(time.time())}.png"
                cv2.imwrite(filename, display_img)
                print(f"📸 Saved: {filename}")
            elif key == ord("c"):
                mode = "contours" if mode != "contours" else "combined"
                print(f"🎨 Switched to {mode} mode")
            elif key == ord("e"):
                mode = "colors" if mode != "colors" else "combined"
                print(f"🎨 Switched to {mode} mode")
            elif key == ord("d"):
                debug_mode = not debug_mode
                print(f"🔍 Debug mode {'enabled' if debug_mode else 'disabled'}")
            elif key == ord("f"):
                fullscreen = not fullscreen
                fullscreen_mode = fullscreen
                if fullscreen:
                    cv2.setWindowProperty(
                        "AR Sandbox - Contour Lines",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    cv2.setWindowProperty(
                        "AR Sandbox - Contour Lines",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL,
                    )
                print(f"🖥️  Fullscreen {'enabled' if fullscreen else 'disabled'}")
            elif key == ord("m"):
                print("🔧 Entering calibration mode...")
                # Save current fullscreen state
                was_fullscreen = fullscreen
                run_unified_calibration()
                # Restore fullscreen state after calibration
                if was_fullscreen:
                    cv2.setWindowProperty(
                        "AR Sandbox - Contour Lines",
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                    fullscreen = True
                    fullscreen_mode = True
                    print("🖥️  Restored fullscreen mode")
            elif key == ord("r"):
                sandbox_rotation = (sandbox_rotation + 90) % 360
                print(f"🔄 Rotation: {sandbox_rotation}°")
                save_calibration()  # Save rotation change
            elif key == ord("v"):
                mirror_flip = not mirror_flip
                print(f"🪞 Mirror flip {'enabled' if mirror_flip else 'disabled'}")
                save_calibration()  # Save mirror flip change
            elif key == ord("k"):
                kinect_should_try = True  # Allow trying again on next frame
            elif key == ord("1"):
                fishes.append(Fish(640, 480, fish_type="whale", immediate=True))
                print("🐋 Whale spawned!")
            elif key == ord("2"):
                fishes.append(Fish(640, 480, fish_type="mermaid", immediate=True))
                print("🧜‍♀️ Mermaid spawned!")
            elif key == ord("3"):
                fishes.append(Fish(640, 480, fish_type="car", immediate=True))
                print("🚗 Car spawned!")
            elif key == ord("4"):
                fishes.append(Fish(640, 480, fish_type="dolphin", immediate=True))
                print("🐬 Dolphin spawned!")

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
        ("Green → Blue", "middle to far boundary"),
    ]

    captured_depths = []
    temp_thresholds = [
        WHITE_BROWN_THRESHOLD,
        BROWN_GREEN_THRESHOLD,
        GREEN_BLUE_THRESHOLD,
    ]

    try:
        for step_idx, (boundary_name, description) in enumerate(calibration_steps):
            print(f"\n📍 Step {step_idx + 1}: {boundary_name}")
            print(f"   Position surface at {description}")
            print(f"   Press 'c' to capture, 's' to skip, 'q' to quit")
            print(f"   Use arrow keys to adjust threshold in real-time")

            captured = False
            while not captured:
                depth, _ = freenect.sync_get_depth()
                if depth is not None:
                    # Create color visualization with current thresholds
                    colored = create_elevation_colors_with_thresholds(
                        depth,
                        temp_thresholds[0],
                        temp_thresholds[1],
                        temp_thresholds[2],
                    )

                    # Highlight center calibration region ON RAW IMAGE
                    height, width = depth.shape
                    center_x, center_y = width // 2, height // 2
                    region_size = min(width, height) // 8

                    x1 = max(0, center_x - region_size)
                    x2 = min(width, center_x + region_size)
                    y1 = max(0, center_y - region_size)
                    y2 = min(height, center_y + region_size)

                    # Calculate current depth at absolute center point
                    current_depth = depth[center_y, center_x]

                    # Check if depth is invalid
                    is_invalid = current_depth == 0 or current_depth == 2047

                    # Draw rectangle around calibration region (red if invalid, yellow if valid)
                    rect_color = (0, 0, 255) if is_invalid else (255, 255, 0)
                    cv2.rectangle(colored, (x1, y1), (x2, y2), rect_color, 3)

                    # Draw crosshair at absolute center point
                    crosshair_color = (0, 0, 255) if is_invalid else (255, 0, 255)
                    cv2.drawMarker(
                        colored,
                        (center_x, center_y),
                        crosshair_color,
                        cv2.MARKER_CROSS,
                        20,
                        3,
                    )

                    # APPLY TRANSFORMATION
                    display = apply_transformation(colored)
                    disp_h, disp_w = display.shape[:2]

                    # Show calibration info on TRANSFORMED image
                    # Step indicator
                    cv2.putText(
                        display,
                        f"DEPTH CALIBRATION - Step {step_idx + 1} of 3",
                        (disp_w // 2 - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        3,
                    )

                    # Current boundary
                    cv2.putText(
                        display,
                        boundary_name,
                        (disp_w // 2 - 100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                    # Instructions
                    cv2.putText(
                        display,
                        f"Place surface at {description}",
                        (disp_w // 2 - 150, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (200, 200, 200),
                        2,
                    )

                    # Current values
                    cv2.putText(
                        display,
                        f"Raw depth: {current_depth}",
                        (10, disp_h - 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        display,
                        f"Threshold: {temp_thresholds[step_idx]}",
                        (10, disp_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # Status
                    status_text = "VALID" if not is_invalid else "INVALID"
                    status_color = (0, 255, 0) if not is_invalid else (0, 0, 255)
                    cv2.putText(
                        display,
                        f"Status: {status_text}",
                        (10, disp_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2,
                    )

                    # Controls
                    cv2.putText(
                        display,
                        "C: Capture | S: Skip | Arrows: Adjust | Q: Quit",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # Add color sample squares in top right corner
                    square_size = 40
                    margin = 10
                    start_x = disp_w - (square_size + margin) * 4
                    start_y = margin

                    # White square
                    cv2.rectangle(
                        display,
                        (start_x, start_y),
                        (start_x + square_size, start_y + square_size),
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        display,
                        "W",
                        (start_x + 12, start_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                    )

                    # Brown square
                    cv2.rectangle(
                        display,
                        (start_x + (square_size + margin), start_y),
                        (start_x + (square_size + margin) * 2, start_y + square_size),
                        (19, 69, 139),
                        -1,
                    )
                    cv2.putText(
                        display,
                        "B",
                        (start_x + (square_size + margin) + 12, start_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                    # Green square
                    cv2.rectangle(
                        display,
                        (start_x + (square_size + margin) * 2, start_y),
                        (start_x + (square_size + margin) * 3, start_y + square_size),
                        (34, 139, 34),
                        -1,
                    )
                    cv2.putText(
                        display,
                        "G",
                        (start_x + (square_size + margin) * 2 + 12, start_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                    # Blue square
                    cv2.rectangle(
                        display,
                        (start_x + (square_size + margin) * 3, start_y),
                        (start_x + (square_size + margin) * 4, start_y + square_size),
                        (200, 100, 0),
                        -1,
                    )
                    cv2.putText(
                        display,
                        "L",
                        (start_x + (square_size + margin) * 3 + 12, start_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

                    cv2.imshow("AR Sandbox - Contour Lines", display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("c"):
                        # Capture absolute center point depth
                        center_depth = depth[height // 2, width // 2]
                        captured_depths.append(center_depth)
                        print(f"\n✅ Captured center point: Raw={center_depth}")
                        captured = True
                    elif key == ord("s"):
                        print("\n⏭️  Skipping depth calibration")
                        return
                    elif key == ord("q"):
                        print("❌ Calibration cancelled")
                        return
                    elif key == 81:  # Left arrow
                        temp_thresholds[step_idx] = max(
                            0, temp_thresholds[step_idx] - 5
                        )
                        print(f"Threshold: {temp_thresholds[step_idx]}")
                    elif key == 83:  # Right arrow
                        temp_thresholds[step_idx] = min(
                            255, temp_thresholds[step_idx] + 5
                        )
                        print(f"Threshold: {temp_thresholds[step_idx]}")

    except Exception as e:
        print(f"Calibration error: {e}")
        return

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
