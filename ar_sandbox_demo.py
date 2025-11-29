#!/usr/bin/env python3
"""
Basic AR Sandbox Demo with Contour Lines
Works with simulated terrain data (no Kinect required)
"""
import numpy as np
import cv2

def normalize_array(arr, target_min=0, target_max=255):
    """Custom normalization function"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr).astype(np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min) * (target_max - target_min) + target_min
    return normalized.astype(np.uint8)

def create_simulated_terrain(width=512, height=424):
    """Create simulated terrain data"""
    print("🏗️  Creating simulated terrain...")
    
    # Create coordinate grids
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 8, height)
    X, Y = np.meshgrid(x, y)
    
    # Create interesting terrain with multiple features
    terrain = (
        np.sin(X * 0.5) * np.cos(Y * 0.5) * 50 +  # Rolling hills
        np.exp(-((X-5)**2 + (Y-4)**2) / 10) * 100 +  # Central peak
        np.exp(-((X-2)**2 + (Y-6)**2) / 8) * 80 +    # Secondary peak
        np.random.normal(0, 5, (height, width))        # Noise
    )
    
    # Normalize to realistic depth range (0-400mm like Kinect)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 400
    
    print(f"📐 Generated terrain: {terrain.shape}")
    print(f"📏 Elevation range: {terrain.min():.1f} - {terrain.max():.1f} mm")
    
    return terrain

def create_contour_lines(terrain_data, levels=15):
    """Create contour lines for terrain"""
    print("🎨 Generating contour lines...")
    
    # Normalize terrain for contour detection
    normalized = normalize_array(terrain_data, 0, 255)
    
    # Create contour levels
    min_val, max_val = terrain_data.min(), terrain_data.max()
    level_values = np.linspace(min_val, max_val, levels)
    
    # Create blank image for contours
    contour_img = np.zeros_like(normalized)
    
    # Draw contour lines
    for level in level_values:
        # Create binary mask for this level
        level_threshold = (level - min_val) / (max_val - min_val) * 255
        mask = ((normalized >= level_threshold) & 
                (normalized < level_threshold + 255/levels)).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(contour_img, contours, -1, 255, 1)
    
    return contour_img

def create_elevation_colormap(terrain_data):
    """Create elevation-based color mapping"""
    print("🌈 Creating elevation color map...")
    
    # Normalize terrain to 0-255 range
    normalized = normalize_array(terrain_data, 0, 255)
    
    # Apply colormap (similar to matplotlib terrain colormap)
    # Create custom terrain-like colormap
    colored = np.zeros((terrain_data.shape[0], terrain_data.shape[1], 3), dtype=np.uint8)
    
    # Low elevations: blue/green (water/valleys)
    low_mask = normalized < 85
    colored[low_mask] = [0, 100, 200]  # Blue
    
    # Mid elevations: green/brown (hills)
    mid_mask = (normalized >= 85) & (normalized < 170)
    colored[mid_mask] = [34, 139, 34]  # Forest green
    
    # High elevations: brown/white (mountains)
    high_mask = normalized >= 170
    colored[high_mask] = [139, 69, 19]  # Brown
    
    # Add some variation based on actual elevation
    for i in range(3):
        colored[:,:,i] = cv2.addWeighted(colored[:,:,i], 0.7, normalized, 0.3, 0)
    
    return colored

def create_combined_visualization(terrain_data, contour_img, colored_terrain):
    """Combine elevation colors with contour lines"""
    print("🖼️  Creating combined visualization...")
    
    # Ensure contour_img has 3 channels
    if len(contour_img.shape) == 2:
        contour_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
    
    # Make contour lines white for better visibility
    contour_lines = np.zeros_like(colored_terrain)
    contour_lines[contour_img[:,:,0] > 0] = [255, 255, 255]  # White contours
    
    # Blend colored terrain with contour lines
    combined = cv2.addWeighted(colored_terrain, 0.8, contour_lines, 0.2, 0)
    
    return combined

def save_visualizations(terrain_data, contour_img, colored_terrain, combined):
    """Save all visualization components"""
    print("💾 Saving visualizations...")
    
    # Save raw terrain data (normalized for visualization)
    terrain_normalized = normalize_array(terrain_data, 0, 255)
    cv2.imwrite('terrain_raw.png', terrain_normalized)
    
    # Save contour lines
    cv2.imwrite('terrain_contours.png', contour_img)
    
    # Save colored elevation map
    cv2.imwrite('terrain_colored.png', colored_terrain)
    
    # Save combined visualization
    cv2.imwrite('terrain_combined.png', combined)
    
    print("✅ Files saved:")
    print("   - terrain_raw.png (raw elevation data)")
    print("   - terrain_contours.png (contour lines only)")
    print("   - terrain_colored.png (elevation colors)")
    print("   - terrain_combined.png (final AR overlay)")

def create_simple_demo():
    """Create simple demo without external dependencies"""
    print("🎯 AR Sandbox Demo - Contour Line Visualization")
    print("=" * 50)
    
    # Create simulated terrain
    terrain_data = create_simulated_terrain()
    
    # Create visualizations
    contour_img = create_contour_lines(terrain_data, levels=20)
    colored_terrain = create_elevation_colormap(terrain_data)
    combined = create_combined_visualization(terrain_data, contour_img, colored_terrain)
    
    # Save visualizations
    save_visualizations(terrain_data, contour_img, colored_terrain, combined)
    
    print("\n✨ Demo complete! Your AR sandbox software is ready.")
    print("📋 Next steps:")
    print("   1. Connect your Kinect v1 device")
    print("   2. Run calibration procedures")
    print("   3. Set up your projector")
    print("   4. Start the full AR sandbox system")
    
    print("\n📂 Check the generated PNG files to see the output!")
    print("🔧 To use with real Kinect, modify the code to use freenect library")

def test_kinect_simulation():
    """Test function to simulate Kinect data flow"""
    print("\n🧪 Testing Kinect simulation...")
    
    # Simulate depth data that would come from Kinect
    simulated_depth = create_simulated_terrain()
    
    # Simulate processing pipeline
    print("📡 Simulating Kinect data capture...")
    print("🔄 Simulating depth processing...")
    print("🎨 Simulating contour generation...")
    print("🖼️  Simulating projection overlay...")
    
    # Create final visualization
    contour_img = create_contour_lines(simulated_depth)
    colored_terrain = create_elevation_colormap(simulated_depth)
    final_overlay = create_combined_visualization(simulated_depth, contour_img, colored_terrain)
    
    cv2.imwrite('kinect_simulation_output.png', final_overlay)
    print("✅ Kinect simulation complete! Output saved as 'kinect_simulation_output.png'")

if __name__ == "__main__":
    create_simple_demo()
    test_kinect_simulation()