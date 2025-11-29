#!/usr/bin/env python3
"""
Color test script to verify the color definitions
"""
import cv2
import numpy as np

def test_colors():
    """Create a simple image showing all four colors"""
    # Create blank image
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # Define colors as used in the main script (now in BGR format)
    colors = {
        "White": [255, 255, 255],
        "Brown": [19, 69, 139],  # BGR format for brown
        "Green": [34, 139, 34],  # Green works the same in RGB/BGR
        "Blue": [200, 100, 0]    # BGR format for blue
    }
    
    # Draw color squares
    square_size = 80
    x_offset = 20
    y_offset = 60
    
    for i, (name, color) in enumerate(colors.items()):
        x = x_offset + i * (square_size + 20)
        cv2.rectangle(img, (x, y_offset), (x + square_size, y_offset + square_size), color, -1)
        cv2.putText(img, name, (x + 10, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add title
    cv2.putText(img, "Color Test (RGB values as used in code)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the image
    cv2.imshow("Color Test", img)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nColor values used in code:")
    for name, color in colors.items():
        print(f"{name}: {color} (RGB format)")

if __name__ == "__main__":
    test_colors()