# 🎯 Projection Alignment Guide

## How to Ensure Your Projection Overlays with Real World

### **Problem Solved:**
Your projected height map needs to perfectly align with your physical sandbox so that:
- Contour lines match the actual sand surface
- Elevation colors correspond to real sand heights
- Virtual projection appears to be "painted" on the sand

### **Step-by-Step Alignment Process:**

#### **1. Initial Setup**
```bash
python3 real_kinect_sandbox.py
```

#### **2. First-Time Calibration**
Press **'m'** to enter full calibration mode:
- **Step 1**: Sandbox dimensions (click 4 corners)
- **Step 2**: Projection alignment (fine-tune corners)
- **Step 3**: Depth thresholds (color boundaries)

#### **3. Quick Realignment (Recommended)**
Press **'a'** for quick projection alignment:
- Shows calibration pattern with corner markers
- Adjust until markers align with physical sandbox corners
- Use arrow keys to move selected corner
- Press TAB to switch between corners

### **Alignment Pattern Features:**
- **🟡 Yellow Crosshairs**: Corner alignment markers
- **🟣 Magenta Circle**: Center reference point
- **📐 Grid Pattern**: Spatial reference
- **⬜ White Border**: Boundary reference

### **Real-World Alignment Tips:**

#### **Physical Setup:**
1. **Projector Position**: Mount projector directly above or at an angle to the sandbox
2. **Keystone Correction**: Use projector's keystone correction if image is distorted
3. **Focus**: Ensure projected image is sharp and in focus
4. **Ambient Light**: Dim room lights for better visibility

#### **Alignment Process:**
1. **Project Pattern**: Press 'a' to show alignment pattern
2. **Match Corners**: Adjust until yellow crosshairs align with your sandbox corners
3. **Check Center**: Verify magenta circle appears at sandbox center
4. **Test with Sand**: Add some sand and verify contour lines follow sand shapes

#### **Fine-Tuning:**
- **Arrow Keys**: Move corners pixel by pixel
- **Large Adjustments**: Use 'm' for full recalibration
- **Rotation**: Press 'r' if projection needs rotation

### **Troubleshooting:**

#### **Pattern Doesn't Fit Sandbox:**
- **Solution**: Press 'm' → Step 1 → Redefine sandbox corners
- **Cause**: Initial corner selection was inaccurate

#### **Projection Appears Distorted:**
- **Solution**: Adjust projector keystone correction
- **Cause**: Projector angle causing trapezoid distortion

#### **Colors Don't Match Sand Height:**
- **Solution**: Press 'm' → Step 3 → Recalibrate depth thresholds
- **Cause**: Kinect sensor angle or lighting conditions changed

### **Verification Methods:**

#### **1. Visual Test:**
- Build a small hill in sand
- Verify contour lines circle the hill
- Check elevation colors make sense (white=peak, blue=valley)

#### **2. Physical Test:**
- Place flat object at known height
- Verify it shows expected color band
- Test multiple height levels

#### **3. Edge Test:**
- Fill sand to exact sandbox edges
- Verify projection stops at sandbox boundaries
- Check no spillage outside sandbox

### **Advanced Tips:**

#### **Multiple Projectors:**
- Calibrate each projector separately
- Use seamless blending for overlapping areas

#### **Different Sandbox Shapes:**
- System works with any 4-corner shape
- Can handle rectangular, square, or trapezoidal sandboxes

#### **Environmental Factors:**
- Recalibrate if room lighting changes
- Adjust if Kinect is moved
- Realign if projector position changes

### **Quick Reference:**
- **'m'**: Full calibration (sandbox → alignment → depths)
- **'a'**: Quick projection alignment
- **'r'**: Rotate projection (90° increments)
- **'d'**: Debug mode (shows depth values)
- **'s'**: Save calibration screenshot

### **Success Indicators:**
✅ Corner markers align with physical sandbox corners  
✅ Contour lines follow sand shapes in real-time  
✅ Elevation colors correspond to actual sand heights  
✅ Projection stays within sandbox boundaries  
✅ No distortion or misalignment visible  

When these conditions are met, your virtual projection will appear to be "painted" directly onto your sand surface!