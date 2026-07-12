"""
vision_executor.py — Real Vision AI Target Tracking & Visual Servoing

Connects to the FPV camera stream (via GStreamer UDP port 5600) and uses OpenCV
to perform real-time target tracking (color-based for RAM efficiency).
It issues continuous MAVSDK velocity commands (Visual Servoing) to keep the target centered.
"""

import asyncio
import cv2
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError
from ..utils import AuditLog, print_info, print_success, print_error, print_warning, setup_logger

logger = setup_logger("vision_executor")

# Image dimensions from fpv_cam.sdf
WIDTH = 320
HEIGHT = 240

# HSV Color ranges
COLOR_RANGES = {
    "red_box": (np.array([0, 100, 100]), np.array([10, 255, 255])),
    "blue_box": (np.array([100, 100, 100]), np.array([140, 255, 255])),
    "green_box": (np.array([40, 100, 100]), np.array([80, 255, 255])),
}

async def execute_vision_mission(mission: dict, drone: System, audit: AuditLog, telemetry_state: dict):
    print_info("Starting Real Vision AI Mission...")
    target_class = mission.get("target_class", "red_box").lower()
    audit.record("vision_start", {"target": target_class})
    
    # Wait for GPS fix
    print_info("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
            print_success("Global position estimate OK")
            break
            
    # 1. Takeoff
    print_info("Arming and taking off to search altitude...")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(10.0)
    await drone.action.takeoff()
    
    # Wait for takeoff
    for _ in range(20):
        async for pos in drone.telemetry.position():
            if pos.relative_altitude_m >= 9.0:
                break
        await asyncio.sleep(1)
    print_success("Takeoff complete. Search altitude reached.")
    
    # 2. Start Video Capture
    # GStreamer pipeline to receive UDP stream from Gazebo GstCameraPlugin
    gst_str = f"udpsrc port=5600 ! application/x-rtp, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink"
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print_warning("Could not open GStreamer stream directly. Trying fallback UDP stream...")
        cap = cv2.VideoCapture("udp://127.0.0.1:5600")
        
    lower_hsv, upper_hsv = COLOR_RANGES.get(target_class, COLOR_RANGES["red_box"])
    
    # Enter search pattern/offboard control
    # Set initial zero velocity
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        if "already active" not in str(e).lower():
            raise e
            
    print_info("Offboard mode active. Scanning for target...")
    
    tracking_active = False
    search_yaw_speed = 10.0 # deg/s
    
    try:
        # Loop for 60 seconds of search/tracking
        for _ in range(120): # 120 * 0.5s = 60s
            ret, frame = False, None
            if cap.isOpened():
                ret, frame = cap.read()
                
            cx, cy = None, None
            if ret and frame is not None:
                # Resize to standard width/height
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                moments = cv2.moments(mask)
                
                if moments["m00"] > 500: # Target detected (area threshold)
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
            # Control Logic
            if cx is not None and cy is not None:
                if not tracking_active:
                    print_success(f"🎯 TARGET DETECTED: {target_class} at x={cx}, y={cy}! Initiating Visual Servoing...")
                    audit.record("target_acquired", {"x": cx, "y": cy})
                    tracking_active = True
                    
                # Proportional control offsets from center
                err_x = cx - (WIDTH / 2) # Range: -160 to 160
                err_y = cy - (HEIGHT / 2) # Range: -120 to 120
                
                # Gains
                kp_yaw = -0.15 # Turn towards target
                kp_forward = -0.05 # Move forward if target is lower in frame
                
                yaw_speed = err_x * kp_yaw
                forward_vel = err_y * kp_forward
                
                # Clamp outputs for safety
                yaw_speed = np.clip(yaw_speed, -30.0, 30.0) # deg/s
                forward_vel = np.clip(forward_vel, -1.0, 2.0) # m/s
                
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(forward_vel, 0.0, 0.0, yaw_speed))
            else:
                if tracking_active:
                    print_warning("Target lost! Resuming search pattern...")
                    audit.record("target_lost")
                    tracking_active = False
                    
                # Search mode: Slowly spin in place to scan
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, search_yaw_speed))
                
            await asyncio.sleep(0.5)
            
    finally:
        if cap.isOpened():
            cap.release()
            
    print_info("Vision mission duration finished. Landing...")
    audit.record("vision_landing")
    try:
        await drone.offboard.stop()
    except Exception:
        pass
    await drone.action.land()
    print_success("Vision Mission Complete.")
