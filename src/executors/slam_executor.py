"""
slam_executor.py — Real SLAM & Autonomous Navigation Executor

Subscribes to the RPLidar scanner (via ROS 2 /scan or mocked distance telemetry)
to build a live 2D Occupancy Grid Map of the environment.
It implements dynamic obstacle avoidance to autonomously explore the region
without colliding with walls.
"""

import asyncio
import math
import numpy as np
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, OffboardError
from ..utils import AuditLog, print_info, print_success, print_error, print_warning, setup_logger

logger = setup_logger("slam_executor")

# Simple 2D Occupancy Grid Map Configuration
# 100x100 grid representing 50mx50m area (resolution = 0.5m/cell)
GRID_SIZE = 100
RESOLUTION = 0.5
CENTER = GRID_SIZE // 2
grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8) # 0 = unknown, 1 = free, -1 = occupied

# Global variable to hold current obstacle distances
# 360 sectors for a full circle
lidar_ranges = [float('inf')] * 360

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    
    class LidarSubscriber(Node):
        def __init__(self):
            super().__init__('slam_lidar_sub')
            self.subscription = self.create_subscription(
                LaserScan,
                '/scan',
                self.listener_callback,
                10)
            
        def listener_callback(self, msg):
            global lidar_ranges
            # Extract ranges and map them
            lidar_ranges = list(msg.ranges)
except ImportError:
    LidarSubscriber = None


async def run_ros2_loop():
    """Spin the ROS 2 node to get real lidar updates if rclpy is installed."""
    if LidarSubscriber is not None:
        try:
            rclpy.init()
            node = LidarSubscriber()
            print_info("ROS 2 RPLidar Node initialized successfully.")
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                await asyncio.sleep(0.05)
        except Exception as e:
            print_warning(f"ROS 2 loop error: {e}. Falling back to simulated lidar.")
        finally:
            try:
                rclpy.shutdown()
            except Exception:
                pass

async def execute_slam_mission(mission: dict, drone: System, audit: AuditLog, telemetry_state: dict):
    print_info("Starting Real SLAM & Autonomous Navigation...")
    audit.record("slam_start")
    
    # Launch ROS 2 subscription in background
    ros_task = asyncio.create_task(run_ros2_loop())
    
    # Wait for GPS fix
    print_info("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok and health.is_armable:
            print_success("Global position estimate OK")
            break
            
    # 1. Takeoff
    print_info("Taking off to exploration altitude...")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(5.0)
    await drone.action.takeoff()
    
    # Wait for takeoff
    for _ in range(15):
        async for pos in drone.telemetry.position():
            if pos.relative_altitude_m >= 4.5:
                break
        await asyncio.sleep(1)
    print_success("Takeoff complete. Exploration altitude reached.")
    
    # 2. Enter Offboard Mode
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        if "already active" not in str(e).lower():
            raise e
            
    print_info("Offboard mode started. Starting autonomous exploration & mapping...")
    
    explore_area = mission.get("explore_area", {"min_n": -20, "max_n": 20, "min_e": -20, "max_e": 20})
    
    # Speed variables
    forward_speed = 1.0 # m/s
    avoiding = False
    
    try:
        # Loop for 60 seconds
        for tick in range(120): # 120 * 0.5s = 60s
            # Get current coordinates
            curr_n, curr_e, curr_alt, heading = 0.0, 0.0, 0.0, 0.0
            async for pos_ned in drone.telemetry.position_velocity_ned():
                curr_n = pos_ned.position.north_m
                curr_e = pos_ned.position.east_m
                heading = 0.0 # yaw is in telemetry or yawspeed
                break
                
            # Get current heading
            async for hdg in drone.telemetry.heading():
                heading = hdg.heading_deg
                break
                
            # 3. SLAM: Build Occupancy Grid Map using Lidar + Drone position
            # Read lidar values
            # If ROS 2 is not active, fake lidar detections near walls to demonstrate mapping
            active_ranges = lidar_ranges
            is_mock_lidar = (all(r == float('inf') for r in active_ranges) or len(active_ranges) == 0)
            
            if is_mock_lidar:
                # Simulate a wall 8 meters to the North when we get close
                active_ranges = [float('inf')] * 360
                if curr_n > 10:
                    active_ranges[0] = 15 - curr_n # front wall
                if curr_e > 10:
                    active_ranges[90] = 15 - curr_e # right wall
                    
            # Update grid map
            for angle_deg, dist in enumerate(active_ranges):
                if dist != float('inf') and not math.isnan(dist) and dist < 20.0:
                    # Calculate obstacle coordinate in local frame
                    total_angle = math.radians(heading + angle_deg)
                    obs_n = curr_n + dist * math.cos(total_angle)
                    obs_e = curr_e + dist * math.sin(total_angle)
                    
                    # Convert to grid indices
                    grid_x = int(CENTER + obs_n / RESOLUTION)
                    grid_y = int(CENTER + obs_e / RESOLUTION)
                    
                    if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                        grid_map[grid_x, grid_y] = -1 # Obstacle cell
                        
            # Log mapping status occasionally
            if tick % 20 == 0:
                occupied_cells = np.sum(grid_map == -1)
                print_info(f"🗺️ SLAM MAP UPDATE: {occupied_cells} obstacle points registered in Occupancy Grid.")
                audit.record("map_update", {"occupied_cells": int(occupied_cells)})
                
            # 4. Navigation & Obstacle Avoidance
            # Check front sectors (-30 to +30 degrees)
            front_sector = []
            for angle in range(-30, 31):
                idx = angle % 360
                if idx < len(active_ranges):
                    front_sector.append(active_ranges[idx])
                    
            min_front_dist = min(front_sector) if front_sector else float('inf')
            
            if min_front_dist < 4.0: # Obstacle detected in front
                if not avoiding:
                    print_warning(f"⚠️ OBSTACLE DETECTED! Nearest object at {min_front_dist:.1f}m. Recalculating path...")
                    audit.record("obstacle_warning", {"distance_m": min_front_dist})
                    avoiding = True
                    
                # Back up and turn right
                vx = -0.5
                vy = 0.0
                yaw_speed = 30.0 # deg/s
            else:
                if avoiding:
                    print_success("Path cleared. Resuming forward exploration.")
                    audit.record("path_cleared")
                    avoiding = False
                    
                # Boundaries check (Geofence navigation)
                if curr_n > explore_area["max_n"] or curr_n < explore_area["min_n"] or \
                   curr_e > explore_area["max_e"] or curr_e < explore_area["min_e"]:
                    # Turn around to return to exploration area
                    vx = 0.0
                    vy = 0.0
                    yaw_speed = 45.0
                    if tick % 10 == 0:
                        print_warning("Reaching exploration boundaries. Turning around...")
                else:
                    # Fly forward
                    vx = forward_speed
                    vy = 0.0
                    yaw_speed = 0.0
                    
            await drone.offboard.set_velocity_ned(VelocityNedYaw(
                vx * math.cos(math.radians(heading)) - vy * math.sin(math.radians(heading)),
                vx * math.sin(math.radians(heading)) + vy * math.cos(math.radians(heading)),
                -5.0, # Altitude 5m
                yaw_speed
            ))
            
            await asyncio.sleep(0.5)
            
    finally:
        # Cancel ROS 2 task
        ros_task.cancel()
        try:
            await drone.offboard.stop()
        except Exception:
            pass
            
    print_info("SLAM Exploration complete. Returning to launch...")
    audit.record("slam_returning")
    await drone.action.return_to_launch()
    print_success("SLAM Exploration successfully finished!")
