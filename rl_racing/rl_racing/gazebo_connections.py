#!/usr/bin/env python3

import time
import rclpy
import numpy as np
from rclpy.node import Node
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class GazeboConnection(Node):
    """
    This class defines all the methods to:
        - Publish actions to the agent (move the robot)
        - Subscribe to sensors of the agent (get laser scans and Imu data)
        - Reset the simulation

    Topics list:
        - /cmd_vel : publish linear and angular velocity of the robot
        - /imu/data : current linear and angular acceleration of the robot
        - /front_laser/scan : laser readings
    
    Services not used:
        - /reset_simulation : resets the gazebo simulation
        - /reset_world : resets the gzebo world
        - /pause_physics : pause the simulation
        - /unpause_physics : unpause the simulation
    """
    
    def __init__(self, max_retry = 10):
        super().__init__('gazebo_connection')
        
        self.max_retry_ = max_retry
        
        # Topics
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imu_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/front_laser/scan', self.laser_callback, 10) 
        
        # Services
        self.unpause_sim = self.create_client(Empty, '/unpause_physics')
        self.pause_sim  = self.create_client(Empty, '/pause_physics')
        self.reset_sim = self.create_client(Empty, '/reset_simulation')
        self.reset_world = self.create_client(Empty, '/reset_world')
    
    def publish_velocity(self, velocity):
        msg = Twist()
        msg.linear.x = float(velocity[0])
        msg.angular.z = float(velocity[1])
        self.action_pub.publish(msg)

    def laser_callback(self, msg: LaserScan):
        self.laser_data_ = np.array(msg.ranges)
        self.laser_data_[self.laser_data_ == np.inf] = np.float32(10)
        self.done_laser_ = True
    
    def get_laser_scan(self):
        return self.laser_data_
        
    def odom_callback(self, msg: Odometry):
        self.current_linear_vel_ = msg.twist.twist.linear.x
        self.done_odom_ = True
    
    def get_current_linear_vel(self): 
        return  self.current_linear_vel_ 
    
    def resetSim(self):
        self.get_logger().info("Resetting simulation")
        
        # Wait for the reset simulation service to become available
        while not self.reset_sim.wait_for_service(self.max_retry_):
            self.get_logger().info("Waiting for reset simulation service...")

        self.get_logger().info("Reset simulation service found!")
        
        # Create and send the reset request
        request = Empty.Request()
        future = self.reset_sim.call_async(request)
        
        # Wait for the service call to complete
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info("Simulation successfully reset!")
            self.resetSim_done_ = True
        else:
            self.get_logger().error(f"Service call failed: {future.exception()}")
    

   