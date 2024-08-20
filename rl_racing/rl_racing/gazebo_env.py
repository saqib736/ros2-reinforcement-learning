#!/usr/bin/env python3

import rclpy
import numpy as np
from gymnasium import Env
from gymnasium.utils import seeding
from gymnasium.spaces import Dict, Box

from rl_racing.gazebo_connections import GazeboConnection

class GazeboEnv(GazeboConnection, Env):
    def __init__(self):
        super().__init__()
        
        self.get_logger().info("Gazebo connections have been initialized, e.g., topics/services")
        self.seed()
        
        self.episode_num = 0
        self.cumulated_episode_reward = 0
        self.max_linear_vel = 1.0
        self.min_linear_vel = 0.0
        self.angular_vel = 1.0
        self.min_obstacle_dist = 0.2
        
        self.action_space = Box(low=np.array([-1, -1]), 
                                high=np.array([1, 1]),
                                dtype=np.float32)
        
        self.observation_space = Box(low=0, high=1, shape=(60,), dtype=np.float32)
        
        # Initialize the last action for repeat action penalty
        self.last_action = np.zeros(2)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def spin(self):
        self.done_laser_ = False
        self.done_odom_ = False
        while not self.done_laser_ or not self.done_odom_:
            rclpy.spin_once(self) 
    
    def step(self, action):
        # Apply the action to the robot
        self._set_action(action)
        self.spin()
        
        # Get observations, check if the episode is done, and compute the reward
        obs = self._get_obs()
        done = self._is_done()
        info = {}
        
        reward = self._compute_reward(obs, done, action)
        self.cumulated_episode_reward += reward
        
        # Update the last action
        self.last_action = action
        
        return obs, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        self.get_logger().info("Resetting gazebo environment")      
        
        self.resetSim()
        self._update_episode()
        
        self.spin()
        obs = self._get_obs()
        
        self.get_logger().info("End resetting gazebo environment")
        
        # Reset the last action to avoid unfair penalties after reset
        self.last_action = np.zeros(2)
        
        info = {}
        
        return obs, info
    
    def render(self):
        pass
    
    def close(self):
        self.destroy_node()
     
    def _update_episode(self):
        self.get_logger().info("Reward ="+str(self.cumulated_episode_reward)+", EP="+str(self.episode_num))
        
        self.episode_num += 1
        self.cumulated_episode_reward = 0

    def _set_action(self, action):
        # Convert the normalized action values to linear and angular velocities
        action_linear = ((self.max_linear_vel * (action[0] + 1)) +
                         (self.min_linear_vel * (1 - action[0]))) / 2
        action_angular = ((self.angular_vel * (action[1] + 1)) +
                          (-self.angular_vel * (1 - action[1]))) / 2
        
        comb_action = np.array([action_linear, action_angular], dtype=np.float32)
        self.publish_velocity(comb_action)
         
    def _get_obs(self):
        # Retrieve laser scan data and normalize it
        observations = self.get_laser_scan()
        obs = observations / 10  # Scale down the laser readings to a 0-1 range
        
        return obs   
    
    def _is_done(self):
        # Determine if the robot has crashed
        distances = self.get_laser_scan()
        done = any(distances < self.min_obstacle_dist)
        if done:
            self.get_logger().info("Robot Crashed...") 
        
        return done
    
    def _compute_reward(self, obs, done, action):
        # Reward based on forward progress (linear velocity)
        linear_velocity = self.get_current_linear_vel()
        progress_reward = linear_velocity * 5
        
        # Penalty for being close to obstacles (safety penalty)
        min_distance = np.min(obs)
        safety_penalty = -1.0 * (self.min_obstacle_dist - min_distance) if min_distance < self.min_obstacle_dist else 0.0
        
        # Penalty for crashing (done)
        crash_penalty = -20.0 if done else 0.0
        
        # Small reward for staying alive
        alive_reward = 0.1
        
        # Penalty for repeated actions
        if np.array_equal(action, self.last_action):
            repeat_action_penalty = -2.0
        else:
            repeat_action_penalty = 0.0
        
        # Total reward
        reward = progress_reward + safety_penalty + crash_penalty + alive_reward + repeat_action_penalty
        
        return reward
