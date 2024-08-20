import os
import rclpy
import gymnasium as gym
import numpy as np
from rclpy.node import Node
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.envs.registration import register
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from rl_racing.gazebo_env import GazeboEnv

class TrainingNode(Node):
    def __init__(self):
        super().__init__('RL_RACE_Training', allow_undeclared_parameters=True, 
                            automatically_declare_parameters_from_overrides=True)

def main(args=None):
    
    rclpy.init()
    node = TrainingNode()
    node.get_logger().info("Training node has been created")
    
    home_dir = os.path.expanduser('~')
    pkg_dir = 'ros2_ws/src/rl_racing'
    trained_model_dir = os.path.join(home_dir, pkg_dir, 'rl_models')
    log_dir = os.path.join(home_dir, pkg_dir, 'logs')
    
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    register(
    id='RacingBotEnv-v0',
    entry_point='rl_racing.gazebo_env:GazeboEnv',
    max_episode_steps=1000,
    )
    
    node.get_logger().info("Environment registered...")
        
    env = gym.make('RacingBotEnv-v0')
    env = Monitor(env)
    
    check_env(env)
    node.get_logger().info("Environmental check finished")
    
    # Define the action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
    
    # Initialize the TD3 model with action noise
    model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, 
                learning_rate=0.00001, action_noise=action_noise)
    
    total_timesteps = 40000000
    eval_freq = 100000  # Evaluate the model every 100k steps
    best_mean_reward = -float('inf')
    
    for step in range(0, total_timesteps, eval_freq):
        node.get_logger().info(f"Learning from timestep {step} to {step + eval_freq}")
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        
        node.get_logger().info(f"Evaluating the model at timestep {step + eval_freq}")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        
        node.get_logger().info(f"Mean reward: {mean_reward} +/- {std_reward}")
        
        # Save the best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            node.get_logger().info(f"New best mean reward: {best_mean_reward}. Saving model...")
            model.save(f"{trained_model_dir}/best_model")
        
        # Optional: Implement early stopping based on a reward threshold
        if mean_reward >= 10000:
            node.get_logger().info(f"Reward threshold reached: {mean_reward}. Stopping training...")
            break

    # Save the final model
    model.save(f"{trained_model_dir}/final_model")
    
    node.get_logger().info("Training Finished, Destroying Node...")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
