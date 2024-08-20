# ROS 2 Reinforcement Learning with Gazebo Simulation

This project demonstrates a simple and effective way to implement reinforcement learning (RL) for robotic tasks using ROS 2 Humble, Gazebo, Stable-Baselines3, and Gymnasium. The project contains two main packages: one for Gazebo simulation (`sim_package`) and another for reinforcement learning scripts (`rl_racing`). The goal is to provide an easy-to-understand framework for learning and applying RL in robotics.

## Project Structure

```bash
src/
┣ rl_racing/
┃ ┣ launch/
┃ ┃ ┗ start_training.launch.py
┃ ┣ rl_racing/
┃ ┃ ┣ __init__.py
┃ ┃ ┣ gazebo_connections.py
┃ ┃ ┣ gazebo_env.py
┃ ┃ ┣ start_learning.py
┃ ┣ package.xml
┃ ┣ setup.cfg
┃ ┗ setup.py
┗ sim_package/
  ┣ launch/
┃ ┃ ┣ pioneer3dx.rviz
┃ ┃ ┣ robot_state_publisher.launch.py
┃ ┃ ┗ start_simulation.launch.py
  ┣ models/
┃ ┃ ┣ racingtrack_logo/
┃ ┃ ┗ td_robot/
  ┣ urdf/
┃ ┃ ┣ meshes/
┃ ┃ ┗ td_robot.urdf
  ┣ worlds/
┃ ┃ ┣ racing_track.world
┃ ┃ ┗ td3.world
  ┣ CMakeLists.txt
  ┗ package.xml
```

## Dependencies

The project relies on the following major dependencies:

- **ROS 2 Humble**: The latest LTS version of ROS 2. [Installation Instructions](https://docs.ros.org/en/humble/Installation.html)
- **Gazebo**: A powerful robot simulation tool.
- **Stable-Baselines3**: A set of reliable implementations of reinforcement learning algorithms. [Installation Instructions](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)
- **Gymnasium**: A toolkit for developing and comparing reinforcement learning algorithms. [Installation Instructions](https://gymnasium.farama.org/index.html)

Ensure that these dependencies are installed and properly configured before running the project.

## Installation

1. Clone this repository into the `src` directory of your ROS 2 workspace:

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/saqib736/ros2-reinforcement-learning.git
   ```

2. Build the workspace using `colcon`:

   ```bash
   cd ~/ros2_ws
   colcon build
   ```

3. Source the workspace:

   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

## Running the Project

### Starting the Simulation

To start the Gazebo simulation, use the following command:

```bash
ros2 launch sim_package start_simulation.launch.py
```

This will launch the Gazebo environment specified in the `racing_track.world` file.

### Starting the Learning Script

To begin training the reinforcement learning model, run:

```bash
ros2 launch rl_racing start_training.launch.py
```

This script utilizes Stable-Baselines3 and Gymnasium to train the model in the simulated environment. The training parameters can be adjusted by editing the `start_learning.py` file located in the `rl_racing` package.

## Customization

### Modifying Training Parameters

You can customize the training process by editing the `start_learning.py` file. This includes adjusting hyperparameters such as learning rate, number of timesteps, and the algorithm used.

### TODO: Running the Trained Agent

A script to run the trained agent is yet to be implemented. This will allow you to deploy the trained model in the Gazebo simulation to observe its performance.

## Contributing

Contributions to this project are welcome! Feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of a learning exercise in applying reinforcement learning to robotic simulations. Special thanks to the open-source community for providing the tools and libraries that made this project possible.
