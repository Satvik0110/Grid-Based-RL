# GridWorld Q-Learning Simulation

## Project Overview
This project implements a GridWorld environment where two agents learn optimal navigation strategies using the Q-learning algorithm. The environment features walls, kill zones, terminal states, and a graphical user interface (GUI) for real-time visualization of learning and agent movement.

## Objectives
- Develop a grid-based environment for agents to learn reward-maximizing behaviors.
- Implement the Q-learning algorithm to teach agents optimal actions.
- Provide a GUI to visualize learning processes and the agents' paths.

## Key Features
### 1. GridWorld Environment
- **Dynamic Grid Initialization**: Configurable dimensions with random starting positions.
- **Kill Zones**: Cells penalizing agents upon entry.
- **Terminal States**: Goal cells with associated rewards.
- **Walls**: Impassable barriers restricting agent movements.
- **Epsilon-Greedy Action Selection**: Balances exploration and exploitation.

### 2. Q-Learning Algorithm
- **Action Selection**: Chooses actions based on Q-values.
- **Q-value Updates**: Rewards and maximum future Q-values guide updates.
- **Epsilon Decay**: Reduces exploration over time, favoring exploitation.

### 3. Graphical User Interface (GUI)
- **Grid Visualization**: Displays players, walls, kill zones, and terminal states.
- **User Input**: Configure grid size and episode count.
- **Simulation Control**: Start training and observe real-time agent movements.

### 4. Visualizations
- **Heatmap**: Displays state-action Q-values, highlighting optimal strategies.
- **Performance Graphs**:
  - **Rewards vs. Episodes**: Tracks cumulative reward per episode.
  - **Steps vs. Episodes**: Monitors steps per episode to measure learning efficiency.

## Code Structure
- **Classes**:
  - `GridWorld`: Main environment management class.
  - `GridWorldGUI`: GUI implementation using Tkinter.
- **Key Methods**:
  - `__init__`: Initializes the grid and player positions.
  - `place_killzones`: Sets up penalizing cells.
  - `set_terminal_state`: Configures goal states.
  - `action_e_greedy`: Implements epsilon-greedy strategy.
  - `q_learning_algorithm`: Trains agents over specified episodes.

## How to Run the Project
### Prerequisites
- Python 3.8 or higher
- Required libraries: `numpy`, `tkinter`, and `matplotlib`

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd GridWorld-Q-Learning
   ```
2. Run the simulation after installing dependencies:
   ```bash
   python dc.py
   ```
3. Follow the GUI prompts to configure and start the simulation.

## Demonstrations
1. **Training Visualization**: Observe agents navigating the grid, avoiding penalties, and optimizing movements toward terminal states.
2. **Graphs**:
   - **Cumulative Rewards**: Shows performance improvements over episodes.
   - **Steps per Episode**: Demonstrates increasing efficiency during training.

## Future Enhancements
- **Dynamic Walls and Kill Zones**: Randomize placements for increased challenge.
- **Opponent Interaction**: Incorporate adversarial players for strategic complexity.
- **Additional Algorithms**: Compare Q-learning with other reinforcement learning approaches.

## Conclusion
This project combines algorithmic reinforcement learning with an interactive GUI, providing an engaging way to understand Q-learning principles in a grid-based environment. The simulation demonstrates the effectiveness of Q-learning in teaching agents to optimize navigation strategies.

