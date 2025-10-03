# Robot Walking with Reinforcement Learning

Teach a robot to walk using a simple *Q-Learning* agent.  
The robot learns to adjust its stride length and leg force on different terrains by receiving *rewards for staying upright* and *penalties for falling*.

---

## Overview

This repository contains a Python simulation of a walking robot trained with tabular Q-Learning.  
It demonstrates how a reinforcement learning agent can learn gait control policies in a simplified environment.

- *Terrains:* flat, uphill, downhill, rough  
- *Robot states:* balanced, leaning_forward, leaning_backward, unstable  
- *Actions:* 9 combinations of (stride_length, leg_force)  
- *Rewards:* +10 for successful steps, −5 for losing balance, −50 for falling  

Originally implemented as a course project by [Aritra Dey](https://github.com/Aritra-Dey-117-XT) & [Audrija Pal](https://github.com/audrijaishere).

---

## Resources

- [Original Project Report](https://drive.google.com/file/d/1WvNkivJuni1AmiYRXgO9e7_-9C-YLsSb/view?usp=sharing)  
- [Colab Notebook](https://colab.research.google.com/drive/1Fg50cyvBXkqWNxZb5Lq3qBOV4Skl_QA5?usp=sharing)

## How It Works

1. *Environment*  
   - Encapsulates terrain type, robot state, available actions.  
   - Handles state transitions, reward assignment, and random terrain changes.  

2. *Agent (QLearningAgent)*  
   - Maintains a Q-table of shape (16 states × 9 actions).  
   - Uses ε-greedy policy to balance exploration and exploitation.  
   - Updates Q-values using the Q-Learning update rule:  

   `Q[s,a] ← Q[s,a] + α * [r + γ * max_a'(Q[s',a']) − Q[s,a]]`
   

3. *Training Loop*  
   - 1000 episodes by default.  
   - Tracks total reward and steps per episode.  
   - Decays exploration rate (ε) after each episode.  
   - Prints stats & sample Q-values every 10 episodes.  

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Aritra-Dey-117-XT/WalkerAgent.git
cd WalkerAgent
```


### 2. Install dependencies

```bash
pip install numpy
```


### 3. Run training

```bash
python Robot_Walking_using_RL.py
```


You’ll see the agent’s exploration rate, average reward, and average steps improve over time.  
At the end, Q-values for each terrain will be printed to show the learned policy.

---

## Example Output

Episode: 1000/1000<br>
Exploration rate: 0.01<br>
Average reward (last 10): 303.50<br>
Average steps (last 10): 66.80<br>

Q-table sample (Terrain: flat, State: balanced):<br>
Action (stride, force) -> Q-value<br>
------------------------------<br>
(0.1, 0.5) -> 42.46<br>
(0.1, 1.0) -> 41.69<br>
(0.1, 1.5) -> 42.55<br>
(0.3, 0.5) -> 41.27<br>
(0.3, 1.0) -> 55.94<br>
(0.3, 1.5) -> 41.36<br>
(0.5, 0.5) -> 39.93<br>
(0.5, 1.0) -> 42.30<br>
(0.5, 1.5) -> 42.94<br>

## Results

*Training completed!*  
Final exploration rate: 0.01  
Final average reward (last 10): 303.50  
Final average steps (last 10): 66.80  

### Learned Q-values for flat terrain (balanced state)

| Action (stride, force) | Q-value |
|------------------------|---------|
| (0.1, 0.5)             | 42.46   |
| (0.1, 1.0)             | 41.69   |
| (0.1, 1.5)             | 42.55   |
| (0.3, 0.5)             | 41.27   |
| (0.3, 1.0)             | 55.94   |
| (0.3, 1.5)             | 41.36   |
| (0.5, 0.5)             | 39.93   |
| (0.5, 1.0)             | 42.30   |
| (0.5, 1.5)             | 42.94   |

### Learned Q-values for uphill terrain (balanced state)

| Action (stride, force) | Q-value |
|------------------------|---------|
| (0.1, 0.5)             | 19.51   |
| (0.1, 1.0)             | 19.95   |
| (0.1, 1.5)             | 36.30   |
| (0.3, 0.5)             | -48.28  |
| (0.3, 1.0)             | -16.42  |
| (0.3, 1.5)             | 12.31   |
| (0.5, 0.5)             | -48.87  |
| (0.5, 1.0)             | -39.43  |
| (0.5, 1.5)             | 3.45    |

### Learned Q-values for downhill terrain (balanced state)

| Action (stride, force) | Q-value |
|------------------------|---------|
| (0.1, 0.5)             | 6.02    |
| (0.1, 1.0)             | -23.68  |
| (0.1, 1.5)             | -43.92  |
| (0.3, 0.5)             | 0.24    |
| (0.3, 1.0)             | -46.41  |
| (0.3, 1.5)             | -46.77  |
| (0.5, 0.5)             | 22.25   |
| (0.5, 1.0)             | 9.64    |
| (0.5, 1.5)             | 8.65    |

### Learned Q-values for rough terrain (balanced state)

| Action (stride, force) | Q-value |
|------------------------|---------|
| (0.1, 0.5)             | -15.84  |
| (0.1, 1.0)             | -13.05  |
| (0.1, 1.5)             | -13.64  |
| (0.3, 0.5)             | -15.60  |
| (0.3, 1.0)             | 12.19   |
| (0.3, 1.5)             | -15.53  |
| (0.5, 0.5)             | -23.22  |
| (0.5, 1.0)             | -15.44  |
| (0.5, 1.5)             | -20.02  |

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/6b3c4da9-e13e-4f8b-aeb2-982ee8cb0156" />

---

## Limitations

- Simplified discrete environment and action space.  
- No real physics simulation (uses a hand-crafted success probability).  
- No generalization beyond the tabular Q-table.  
- Learning can be slow for larger state/action spaces.  

---

## Future Scope

- Replace Q-table with a Deep Q-Network (DQN) for high-dimensional states.  
- Integrate a physics engine (PyBullet, MuJoCo) for realistic dynamics.  
- Adopt continuous-action algorithms (DDPG, PPO).  
- Add richer sensor feedback (IMU, force sensors).  
- Train on dynamic terrains or multi-robot cooperative scenarios.  
- Transfer learning from simulation to real robots.  

---

## License

MIT License.

---

## Acknowledgements

Project report: “[To help a robot walk using Reinforcement Learning](https://drive.google.com/file/d/1WvNkivJuni1AmiYRXgO9e7_-9C-YLsSb/view?usp=sharing)”<br>
Machine Learning CSC602 — by [Aritra Dey](https://github.com/Aritra-Dey-117-XT) & [Audrija Pal](https://github.com/audrijaishere).
