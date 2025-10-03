import numpy as np
import time
import random
import os

class RobotEnvironment:
    """Simulates a simple robot walking environment"""

    def __init__(self):
        # Environment parameters
        self.terrain_types = ["flat", "uphill", "downhill", "rough"]
        self.robot_states = ["balanced", "leaning_forward", "leaning_backward", "unstable"]

        # Actions: stride_length, leg_force
        self.actions = [
            (0.1, 0.5), (0.1, 1.0), (0.1, 1.5),  # Short stride with varying force
            (0.3, 0.5), (0.3, 1.0), (0.3, 1.5),  # Medium stride with varying force
            (0.5, 0.5), (0.5, 1.0), (0.5, 1.5),  # Long stride with varying force
        ]

        # Current state
        self.current_terrain = random.choice(self.terrain_types)
        self.current_state = "balanced"
        self.steps_taken = 0
        self.max_steps = 100
        self.fallen = False

    def get_state_index(self):
        """Convert current state to an index for Q-table"""
        terrain_idx = self.terrain_types.index(self.current_terrain)
        state_idx = self.robot_states.index(self.current_state)
        return terrain_idx * len(self.robot_states) + state_idx

    def take_action(self, action_idx):
        """Apply the selected action and return reward"""
        stride, force = self.actions[action_idx]

        # Determine outcome based on terrain and action
        success_prob = self._calculate_success_probability(stride, force)
        outcome = random.random() < success_prob

        # Update robot state based on outcome
        if outcome:
            # Successful step
            if self.current_state != "balanced":
                self.current_state = "balanced"  # Robot recovers balance
            reward = 10  # Reward for a successful step
        else:
            # Unsuccessful step
            if self.current_terrain == "flat":
                self.current_state = random.choice(["leaning_forward", "leaning_backward"])
            elif self.current_terrain == "uphill":
                self.current_state = "leaning_backward" if stride < 0.3 else "unstable"
            elif self.current_terrain == "downhill":
                self.current_state = "leaning_forward" if stride > 0.3 else "unstable"
            else:  # rough terrain
                self.current_state = random.choice(["leaning_forward", "leaning_backward", "unstable"])

            if self.current_state == "unstable":
                self.fallen = True
                reward = -50  # Big penalty for falling
            else:
                reward = -5  # Small penalty for losing balance

        # Occasionally change terrain
        if random.random() < 0.1:
            self.current_terrain = random.choice(self.terrain_types)

        self.steps_taken += 1
        done = self.fallen or self.steps_taken >= self.max_steps

        return reward, self.get_state_index(), done

    def _calculate_success_probability(self, stride, force):
        """Calculate probability of successful step based on terrain and action"""
        if self.current_terrain == "flat":
            return 0.9 - abs(stride - 0.3) - abs(force - 1.0)
        elif self.current_terrain == "uphill":
            return 0.8 - abs(stride - 0.3) - abs(force - 1.5)
        elif self.current_terrain == "downhill":
            return 0.8 - abs(stride - 0.1) - abs(force - 0.5)
        else:  # rough terrain
            return 0.7 - abs(stride - 0.3) - abs(force - 1.0)

    def reset(self):
        """Reset the environment for a new episode"""
        self.current_terrain = random.choice(self.terrain_types)
        self.current_state = "balanced"
        self.steps_taken = 0
        self.fallen = False
        return self.get_state_index()


class QLearningAgent:
    """Q-learning agent that learns to control the robot"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))

        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning algorithm"""
        # Q-learning formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_robot():
    """Main training function"""
    # Initialize environment and agent
    env = RobotEnvironment()
    state_size = len(env.terrain_types) * len(env.robot_states)
    action_size = len(env.actions)
    agent = QLearningAgent(state_size, action_size)

    # Training parameters
    episodes = 1000

    # For tracking progress
    rewards_history = []
    steps_history = []

    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # Episode loop
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            reward, next_state, done = env.take_action(action)

            # Learn from experience
            agent.learn(state, action, reward, next_state, done)

            # Update state and reward
            state = next_state
            total_reward += reward

        # Decay exploration rate
        agent.decay_epsilon()

        # Record history
        rewards_history.append(total_reward)
        steps_history.append(env.steps_taken)

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            clear_console()
            print(f"Episode: {episode + 1}/{episodes}")
            print(f"Exploration rate: {agent.epsilon:.2f}")
            print(f"Average reward (last 10): {np.mean(rewards_history[-10:]):.2f}")
            print(f"Average steps (last 10): {np.mean(steps_history[-10:]):.2f}")
            print("\nQ-table sample (Terrain: flat, State: balanced):")
            state_idx = env.terrain_types.index("flat") * len(env.robot_states)
            print_q_values(agent.q_table[state_idx], env.actions)
            time.sleep(1)

    # Final results
    clear_console()
    print("\n\n")
    print("=" * 75)
    print("Training completed!")
    print(f"Final exploration rate: {agent.epsilon:.2f}")
    print(f"Final average reward (last 10): {np.mean(rewards_history[-10:]):.2f}")
    print(f"Final average steps (last 10): {np.mean(steps_history[-10:]):.2f}")

    # Show learned Q-values for different terrains
    for terrain in env.terrain_types:
        print(f"\nLearned Q-values for {terrain} terrain (balanced state):")
        state_idx = env.terrain_types.index(terrain) * len(env.robot_states)
        print_q_values(agent.q_table[state_idx], env.actions)


def print_q_values(q_values, actions):
    """Print Q-values in a readable format"""
    print("Action (stride, force) -> Q-value")
    print("-" * 30)
    for i, ((stride, force), q_value) in enumerate(zip(actions, q_values)):
        print(f"({stride:.1f}, {force:.1f}) -> {q_value:.2f}")


def clear_console():
    """Clear the console for better visualization"""
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    print("Robot Walking Reinforcement Learning Simulation")
    print("=" * 50)
    print("Teaching a robot to walk using Q-learning...")
    print("The robot will learn to adjust its stride length and leg force")
    print("based on different terrain types.")
    print("=" * 50)
    time.sleep(2)
    train_robot()
