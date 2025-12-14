import numpy as np
import matplotlib.pyplot as plt
import random

num_episodes = 500

class env():
    def __init__(self):
        self.grid_size = 7
        self.goal_state = (0,6)
        self.start_state = (0,0)
        self.current_state = self.start_state
        self.maze = np.zeros((self.grid_size,self.grid_size))
    
    def step(self, action):
        """Apply an Action and return next state , reward and termination status"""
        prev_state = self.current_state

        x, y = self.current_state
        x += action[0]
        y += action[1]

        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))
        self.current_state = (x, y)

        reward = self.reward(prev_state)
        done = self.terminate()

        return self.current_state, reward, done

    def reward(self,prev_state):
        """Computing the reward based on the transition from previous step to current step"""
        reward = 0

        if self.current_state == self.goal_state:
            reward += 10
        elif self.manhattan_distance(self.current_state) < self.manhattan_distance(prev_state):
            reward += 1
        else:
            reward -= 0.1
        return reward
    
    def manhattan_distance(self,state):
        """Calculating the distance to the goal state from any given state"""
        x, y = state
        xg, yg = self.goal_state

        distance = abs(xg - x) + abs(yg - y)
        return distance

    def terminate(self):
        """Checking if the current state has reached the goal"""
        if self.current_state == self.goal_state:
            return True
        else:
            return False

    def reset(self):
        """Resetting the environment to start a new episode"""
        # self.current_state = self.start_state
        self.current_state = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        return self.current_state

class agent():
    def __init__(self):
        #Q learning constants
        self.lr = 0.1
        self.discount_rate = 0.95

        # Greedy method constants
        self.min_epsilon = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

        #actions
        self.actions = [(0,1),(1,0),(0,-1),(-1,0)]

        self.grid_size = 7
        self.Qtable = np.zeros((self.grid_size, self.grid_size, len(self.actions)))

    def qlearn(self,state, action, reward, next_state, done):
        """Applying Q learning to improve understanding on the environment"""
        x, y = state
        xn, yn = next_state
        action_idx = self.actions.index(action)

        if done:
            target = reward
        else:
            target = reward + self.discount_rate*np.max(self.Qtable[xn][yn])

        self.Qtable[x][y][action_idx] += self.lr*(target - self.Qtable[x][y][action_idx])

    def take_action(self, state):
        """Executing the action moving from exploration to exploitation"""
        if random.random() < self.epsilon:
            self.last_action = random.choice(self.actions)
        else:
            x, y = state
            self.last_action = self.actions[np.argmax(self.Qtable[x][y])]

        return self.last_action

    def decay(self):
         """Changing value to move from exploration to exploitation"""
         self.epsilon = max(self.min_epsilon, self.epsilon_decay*self.epsilon)

env = env()
agent = agent()

def visualize_episode(env, agent, max_steps=100, pause_time=0.3):
    state = env.reset()
    done = False
    path = [state]

    fig, ax = plt.subplots(figsize=(5, 5))

    for step in range(max_steps):
        ax.clear()

        # Background grid
        ax.imshow(
            np.ones((env.grid_size, env.grid_size)),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        # Goal
        gx, gy = env.goal_state
        ax.scatter(gy, gx, c="green", s=250, marker="*", label="Goal")

        # Path so far
        xs = [s[0] for s in path]
        ys = [s[1] for s in path]
        ax.plot(ys, xs, c="orange", linewidth=2, label="Path")

        # Agent
        x, y = state
        ax.scatter(y, x, c="blue", s=200, label="Agent")

        ax.set_title(f"Step {step}")
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(env.grid_size - 0.5, -0.5)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        plt.pause(pause_time)

        # Take action
        action = agent.take_action(state)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state

        if done:
            break

    # Pause briefly on success, then close
    plt.pause(0.8)
    plt.close(fig)

def train():
    for episodes in range(num_episodes):
        state = env.reset()
        done = False
        if episodes % 50 == 0:
            visualize_episode(env, agent)
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            agent.qlearn(state, action, reward, next_state, done)
            state = next_state
        agent.decay()

if __name__ == "__main__":
    train()