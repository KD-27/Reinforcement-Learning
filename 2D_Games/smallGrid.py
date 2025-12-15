import numpy as np
import matplotlib.pyplot as plt
import random

num_episodes = 5000
grid_size = 8

learning_rate = 0.05
Discount_rate = 0.9
epsilon_decay = 0.999

class env():
    def __init__(self):
        self.grid_size = grid_size
        self.goal_state = (7,7)
        self.start_state = (0,0)
        self.current_state = self.start_state
        self.actions = [(0,1),(1,0),(0,-1),(-1,0)]
        self.obstacles = [(0,3),(1,3),(2,3),(2,4),(3,4),(4,4),(4,1),(4,2),(4,3),(3,6),(4,6),(5,6),(6,6),(7,6)]

    def step(self, action):
        """Apply an Action and return next state , reward and termination status"""
        prev_state = self.current_state

        # Adding stochasticity 80% intended choice else noice
        prob = random.random()
        if prob < 0.8:
            chosen_action = action
        else:
            chosen_action = random.choice(self.actions)

        x, y = self.current_state
        dx, dy = chosen_action
        x += dx
        y += dy
        self.current_state = (x, y)
        
        if (x, y) in self.obstacles:
            x -= dx
            y -= dy

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
        # For Sparse reward version comment the condition below
        # elif self.manhattan_distance(self.current_state) < self.manhattan_distance(prev_state):
        #     reward += 1
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
        while True:
            state = (
                np.random.randint(self.grid_size),
                np.random.randint(self.grid_size)
            )
            if state not in self.obstacles:
                self.current_state = state
                return self.current_state

class agent():
    def __init__(self):
        #Q learning constants
        self.lr = learning_rate
        self.discount_rate = Discount_rate

        # Greedy method constants
        self.min_epsilon = 0.1
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay

        #actions
        self.actions = [(0,1),(1,0),(0,-1),(-1,0)]

        self.grid_size = grid_size
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

def visualize_train(env, agent, max_steps=200, pause_time=0.3):
    state = env.reset()
    done = False
    path = [state]

    fig, ax = plt.subplots(figsize=(5, 5))

    for step in range(max_steps):

        # -------- DRAW --------
        ax.clear()

        # Background grid
        ax.imshow(
            np.ones((env.grid_size, env.grid_size)),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        # Obstacles
        for (ox, oy) in env.obstacles:
            ax.add_patch(
                plt.Rectangle(
                    (oy - 0.5, ox - 0.5),
                    1, 1,
                    color="black",
                    alpha=0.85
                )
            )

        # Goal
        gx, gy = env.goal_state
        ax.scatter(gy, gx, c="limegreen", s=300, marker="*", label="Goal")

        # Path
        xs = [s[0] for s in path]
        ys = [s[1] for s in path]
        ax.plot(ys, xs, c="orange", linewidth=2, label="Path")

        # Agent
        x, y = state
        ax.scatter(y, x, c="dodgerblue", s=220, label="Agent")

        ax.set_title(f"Step {step}")
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(env.grid_size - 0.5, -0.5)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        plt.pause(pause_time)

        # -------- STEP --------
        action = agent.take_action(state)
        next_state, _, done = env.step(action)
        state = next_state
        path.append(state)

        # -------- FINAL RENDER --------
        if done:
            ax.clear()

            ax.imshow(
                np.ones((env.grid_size, env.grid_size)),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            for (ox, oy) in env.obstacles:
                ax.add_patch(
                    plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1, color="black")
                )

            ax.scatter(gy, gx, c="limegreen", s=350, marker="*", label="Goal")

            xs = [s[0] for s in path]
            ys = [s[1] for s in path]
            ax.plot(ys, xs, c="orange", linewidth=2)

            x, y = state
            ax.scatter(y, x, c="red", s=260, label="Agent (Goal)")

            ax.set_title("Goal Reached")
            ax.set_xlim(-0.5, env.grid_size - 0.5)
            ax.set_ylim(env.grid_size - 0.5, -0.5)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

            plt.pause(0.8)
            break

    plt.close(fig)

def train():
    for episodes in range(num_episodes):
        state = env.reset()
        done = False
        if episodes % 2500 == 0:
            visualize_train(env, agent)
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            agent.qlearn(state, action, reward, next_state, done)
            state = next_state
        agent.decay()

def visualize_test(env, path, pause_time=0.3):
    fig, ax = plt.subplots(figsize=(5,5))

    for step, state in enumerate(path):
        ax.clear()

        # Background
        ax.imshow(np.ones((env.grid_size, env.grid_size)), cmap="gray", vmin=0, vmax=1)

        # Obstacles
        for (ox, oy) in env.obstacles:
            ax.add_patch(plt.Rectangle((oy - 0.5, ox - 0.5), 1, 1, color="black", alpha=0.85))

        # Goal
        gx, gy = env.goal_state
        ax.scatter(gy, gx, c="limegreen", s=300, marker="*", label="Goal")

        # Path so far
        xs = [s[0] for s in path[:step+1]]
        ys = [s[1] for s in path[:step+1]]
        ax.plot(ys, xs, c="orange", linewidth=2, label="Path")

        # Agent
        x, y = state
        color = "red" if state == env.goal_state else "dodgerblue"
        ax.scatter(y, x, c=color, s=220, label="Agent")

        ax.set_title(f"Test | Step {step}")
        ax.set_xlim(-0.5, env.grid_size - 0.5)
        ax.set_ylim(env.grid_size - 0.5, -0.5)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        plt.pause(pause_time)

    plt.pause(1.0)
    plt.close(fig)

def test(env, agent, start, max_steps=200):
    """Run a single episode with a fixed start, return path and success."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure greedy

    state = start
    env.current_state = state
    path = [state]
    done = False

    for _ in range(max_steps):
        action = agent.take_action(state)
        next_state, _, done = env.step(action)
        state = next_state
        path.append(state)
        if done:
            break

    agent.epsilon = old_epsilon
    return path, done


if __name__ == "__main__":
    # Training Tabular Q-Learning Model
    train()

    # Testing Tabular Q-Learning Model
    starting_points = [(0,0), (3,3)]
    for start in starting_points:
        path, done = test(env, agent, start)
        print(f"Start {start} | Path length: {len(path)} | Reached goal: {done}")
        visualize_test(env, path)