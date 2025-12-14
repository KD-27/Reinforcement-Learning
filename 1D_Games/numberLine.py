import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_episodes = 1000

class env():

    def __init__(self, start_state = 0, goal = 5, state_bound = range(-10,11)):
        self.goal = goal
        self.states = state_bound
        self.start_state = start_state
        
    def step(self, action):
        """Apply an Action and return next state , reward and termination status
        """
        prev_state = self.current_state

        self.current_state += action
        self.current_state = max(min(self.states), min(self.current_state, max(self.states)))

        reward = self.reward(prev_state)
        done = self.terminate()

        return self.current_state, reward, done

    def reward(self, prev_state):
        """Computing the reward based on the transition from previous step to current step"""
        reward = 0
        if self.current_state == self.goal: # Reaching the goal
            reward += 10
        elif self.current_state == min(self.states) or self.current_state == max(self.states):
            reward -= 10
        elif abs(self.goal - self.current_state) < abs(self.goal - prev_state): # Moving towards the goal
            reward += 1
        else:
            reward -= 8 # Moving Away from the goal
        return reward

    def terminate(self):
        """Checking if the current state meet the termination condition"""
        if self.current_state == self.goal:
            return True
        
        else:
            return False
        
    def reset(self):
        """Resetting the environment to start a new episode"""
        self.current_state = random.choice(self.states)
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

        self.state_space = range(-10, 11)
        self.actions = [-1, 1]

        self.Q_table = np.zeros((len(self.state_space), len(self.actions)))
        self.state_index = {state: i for i, state in enumerate(self.state_space)}

    def state_to_index(self, state):
        """Safer Dictionary mapping to index"""
        return self.state_index[state]

    def q_learn(self, state, action, reward, next_state, done):
        """Applying Q learning to improve understanding on the environment"""
        idx = self.state_to_index(state)
        next_idx = self.state_to_index(next_state)
        action_idx = self.actions.index(action)

        if done:
            target = reward
        else:
            target = reward + self.discount_rate*np.max(self.Q_table[next_idx])

        self.Q_table[idx][action_idx] += self.lr*(target - self.Q_table[idx][action_idx])

    def take_action(self,state):
        """Executing the action moving from exploration to exploitation"""
        if random.random() < self.epsilon:
            self.last_action = random.choice(self.actions)
        else:
            idx = self.state_to_index(state)
            self.last_action = self.actions[np.argmax(self.Q_table[idx, :])]

        return self.last_action
    
    def decay(self):
        """Changing value to move from exploration to exploitation"""
        self.epsilon = max(self.min_epsilon, self.epsilon_decay*self.epsilon)

env = env()
agent = agent()

def visualize():
    plt.figure(figsize=(8,6))
    sns.heatmap(agent.Q_table, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=agent.actions, yticklabels=list(agent.state_space))
    plt.title("Q-table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()

    # Optional: visualize best action per state
    best_actions = [agent.actions[np.argmax(agent.Q_table[agent.state_index[s], :])] 
                    for s in agent.state_space]

    plt.figure(figsize=(8,4))
    plt.bar(list(agent.state_space), best_actions, color='skyblue')
    plt.title("Best Action per State")
    plt.xlabel("State")
    plt.ylabel("Action (-1 or +1)")
    plt.show()

def train():
    for episodes in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            agent.q_learn(state, action, reward, next_state, done)
            state = next_state
        agent.decay()

    visualize()

if __name__ == "__main__":
    train()