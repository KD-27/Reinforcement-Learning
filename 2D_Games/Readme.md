# 2D Grid World Q-Learning (with Obstacles & Stochasticity)

This project is a **from-scratch tabular Q-learning implementation** in a **2D grid world**.
An agent learns to navigate around obstacles and reach a goal using **explicit Bellman updates**, **ε-greedy exploration**, and **stochastic transitions** — no RL libraries, no neural networks.

This builds directly on a 1D version and focuses on **intuition, visualization, and correctness**.

---

## Environment

* **Grid:** 8 × 8 discrete grid

* **State:** `(x, y)` grid coordinates

* **Actions:**

  * Up `(-1, 0)`
  * Down `(1, 0)`
  * Left `(0, -1)`
  * Right `(0, 1)`

* **Start State:** Randomized (never inside obstacles)

* **Goal State:** `(7, 7)`

* **Obstacles:** Fixed walls forming corridors and dead ends

* **Transitions:**

  * **Stochastic dynamics**

    * 80% intended action
    * 20% random action (noise)
  * States clipped to grid boundaries
  * Obstacle collisions revert the move

---

## Rewards (Sparse)

* **+10** for reaching the goal
* **−0.1** per step otherwise
* Episode terminates **only on goal reach**

This keeps the reward signal sparse while still encouraging shorter paths.

---

## Agent

* **Tabular Q-learning** (no function approximation)

* **Q-table shape:** `(grid_x, grid_y, action)`

* **Learning rule:** Standard Bellman update

  [
  Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_a Q(s',a) - Q(s,a)\right]
  ]

* **Exploration:** ε-greedy

  * Starts fully random
  * Gradually decays toward exploitation

* **Handles stochastic transitions** naturally through expected value updates

---

## Training

* Online learning at every step
* Randomized starting positions
* ε decays per episode
* Optional live visualization during training

Training converges to **robust paths** that tolerate action noise and avoid obstacles.

---

## Visualization

### 1️⃣ Training Visualization

* Shows agent movement during training
* Displays:

  * Obstacles
  * Path history
  * Current agent position
* Automatically closes when the goal is reached

### 2️⃣ Test-Time Trajectory Visualization

* Runs with **ε = 0 (pure greedy policy)**
* Visualizes:

  * Exact policy behavior
  * Final learned paths
  * Success from different start states

This separation ensures **learning ≠ evaluation**.

---

## How to Run

Train and test the agent:

```bash
python gridWorld.py
```

During execution:

* Training runs first
* After training, fixed start states are tested and visualized

---

## Files

* `gridWorld.py`

  * Environment definition
  * Q-learning agent
  * Training loop
  * Training & test visualizations
* `README.md`

  * Project overview and explanation

---

## Dependencies

* Python 3.x
* `numpy`
* `matplotlib`

---

## Key Takeaways

This project builds intuition for:

* Bellman value propagation in **2D**
* Learning under **stochastic dynamics**
* Sparse vs shaped rewards
* Risk-aware navigation near obstacles
* Why visualization is critical for debugging RL systems

---

