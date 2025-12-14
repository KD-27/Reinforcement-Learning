# 2D Grid World Q-Learning Game

This project extends a **from-scratch tabular Q-learning implementation** from 1D to a **2D grid world**.
An agent learns to navigate a discrete grid and reach a goal using **explicit Bellman updates** and an **ε-greedy policy** — no RL libraries, no function approximation.

---

## Environment

* **Grid:** 7 × 7 discrete grid
* **State:** `(x, y)` grid coordinates
* **Actions:**

  * Up `(-1, 0)`
  * Down `(1, 0)`
  * Left `(0, -1)`
  * Right `(0, 1)`
* **Start State:** `(0, 0)` (optionally randomized)
* **Goal State:** `(0, 6)`
* **Transitions:**

  * Deterministic movement
  * States clipped to grid boundaries
* **Rewards:**

  * **+10** for reaching the goal
  * **+1** for moving closer (Manhattan distance)
  * **−0.1** for moving farther or not improving
* **Episode ends** only when the goal is reached

Reward shaping creates a smooth value surface that propagates outward from the goal.

---

## Agent

* **Tabular Q-learning** (no neural networks)
* **Q-table shape:** `(grid_x, grid_y, action)`
* **Learning rule:** Standard Bellman update with TD error
* **Discount factor:** Propagates long-term rewards correctly
* **Exploration:** ε-greedy policy

  * Starts fully random
  * Gradually shifts toward exploitation via decay

---

## Training

* Agent interacts with the environment using:

  * `reset()`
  * `step(action)`
* Learning occurs **online at every transition**
* ε decays **per episode**
* Training converges to a shortest-path policy

---

## Visualization & Verification

After and during training, the following visualizations are used:

### 1. Value Map + Policy Arrows

* **Heatmap:** `V(x, y) = max_a Q(x, y, a)`
* **Arrows:** Greedy action at each grid cell
* Confirms:

  * Smooth Bellman value propagation
  * Correct directional flow toward the goal
  * Symmetry where expected

### 2. Episode Trajectory Animation

* Shows the **agent moving step-by-step** through the grid
* Visualizes:

  * Exploration vs exploitation
  * Loops or oscillations
  * Convergence to direct paths
* Visualization closes automatically once the goal is reached

---

## How to Run

1. Train the agent:

```bash
python smallGrid.py
```

2. During training:

   * Value maps and policies are visualized every N episodes
   * Agent motion can be animated for individual episodes

---

## Files

* `smallGrid.py` — Environment, agent, training loop, and visualizations
* `README.md` — Project overview and usage instructions

---

## Dependencies

* Python 3.x
* `numpy`
* `matplotlib`

---

## Key Takeaways

This project builds intuition for:

* Bellman value propagation in **2D**
* How geometry shapes value surfaces
* The effect of reward shaping on convergence
* Debugging RL systems visually before scaling up

---
