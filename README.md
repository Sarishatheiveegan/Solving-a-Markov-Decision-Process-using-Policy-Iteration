# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python
import gymnasium as gym
import numpy as np
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=True)
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-10
policy = np.ones(n_states)
action_symbols = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}
# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
def policy_evaluation(policy, env, gamma=0.99, theta=1e-10):

    V = np.zeros(n_states)

    while True:

        delta = 0

        for s in range(n_states):

            old_value = V[s]

            action = policy[s]

            value = 0

            for prob, next_state, reward, done in env.P[s][action]:
                value += prob * (reward + gamma * V[next_state])

            V[s] = value

            delta = max(delta, abs(old_value - V[s]))

        if delta < theta:
            break

    return V

# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------
def policy_improvement(V, env, gamma=0.99):

    new_policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):

        action_values = np.zeros(n_actions)

        for a in range(n_actions):

            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])

        new_policy[s] = np.argmax(action_values)

    return new_policy
print("Name:MARINO SARISHA T")
print("Register Number: 212223240084")
# -------------------------------------------------
# Policy Iteration
# -------------------------------------------------
print("\nInitial Random Policy:\n")

initial_policy_grid = np.array([action_symbols[a] for a in policy])
print(initial_policy_grid.reshape((4, 4)))

# Initial Value Function
initial_V = policy_evaluation(policy, env, gamma, theta)

print("\nInitial State-Value Function:\n")
print(np.round(initial_V.reshape((4, 4)), 4))

iterations = 0

while True:

    iterations += 1

    V = policy_evaluation(policy, env, gamma, theta)

    new_policy = policy_improvement(V, env, gamma)

    if np.array_equal(policy, new_policy):
        break

    policy = new_policy

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

print("\nTotal Policy Iterations:", iterations)

print("\nFinal Optimal State-Value Function:\n")
print(np.round(V.reshape((4, 4)), 4))

print("\nFinal Optimal Policy:\n")

final_policy_grid = np.array([action_symbols[a] for a in policy])
print(final_policy_grid.reshape((4, 4)))

env.close()

```

## Output

<img width="462" height="351" alt="image" src="https://github.com/user-attachments/assets/861756f5-9287-42de-92af-5f0922c47fde" />
<img width="462" height="351" alt="image" src="https://github.com/user-attachments/assets/4775220a-5324-42e4-bfa3-ac464b56e820" />

## Result

The Policy Iteration algorithm was successfully implemented and executed on the FrozenLake environment. The algorithm converged after 4 iterations, producing the optimal policy and the corresponding optimal state-value function, demonstrating its effectiveness in finding the best sequence of actions to maximize the expected cumulative reward.



## Inference
Policy Iteration efficiently solves finite Markov Decision Processes by alternating between policy evaluation and policy improvement. It converges to the optimal policy in a finite number of iterations, making it an effective Dynamic Programming algorithm for environments with known transition probabilities.


