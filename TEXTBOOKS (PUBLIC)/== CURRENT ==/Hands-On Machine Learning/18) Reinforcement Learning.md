
- Reinforcement learning (RL)
	- Policy gradients
	- Deep Q-networks

# Learning to Optimize Rewards

- In RL, a software agent makes observations and takes actions with an environment, and in return is receives rewards from the environment

# Policy Search

- The algorithm a software agents uses to determined its actions is called its policy
- The policy cloud be a NN taking observations as inputs and outputting the action to take

![[Pasted image 20260304125317.png]]

- Policy does not have to be deterministic
- Stochastic policy
	- Involved some randomness
- 2 policy parameters
	- Probability $p$
	- Angle range $r$
- Policy search
	- Brute force approach
- If the policy space is too large, finding a good set of parameters is difficult
- Genetic algorithms
	- Randomly create a first generation of 100 policies, and removes the worst
	- Makes the 20 survivors produce offspring
	- Continue iterating through generations until a good policy is reached

![[Pasted image 20260304125601.png]]


- Policy gradients (PC)
	- Tweaking parameters to optimize the reward

# Introduction to OpenAI Gym

- One of the challenges of RL is that in order to train an agent, there needs to be a working environment
	- Game simulator
- Simulated environment at least for bootstrap training
	- PyBullet
	- MuJoCo
- OpenAI Gym
	- Provides a wide variety of simulated environments

```python
%pip install -q -U gym
%pip install -q -U gym[classic_control,box2d,atari,accept-rom-license]
```
- Installs the libraries and classic environments from control theory to balancing a pole on a cart
- 2D physics engine
- Arcade Learning Environment (ALE)

```python
import gym
env = gym.make("CartPole-v1", render_mode="rgb_array")
```

![[Pasted image 20260304130051.png]]

- After the environment is created, initialize it using it using `reset()`
- Observations depend on the type of environment
	- For this example it is a 1D NumPy array containing 4 floats representing the cart's (horizontal position, velocity, angle, angular velocity)
- Call `render()` to render this environment as an image
- Use `imshow()` to display image
- `Discrete(2)` mean that the possible actions are int 0 and 1 (left, right)

```python
obs, info = env.reset(seed=42)
obs
array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32)
>>> info
{}
img = env.render()
image.shape
(400, 600, 3)
env.action_space
Discrete(2)
action = 1
obs, reward, done, truncated, info = env.step(action)
obsarray([ 0.02727336,  0.18847767,  0.03625453, -0.26141977], dtype=float32)
>>> reward
1.0
>>> done
False
>>> truncated
False
>>> info
{}
```

- `obs`
	- New observation
- `reward`
	- Reward of 1.0 at every step
- `done`
	- `True` when the episode if over
- `truncated`
	- `True` when an episode is interrupted early
- `info`
	- Environment specific dictionary may provide extra information
- `close()`
	- Free resources


```python
# hardcose simple policy
# acc. right when pole is leaning left, and vice versa
def basic_policy(obs):
	angle = obs(2)
	return if angle < 0 else 1
	
totals = []
for episode in range(500):
	episode_rewards = 0
	obs, info = env.reset(seed=episode)
	for step in range(200):
		action = basic_policy(obs)
		obs, reward, done, truncated, info = env.step(action)
		episode_rewards += reward
		if done or truncated:
			break
			
	totals.append(episode_rewards)
	
import numpy as np
>>> np.mean(totals), np.std(totals), min(totals), max(totals)
(41.698, 8.389445512070509, 24.0, 63.0)
```

- With 500 tries, the policy only manages to keep the pole up for 63 consecutive steps
- Cart oscillates left and right


# Neural Network Polices

- Create a NN policy
	- Take an observation as input, and then output the action to be executed
	- Estimate a probability for each action, and select an action randomly

![[Pasted image 20260304130958.png]]

- Choosing random action, instead of the highest score lets the agent find the right balance between exploring new actions and exploiting the actions that are known to work well
- Exploration/exploitation dilemma
- Past actions and observations can be ignored

```python
import tensorflow as tf
model = tf.keras.Sequential([
	tf.keras.layers.Dense(5, activation="relu"),
	tf.keras.layers.Dense(1, activation="sigmoid")
])
```

- Use `Sequential` model to define the policy network
- Input is the observation space of 4, with 5 hidden units
- Output a single probability (going left)

# Evaluating Actions: The Credit Assignment Problem

- Train NN by minimizing the cross entropy between the estimated probability distribution and the target probability distribution
- Guidance is only from agents
- Credit assignment problem
	- When the agent gets a reward, it is hard for it to know which action should get credited for it
- Return evaluation
	- Evaluate an action based on the sum of all the reward that come after it
	- Using a discount factor, $\gamma$
		- $\gamma$ = 0 => future rewards don't count for much compared to immediate rewards
		- $\gamma$ = 1 => future rewards count for more compared to immediate rewards

![[Pasted image 20260304131736.png]]

- A good action may be followed by bad a


# Policy Gradients

# Markov Decision Processes

# Temporal Difference Learning

# Q-Learning

## Exploration Policies
## Approximate Q-Learning and Deep Q-Learning

# Implementing Deep Q-Learning

# Deep Q-Learning Variants

## Fixed Q-Value Targets
## Double DQN
## Prioritized Experience Replay
## Dueling DQN

# Overview of Some Popular RL Algorithms