
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

- A good action may be followed by bad actions, resulting in a good action getting a low return
- Action advantage
	- Estimate how much better or worse an action is, compared to the other possible actions
	- Run many episodes and normalize all the action returns

# Policy Gradients

- PG algorithms optimize the parameters of a policy by following the gradients toward higher rewards
- REINFORCE
	- NN policy play the games several times
		- Computes the gradients that would make the chosen action more likely
	- Computes each action's advantage
	- If action is positive, apply gradient, to make more likely in the future
	- If negative, opposite gradient is applied
	- Compute the mean of the resulting gradient vectors, and use it ti perform a gradient descent step

```python
# one episode
def play_one_step(env, obs, model, loss_fn):
	with tf.GradientTape() as tape:
		left_proba = model(obs[np.newaxis])
		action = (tf.random.uniform([1, 1]) > left_proba)
		y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
		loss = tf.reduce_mean(loss_fn(y_target, left_proba))
		
	grads = tape.gradient(loss, model.trainable_variables)
	obs, reward, done, truncated, info = env.step(int(action))
	return obs, reward, done, truncated, grads
```

- Cal the model with a single observation
- Sample a random float between 0 and 1, and check if it is greater than `left_proba`
- Define the target probability of going left
	- If the action is left, the probability will be 1
	- If the action is right, the probability will be -
- Compute the loss, and use the tape to compute the gradient of the loss with regard to the model's trainable variable
- Play the selection actions, return the new observation, reward, episode ending, truncated, and gradients

```python
# multiple episodes
def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
	all_rewards = []
	all_grads = []
	for episode in range(n_episodes):
		current_rewards = []
		current_grads = []
		obs, info = env.reset()
		for step in range(n_max_steps):
			obs, reward, done, truncated, grads = play_one_step(
				env, obs, model, loss_fn)
			current_rewards.append(reward)
			current_grads.append(grads)
			if done or truncated:
				break
				
		all_rewards.append(current_rewards)
		all_grads.append(current_grads)
	
	return all_rewards, all_grads
```

- Returns a list of reward lists and gradient lists per episode
- The function is play several time, then it will go back and look at all the rewards, discount them, and normalize them
- Compute the sum of future discounted rewards at each step
- Normalize discounted rewards (returns)

```python
def discount_rewards(rewrads, discount_factor):
	discounted = np.array(rewards)
	for step in range(len(rewards) - 2, -1, -1):
		discounted[step] += discounted[step + 1] * discounted_factor
	return discounted
	
def discount_and_normalize_rewards(all_rewards, discount_factor):
	all_dicounted_rewards = [discount_rewards(rewards, discount_factor)
		for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()
	return [(discounted_rewards - rewward_mean) / reward_std
		for discounted_rewards in all_dicounted_rewards]
		
discount_rewards([10, 0, -50], discount_factor=0.8)
array([-22, -40, -50])

discount_and_normalize_rewards([[10, 0, -50], [10, 20]],
...                                discount_factor=0.8)
[array([-0.28435071, -0.86597718, -1.18910299]),
 array([1.26665318, 1.0727777 ])]
```

- The first episode is worse, then the second
- Run the algorithm with the hyperparameters

```python
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = tf.keras.losses.binary_crossentropy
```

```python
# build and run training loop
for iteration in range(n_iteration):
	all_rewards, add_grads = play_multiple_episodes(
		env, n_episodes_per_update, n_max_steps, model, loss_fn
	)
	all_final_rewards = discount_and_normalize_rewards(all_rewards,
			discount_factor)
	all_mean_grads = []
	for var_index in range(len(model.trainable_variables)):
		mean_grads = tf.reduce_mean(
			[final_reward * all_grads[episode_index][step][var_index]
			for episode_index, final_rewards in enumearte(all_final_rewards)
				for step, final_reward in enumerate(final_rewards)], axis=0
		)
		all_mean_grads.append(mean_grads)
		
	optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
```

- At each training iteration, the loop called `play_multiple_episodes()`
	- Returns the rewards and gradients for each step
- Call `discount_and_normalize_rewards()`
	- Compute each action's normalized advantage
- go through each trainable variable and compute the weighted mean of the gradients for that variable over all episodes and steps
- Apply the mean gradients using the optimizer
	- The model's trainable variables are tweaked

- This code will train the NN policy, and will learn to balance the pole on the cart
- Mean reward is 200 per episode (close to max)
- Sample inefficient
	- Needs to explore the game for a long time to make progress
- Actor-critic
	- Class of RL that combine policy-based and value-based approaches

# Markov Decision Processes

- Markov chains
	- Stochastic processes with no memory
	- Fixed number of states
	- Randomly evolved from one state to another
	- Depends only on the pair, not on past states

![[Pasted image 20260304134517.png]]

- Start in state $s_0$, there is a 70% chance that is will remain in that state at the next step
- If it does to state $s_1$, it will then most likely go to state $s_2$ (90%), then back to $s_1$
- Fall into state $s_4$ and remain there forever (terminal state)

- At each step, an agent can choose one of several possible actions, and the transition probabilities depend on the chosen action
- Some state transitions return reward
- Agent's goal is to find a policy to maximize reward over time
- 

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