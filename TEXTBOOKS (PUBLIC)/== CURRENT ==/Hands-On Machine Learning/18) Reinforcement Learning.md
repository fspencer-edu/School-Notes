
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

![[Pasted image 20260304134752.png]]

Circles => States
Diamonds => Discrete actions

- $s_0$ -> $a_0$
- $s_1$ -> $a_0$ or $a_2$ (fire)
- $s_2$ -> $a_1$

- Estimate the optimal state value of any state, $V*(S)$
	- Sum of all discounted feature rewards the agent can expect on average after it reaches the state, assuming is acts optimally
- Bellman optimality equation
	- Recursive equation says that if the agent act optimally, then the optimal value of the current state is equal to the reward it will get on average after taking one optimal action, plus the expected optimal value of all possible next states


- Bellman optimality equation

![[Pasted image 20260304135254.png]]

- Initialize all the state value estimates to zero
- Iteratively update then using the value iteration algorithm
- Estimates will guarantee to converge to the optimal state values


- Value iteration algorithm

![[Pasted image 20260304135351.png]]

- The algorithm is an example of dynamic programming
- Optimal state values help evaluate a policy, but does not give the optimal policy
- Estimate the optimal state-action values (Q-values)
	- State-action pair (s, a), $Q*(s, a)$
	- Sum of discounted feature rewards the agent can expect on average after it reaches the state s and chooses action a, but before it sees the output of this action

- Q-value iteration algorithm

![[Pasted image 20260304135533.png]]

$\pi*(s)$ = optimal policy


![[Pasted image 20260304135604.png]]

```python
transition_probabilities = [  # shape=[s, a, s']
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None]
]
rewards = [  # shape=[s, a, s']
    [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
    [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]
]
possible_actions = [[0, 1, 2], [0, 2], [1]]

# init. Q-values to zero, -inf for impossible actions
Q_values = np.full((3, 3), -np.inf)  # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0  # for all possible actions
    
# Q-value iteration
gamma = 0.90 # discount factor
for iteration in range(50)
	Q_prev = Q_values.copy()
	for s in range(3):
		for a in possible_actions[s]:
			Q_values[s, a] = np.sum([
				transition_probabilities[s][a][sp]
				* (rewards[s][a][sp] + gamma * Q_prev[sp].max())
				for sp in range(3)])
				
>>> Q_values
array([[18.91891892, 17.02702702, 13.62162162],
       [ 0.        ,        -inf, -4.87971488],
       [       -inf, 50.13365013,        -inf]])
       
Q_values.argmax(axis=1)  # optimal action for each state
array([0, 0, 1])
```

Optimal policy
- $s_0$ -> $a_0$
- $s_1$ -> $a_0$
- $s_2$ -> $a_1$

- If the discount factor increases to 0.95 from 0.90, the $s_1$ -> $a_2$ (fire)
	- More value for the future reward, the more pain that is endured


# Temporal Difference Learning

- RL problems with discrete actions can be modelled as Markov decision processes (MDP)
- Agent initially does not know that the transitions probabilities are T(s, a, s') and the rewards R(s, a s')
- For the agent to learn, it must experience each state and transition at least one
- Temporal different (TD) learning
	- Takes into account the agent has partial knowledge of MDP
	- Exploration policy
		- TD learning algorithm updates the estimates of the state values based on the transitions and rewards that are observed


- TD Learning Algorithm

![[Pasted image 20260304140319.png]]

![[Pasted image 20260304140330.png]]

- TD learning is similar to stochastic gradient descent
	- Handles one sample at a time
	- Converge if you gradually reduce the learning rate
	- Otherwise, bound around the optimum Q-values

# Q-Learning

- Q-learning algorithm is an adaptation of the Q-value iteration
- Watching an agent play (randomly) and gradually improves its estimated of the Q-values
- When is has accurate Q-values, then the optimal policy is chosen from the highest Q-value (greedy policy)

- Q-Learning algorithm

![[Pasted image 20260304140542.png]]

- For each state-action pair (s, a), the algorithm keeps track of a running average of the rewards the agents gets upon leaving the state s with action a, plus the sum of discounted future rewards is expects
	- Take the max of the Q-values for the next state s'

```python
# step function, get resulting state and reward
def step(state, action):
	probas = transition_probabilities[state][action]
	next_state = np.random.choice([0, 1, 2], p=probas)
	reward = rewards[state][action][next_state]
	return next_state, reward
	
# exploration policy
def exploration_policy(state):
	return np.random.choice(possible_actions[state])
	
# init. Q-value
alpha0 = 0.05  # initial learning rate
decay = 0.005  # learning rate decay
gamma = 0.90  # discount factor
state = 0  # initial state

for iteration in range(10_000):
	action = exploration_policy(state)
	next_state, reward = step(state, action)
	next_value = Q_values[next_state].max() # greedy policy
	alpha = alpha 0 / (1 + iteartion * decay)
	Q_values[state, action] *= 1 - alpha
	Q_values[state, action] += alpha * (reward + gamma * next_value)
	state = next_state
```
- Algorithm will converge the optimal Q-values
- Q-value iteration converges in less than 20 iterations
- Q-learning algorithm takes about 8,000
	- Not knowing the transition prob. or rewards make it difficult to find the optimal policy

![[Pasted image 20260304141127.png]]

- Q-learning algorithm is also called the off-policy
	- Policy being trained is not necessarily the one. used during training
	- Policy being executed (exploration) was random, while trained was never used
	- After training, the optimal policy corresponds to the systematically choosing the action with the highest Q-value
- Policy gradients is an on-policy
	- Explores the world using the policy being trained

## Exploration Policies

- Q-learning can work only if the exploration policy explores the MDP thoroughly enough
- Better option is to use the $\epsilon$-greedy policy
	- At each step is acts randomly with probability, $\epsilon$ or greedy with probability $1 -\epsilon$
- The advantages is that is will spend more time exploring the interesting parts of the environment
- Rather than relying only on chance for exploration, another approach is to encourage the exploration policy to try actions that is has not tried before

- Q-learning using an exploration function

![[Pasted image 20260304145652.png]]

## Approximate Q-Learning and Deep Q-Learning

- The main problem with Q-learning is that is does not scaled well to large MDPs with many states and actions
- Approximate Q-learning
	- Find function $Q_{\theta}(s,a)$ that approximates the Q-value of any state-action pair (s, a) using a manageable number of parameters
- Using deep NN can work better than linear combinations of extracted features
- DNN used to estimated Q-values is called a deep Q-network (DQN)
	- Deep Q-learning
- Execute the DQN on the next state, s' for all possible actions a'
- Pick the highest Q-value and discount it, resulting in an estimate of the sum of future discounted rewards
- By summing the reward r and future discounted value estimate, we get a target Q-value y(s, a)

- Target Q-value

![[Pasted image 20260304150121.png]]

- The target Q-value can be used during training step using any gradient descent algorithm
- Minimize the squared error between the estimated Q-value and target Q-value, or the Huber loss to reduce the algorithm's sensitivity to large errors

# Implementing Deep Q-Learning

- NN that takes a state-action pair as input, and outputs an approximate Q-value
- Use a NN that takes only a state as input, and outputs one approximate Q-value for each possible action
- Choose the largest predicted Q-value
- Instead of training the DQN on the latest experiences, stores all the experiences in a replay buffer/memory
- Sample a random training batch from it at each training iteration

```python
input_shape = [4]  # == env.observation_space.shape
n_outputs = 2  # == env.action_space.n

model = tf.keras.Sequential([
	tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
	tf.keras.Dense(32, activation="elu"),
	tf.keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
	if np.random.rand() < epsilon:
		return np.random.randint(n_outputs)
	else:
		Q_values = model.predict(state[np.newaxis], verbose=0)[0]
		return Q_values.argmax()
		
from collections import deque
replay_buffer = deque(maxlen=2000)
```
- A deque is a queue elements that can be efficiently added or removed from both ends
- Random access can be slow if the queue gets long
	- Use a circular buffer for long sequences
- 6 elements
	- state
	- action
	- Reward
	- Next state
	- Ended boolean
	- Truncated boolean

```python
# sample a random experience, and return 6 elements
def sample_experiences(batch_size):
	indices = np.random.randint(len(replay_buffer), size=batch_size)
	batch = [replay_buffer[index] for index in indices]
	return [
		np.array([experience[field_index] for experience in batch])
		for field_idnex in range(6)
	] # [states, actions, rewards, next_states, dones, truncateds]
	
# single step
def play_one_step(env, state, epsilon):
	action = epsilon_greedy_policy(state, epsilon)
	next_state, reward, done, truncated, info = env.step(action)
	replay_buffer.append((state, action, reward, next_state, done, truncated))
	return next_state, reward, done, truncated, info
	
# batch of experiences from the replay buffer and train with single grad. descent
batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = tf.keras.losses.mean_squared_error

def training_step(batch_size):
	experiences = sample_experiences(batch_size)
	states, actions, rewards, next_states, dones, truncateds = experiences
	next_Q_values = model.predict(next_states, verbose=0)
	max_next_Q_values = next_Q_values.max(axis=1)
	runs = 1.0 - (dones | truncateds)
	target_Q_values = rewards + runs * discount_factor * max_next_Q_values
	target_Q_values = target_Q_values.reshape(-1, 1)
	mask = tf.hot_one(actions, n_outputs)
	wiith tf.GradientTape() as tape:
		all_Q_values = model(states)
		Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
		loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
		
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

- Define hyperparameters, and create the optimizer and loss function
- `training_step()`
	- Starts by sampling a batch of experiences, then uses the DQN to predict the Q-value for each possible action in each next state
	- Keep the max Q-value for each next state
	- Compute the target Q-value for each state-action pair
- Use DQN to compute the Q-value for each state-action, but the DQN will also output the Q-values for the other possible actions
	- Mask out all the Q-values
	- Use one hot encoding to convert an array of action indices into a mask
- Compute the loss
	- Mean squared error between target and prediction Q-values
- Perform a gradient descent step to minimize the loss with regards to the model's trainable variables

```python
# train model
for episode in range(600):
	obs, info = env.reset()
	for step in range(200):
		epsilon = max(1 - episode / 500, 0.01)
		obs, reward, done, truncated, info = play_one_step(env, obs, epsilon)
		if done or truncated:
			break
			
	if episode > 50:
		training_step(batch_size)
```

- Run 600 eps, each with a max of 200 steps
- At each step, compute the epsilon value for the greedy policy (1 to 0.01 linearly)
- If we past eps 50, call the training step function to train the model on one batch sampled from the replay buffer

![[Pasted image 20260304152131.png]]

- Catastrophic forgetting
	- As the agent explores the environment, it updates it policy, but what it learns in one part of the environment may break what is learned early in other parts of the environment
		- Increase the replay buffer
		- Tune leraning rate
		- Activation function
- Loss is poor indicator of the model's performance

# Deep Q-Learning Variants

## Fixed Q-Value Targets

- In the basic deep Q-learning algorithm
	- Model is used to make predictions and set its own targets
- This can lead to a unstable feedback loop
	- Diverge, oscillate, freeze
- Online model
	- Learns at each step and is used to move the agent around
- Target model
	- Define the target

```python
target = tf.keras.models.clone_model(model)
target.set_weights(model.get_weights())

# use target model instead of online model
next_Q_values = target.predict(next_states, verbose=0)

# copy weights of online model to target model, at regular intervals
if episode % 50 == 0:
	target.set_weights(model.get_weights())
```

- The target model is updated less often than the online model
- Q-value targets are more stable, and the feedback look is dampened

## Double DQN

- Double DQN
	- Target network is prone to overestimating Q-values
	- Some Q-values are greater than others by change
	- The target model will always select the largest Q-value, which overestimated the true Q-value
	- Using online model instead of the target model when selecting the best actions for the next states

```python
def training_step(batch_size):
	experiences = sample_experiences(batch_size)
	states, actions, rewards, next_states, dones, truncateds = experiences
	next_Q_values = model.predict(next_staets, verbose=0)
	best_next_actions = next_Q_values.argamax(axis=1)
	next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
	max_next_Q_values = (target.predict(next_states, verbose=0) * next_mask
				).sum(axis=1)
	[...]
```

## Prioritized Experience Replay

- Importance sampling (IS) or prioritized experience replay (PER)
	- Instead of sample uniformly from the replay buffer, sample important experience more frequently
- Measure the magnitude of the TD error, $\delta = r + \gamma \cdot V(s') - V(s)$
- A large TD error indicates that a transition is surprising, and worth learning from
- When it is sampled, the TD error is computed, and this experience's priority is set to $p = |\delta|$ (small value)
- Since the samples will be biased toward important experiences, compensate for this bias during training by down weighting the experiences according to their importance, or else the model will overfit the important experiences
- Important samples should be sampled more, but will have lower weight during training


## Dueling DQN

- Dueling DQN
	- Q-value of state-action pair => $Q(s, a) = V(s) + A(s, a)$
		- $V(s)$ = value of state s
		- $A(s, a)$ = advantage of taken the action a in state s
- The model estimates both the value of the state and the advantage of each possible action
- Best action should hae an advantage of 0, the model subtracts the max. pred. advatnage from all predicted advantages

```python
input_state = tf.keras.layers.Input(shape=[4])
hidden1 = tf.keras.layers.Dense(32, activation="relu")(input_state)
hidden2 = tf.keras.layers.Dense(32, activation="relu")(hidden1)
state_values = tf.keras.layers.Dense(1)(hidden1)
raw_advantages = tf.keras.layers.Dense(n_outputs)(hidden2)
advantages = raw_advantages - tf.reduce_max(raw_advantages, axis=1,
			keepdims=True)
Q_values = state_values + advantages
model = tf.keras.Model(inputs=[input_states], outputs=[Q_values])
```

- Rainbow
	- Combines 6 techniques into an agent

# Overview of Some Popular RL Algorithms

- AlphaGo
- Actor-critic algorithms
- Asynchronous advantage actor-critic (A3C)
- Advantage actor-critic (A2C)
- Soft actor-critic (SAC)
- Proximal policy optimization (PPO)
- Curiosity-based exploration
- Open-ended learning (OEL)