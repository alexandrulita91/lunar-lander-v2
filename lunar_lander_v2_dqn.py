"""
LunarLander-v2 -- Deep Q-learning with Experience Replay
"""
import os
import random
from collections import deque

import gym
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam


class Agent:
    def __init__(self, state_size, action_size, batch_size=32, memory_size=50000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.tau = 1000
        self.update_frequency = 1
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self):
        if self.batch_size > len(self.memory):
            return

        # Randomly sample a batch from the memory
        random_batch = random.sample(self.memory, self.batch_size)

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = random_batch[i][0]
            action.append(random_batch[i][1])
            reward.append(random_batch[i][2])
            next_state[i] = random_batch[i][3]
            done.append(random_batch[i][4])

        # Batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.target_model(next_state)

        for i in range(len(random_batch)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state),
            np.array(target),
            batch_size=self.batch_size,
            verbose=0
        )

    def load_weights(self, weights_file):
        self.epsilon = self.epsilon_min
        self.model.load_weights(weights_file)

    def save_weights(self, weights_file):
        self.model.save_weights(weights_file)


if __name__ == "__main__":
    # Flag used to enable or disable screen recording
    recording_is_enabled = True

    # Initializes the environment
    env = gym.make('LunarLander-v2')

    # Records the environment
    if recording_is_enabled:
        env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: True, force=True)

    # Defines training related constants
    num_episodes = 5000
    num_episode_steps = env.spec.max_episode_steps  # constant value
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    total_steps = 0

    # Creates an agent
    agent = Agent(state_size=state_size, action_size=action_size)

    # Loads the weights
    if os.path.isfile("lunar_lander-v0.h5"):
        agent.load_weights("lunar_lander-v0.h5")

    for episode in range(num_episodes):
        # Defines the total reward per episode
        total_reward = 0

        # Resets the environment
        observation = env.reset()

        # Gets the state
        state = np.reshape(observation, [1, state_size])

        for episode_step in range(num_episode_steps):
            # Renders the screen after new environment observation
            env.render(mode="human")

            # Gets a new action
            action = agent.act(state)

            # Takes action and calculates the total reward
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            # Gets the next state
            next_state = np.reshape(observation, [1, state_size])

            # Memorizes the experience
            agent.memorize(state, action, reward, next_state, done)

            # Updates the slow target's weights
            if total_steps % agent.tau == 0:
                agent.update_target_network()

            # Updates network weights
            if total_steps % agent.update_frequency == 0:
                agent.replay()

            # Updates the state
            state = next_state

            if done:
                print("Episode %d/%d finished after %d episode steps with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))
                break

            elif episode_step >= num_episode_steps - 1:
                print("Episode %d/%d timed out at %d with total reward = %f."
                      % (episode + 1, num_episodes, episode_step + 1, total_reward))

        # Update epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Saves the weights
        agent.save_weights("lunar_lander-v0.h5")

        total_steps += 1

    # Closes the environment
    env.close()
