# encoding: utf-8

import gym
import random
import os
import numpy as np
import statistics
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self):
        """
        Construye el modelo o lo carga desde un archivo h5
        :return:
        """
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        """
        Guarda el modelo
        :return:
        """
        self.brain.save(self.weight_backup)

    def act(self, state):
        """
        Recibe un estado y regresa una accion
        :param state: un vector con shape (1, 4)
        :return:
        """
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        """
        Guardar una transaccion en replay memory
        :param state: las 4 variables que representan el estado en cartpole
            [[-0.02476985 -0.04408725 -0.0043659   0.03441053]]
        :param action: el indice de la accion
            0 o 1
        :param reward: la recompensa
            1.0
        :param next_state: el nuevo estado
            [[-0.02565159  0.15109704 -0.00367769 -0.25964668]]
        :param done: si el episodio termino
            True o False
        :return:
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            # entrenar: este estado (st ate) debe tener este valor (target_f)
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 100
        self.env = gym.make('CartPole-v1')

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            sample_results = []
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
                    #                    self.env.render()

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                # print("Episode {}# Score: {}".format(index_episode, index + 1))
                sample_results.append(index + 1)
                if index_episode > 0 and index_episode % 5 == 0:
                    print(index_episode)
                    print(index_episode % 5)
                    print("Episode {}# Avg Score: {}, Min: {}, Max: {} ".format(index_episode, sum(sample_results) / len(sample_results), min(sample_results), max(sample_results)))
                    sample_results = []


                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
