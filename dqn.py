#!/usr/bin/env python3

"""
Use apprenticeship learning to make a game-playing bot.
"""

import sys
import random
import gym
import numpy as np
import time as tm
import argparse
import os
from os.path import join as pjoin
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt


class DQLearner:
    """
    Abstract base class for deep Q-learners.
    Override the method build_nnet() to get a concrete class.
    """

    def __init__(self):
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.alpha = 0.01
        self.alpha_decay = 0.01
        self.batch_size = 64

    def build_nnet():
        raise NotImplementedError('You must override the build_nnet method.')

    def memorize(self, state1, action, reward, state2, done):
        self.memory.append((
            state1, #state1.reshape((1,) + state1.shape),
            action, reward,
            state2, #state2.reshape((1,) + state2.shape),
            done))

    def decide(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            qs = self.model.predict(state.reshape((1,) + state.shape))[0]
            return np.argmax(qs)

    def replay(self, epochs):
        minibatch = random.sample(list(self.memory), self.batch_size)
        X = np.array([tup[0] for tup in minibatch])
        Y = self.model.predict(X)
        for i, (state1, action, reward, state2, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(state2.reshape((1,) + state2.shape))[0])
            Y[i][action] = target
            #self.model.train_on_batch(state1, target_f)
        self.model.fit(X, Y, epochs=epochs, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class SimpleLearner(DQLearner):
    """
    Deep Q-learner which learnes from single-dimensional data using an MLP.
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

    def build_nnet(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(3*self.state_size, activation='relu', input_dim=self.state_size))
        model.add(Dense(4*self.state_size, activation='relu'))
        model.add(Dense(3*self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        """
        model.add(Dense(6*self.state_size, input_dim=4, activation='tanh'))
        model.add(Dense(12*self.state_size, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        """

        model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        self.model = model
        return model


def play_game(env, agent, learn=False, render=False, epochs=1):
    done = False
    state = env.reset()
    score = 0
    time = 0
    while not done:
        if render:
            env.render()
        old_state = state
        action = agent.decide(state)
        state, reward, done, _ = env.step(action)
        reward = reward if not done else reward-10
        score += reward
        time += 1
        if learn:
            agent.memorize(old_state, action, reward, state, done)
            if (time+1) % args.replay_interval == 0 and len(agent.memory) > agent.batch_size:
                agent.replay(epochs)
    if learn:
        if len(agent.memory) > agent.batch_size:
            agent.replay(epochs)

    return score, time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game', help='Name of an OpenAI gym environment')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--show-games', type=int, default=0,
        help='The number of games the bot should play after training')
    parser.add_argument('--replay-interval', type=int, default=50,
        help='The number of actions the bot should take between successive memory replays')
    parser.add_argument('--appr-epochs', type=int, default=1)
    parser.add_argument('--learn-epochs', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--score-out', help='Path to text file to output scores')
    parser.add_argument('appr_data', nargs='*',
        help='Path to apprenticeship data, in the format output by OpenAIGaming')
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize and environment and learner
    env = gym.make(args.game)
    env.seed(args.seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = SimpleLearner(state_size, action_size)
    agent.build_nnet()
    print(file=sys.stderr)

    # Load and memorize apprenticeship data
    for dirpath in args.appr_data:
        print('Learning from', dirpath)
        actions = np.load(pjoin(dirpath, 'actions.npy'))
        rewards = np.load(pjoin(dirpath, 'rewards.npy'))
        states = np.load(pjoin(dirpath, 'states.npy'))
        sz = len(actions)
        for i in range(sz):
            agent.memorize(states[i], actions[i], rewards[i], states[i+1], i == sz - 1)
            if (i+1) % args.replay_interval == 0 and len(agent.memory) > agent.batch_size:
                agent.replay(args.appr_epochs)
        if len(agent.memory) > agent.batch_size:
            agent.replay(args.appr_epochs)

    # agent.load("./save/cartpole-dqn.h5")
    start_time = tm.time()

    scores = []
    times = []

    print(file=sys.stderr)
    try:
        for e in range(args.episodes):
            score, time = play_game(env, agent, learn=True, epochs=args.learn_epochs)
            scores.append(score)
            times.append(time)
            line = "episode: {}, score: {}, e: {:.2}".format(e, score, agent.epsilon)
            print(line, file=sys.stderr, end='\n')
            # agent.save("./save/cartpole-dqn.h5")
    except KeyboardInterrupt:
        pass 
    print(file=sys.stderr)
    train_time = tm.time() - start_time
    print('Training time = {} seconds'.format(train_time))

    # output scores
    if args.score_out is not None:
        with open(args.score_out, 'w') as fobj:
            print(*scores, sep='\n', file=fobj)

    score_av_size = 100
    av_scores = [sum(scores[:score_av_size])]
    for i in range(score_av_size, len(scores)):
        av_scores.append(av_scores[-1] - scores[i-score_av_size] + scores[i])
    for i in range(len(av_scores)):
        av_scores[i] /= score_av_size

    # plot scores
    x = list(range(len(scores)))
    plt.plot(x, scores)
    plt.show()
    x = list(range(len(av_scores)))
    plt.plot(x, av_scores)
    plt.show()

    # play game
    for e in range(args.show_games):
        score, time = play_game(env, agent, render=True)
        print('Score =', score)
