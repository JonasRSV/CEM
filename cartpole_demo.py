import gym
import sys
import numpy as np
import tensorflow as tf
import cross_entropy as ce

ACTION_SPACE = 2
STATE_SPACE  = 4

ENV = 'CartPole-v0'

EPOCHS       = 1000
ENV_ITER     = 4
MAX_ENV_STEP = 100

def train(env, actor):

    for e in range(EPOCHS):
        for a in range(len(actor)):
            for _ in range(ENV_ITER):
                s0 = env.reset()

                for _ in range(MAX_ENV_STEP):
                    env.render()
                    action = actor(a, s0)
                    s2, r2, terminal, _ = env.step(action)
                    s1 = s2

                    if terminal:
                        break

    env.close()


def play(env, actor, games=20):
    for i in range(games):
        terminal = False
        s0 = env.reset()


        while not terminal:
            env.render()
            s0 = s0.reshape(1, -1)
            action = actor.predict(s0)[0]

            s0, _, terminal, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    env   = gym.make(ENV)
    actor = ce.cross_entropy(STATE_SPACE, ACTION_SPACE)

    if "-t" in sys.argv:
        train(env, actor)

    if "-p" in sys.argv:
        play(env, actor)
