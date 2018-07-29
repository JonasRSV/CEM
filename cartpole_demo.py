import gym
import sys
import numpy as np
import tensorflow as tf
import ce

ACTION_SPACE = 2
STATE_SPACE  = 4

ENV = 'CartPole-v0'

EPOCHS       = 10
ENV_ITER     = 10
MAX_ENV_STEP = 200

def train(env, actor):

    actor.summarize_sp()
    for e in range(EPOCHS):
        fitnesses = []
        for a in range(len(actor)):
            fitness = 0
            for _ in range(ENV_ITER):
                s = env.reset()

                for _ in range(MAX_ENV_STEP):
                    env.render()

                    action = actor.pred_sp(s.reshape(-1, 1), a)
                    action = action.reshape(-1)
                    action = np.argmax(action)

                    s, r, terminal, _ = env.step(action)

                    fitness += r

                    if terminal:
                        break

            fitness /= ENV_ITER
            fitnesses.append(fitness)

        actor.train_sp(fitnesses)
        actor.summary_add_scalar("Fitness", max(fitnesses))
        actor.summarize_sp()


    print("Running final")
    play(env, actor, 5)


def play(env, actor, games=20):
    apex = actor.get_apex()

    print("Playing")
    score = 0
    for i in range(games):
        terminal = False
        s = env.reset()

        while not terminal:
            env.render()
            action = actor.pred_sp(s.reshape(-1, 1), apex)
            action = action.reshape(-1)
            action = np.argmax(action)

            s, r, terminal, _ = env.step(action)

            score += r

    if (score / games) > 180:
        print("WIN!")

    print(score / games)

    env.close()


if __name__ == "__main__":

    env   = gym.make(ENV)
    actor = ce.CE(STATE_SPACE, ACTION_SPACE, agents=100, inheritance=0.1)
    with tf.Session() as sess:

        actor.spi(sess)
        saver = tf.train.Saver()

        try:
            if "-t" in sys.argv:
                sess.run(tf.global_variables_initializer())
                train(env, actor)
        except KeyboardInterrupt:
            pass

        if "-p" in sys.argv:
            saver.restore(sess, "model/cem.model")
            print("Restored..")
            play(env, actor)

        saver.save(sess, "model/cem.model")
