'''
A model-free Reinforcement learning agent, created in Tensorflow.
A part of my steps into the world of AI -- Created to go with MIT 6.S191 Lab 3
'''

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import gym
import numpy as np
from memory import Memory
from pyvirtualdisplay import Display
import skvideo.io
import __init__ as util
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import time

env = gym.make("CartPole-v0")
env.seed(1)
n_actions = env.action_space.n

def create_cartpole_model():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=32, activation='relu'), tf.keras.layers.Dense(units=n_actions, activation=None)])
    return model

cartpole_model = create_cartpole_model()

def choose_action(model, observation):
    observation = observation.reshape([1, -1])
    logits = model.predict(observation)
    prob_weights = tf.nn.softmax(logits).numpy()
    action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]
    return action

memory = Memory()

def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x

def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return normalize(discounted_rewards)

learning_rate = 1e-3
optimizer = tf.train.AdamOptimizer(learning_rate)
def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss

def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        observations = tf.convert_to_tensor(observations, dtype=tf.float32)
        logits = model(observations)

        loss = compute_loss(logits, actions, discounted_rewards)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

smoothed_reward = util.LossHistory(smoothing_factor=0.9)
plotter = util.PeriodicPlotter(sec=5, xlabel='Iterations', ylabel='Rewards')
for i_episode in range(10000):
    plotter.plot(smoothed_reward.get())
    observation = env.reset()
    while True:
        action =  choose_action(cartpole_model, observation)
        next_obs, reward, done, info = env.step(action)
        memory.add_to_memory(observation, action, reward)
        if done:
            total_reward = sum(memory.rewards)
            smoothed_reward.append(total_reward)
            train_step(cartpole_model, optimizer, observations=np.vstack(memory.observations), actions=np.array(memory.actions), discounted_rewards=discount_rewards(memory.rewards))
            memory.clear()
            break
        observation = next_obs

def save_video_of_model(model, env_name, filename='agent.mp4'):
    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(40, 30))
    display.start()

    env = gym.make(env_name)
    obs = env.reset()
    shape = env.render(mode='rgb_array').shape[0:2]

    out = skvideo.io.FFmpegWriter(filename)

    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        out.writeFrame(frame)
        action = model(tf.convert_to_tensor(obs.reshape((1, -1)), tf.float32)).numpy().argmax()
        obs, reward, done, info = env.step(action)
    out.close()
    print("successfully saved into {}".format(filename))

save_video_of_model(cartpole_model, "CartPole-v0")
