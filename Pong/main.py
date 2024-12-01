import gymnasium as gym
import ale_py
import ViTDQN as vit
import Agent
import ReplayBuffer
import Agent
import numpy as np
import random
import torch
import matplotlib
import base64
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from random import sample
from pathlib import Path

def preprocess_observation(obs, new_size=[64,64]):
    """
        Resize frames: (px_height, px_width)
        Normalize pixel values [0, 1]
        
    """
    
    if(not torch.is_tensor(obs)):
        obs_tensor = torch.from_numpy(obs)
        obs = obs_tensor

    resized_obs = tf.image.resize(obs, new_size).numpy()
    
    normalized_obs = resized_obs/255.0
    
    return normalized_obs


if __name__ == '__main__':
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
    max_episodes = 100
    num_actions = env.action_space.n
        
    model = vit.ViTDQN(vit.vit_model, num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    agent = Agent.Agent(num_actions, 64, 0.01, 0.001)
    
    
    state = preprocess_observation(env.reset()[0])
    done = False
    total_reward = 0
    i = 0
    while not done:
        print(f'i: {i}')
        state_tensor = torch.tensor(state).unsqueeze(0)
        state_tensor = state_tensor.permute(0,3,1,2)
        #f.get_description(state_tensor)
        q_values = model(state_tensor)
        print(q_values)
        action = torch.argmax(q_values).item()
        next_state, reward, done, a, b  = env.step(action)
        total_reward += reward
        next_state = preprocess_observation(next_state)
        state = next_state
        i = i + 1
        print(total_reward)
        
        
    env.close()

