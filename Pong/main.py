import gymnasium as gym
import ale_py
import ViTDQN

import DQN
import ReplayBuffer
import Agent


import numpy as np
from collections import deque
import random
from random import sample
import torch
import matplotlib
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
import base64

from pathlib import Path
import os

import tensorflow as tf
import functions as f


if __name__ == '__main__':
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
    
    num_actions = env.action_space.n
    
    model = ViTDQN.ViTDQN(ViTDQN.vit_model, num_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    state = f.preprocess_observation(env.reset()[0])

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
        next_state = f.preprocess_observation(next_state)
        state = next_state
        i = i + 1
        print(total_reward)
        
        
    env.close()

