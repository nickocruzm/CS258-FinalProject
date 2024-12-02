import gymnasium as gym
import ale_py
import ViTDQN as vit
import Agent
import ReplayBuffer
import Agent


import numpy as np
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
import logging

from datetime import date

today = date.today()
formatted_date = today.strftime("%d-%m-%Y")

logging.basicConfig(filename=f'Pong/{formatted_date}_version0.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
    
    max_episodes = 100
    num_actions = env.action_space.n
    
    # model = vit.ViTDQN(vit.vit_model, num_actions)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    
    done = False
    total_reward = 0
    i = 0
    
    means = []
    logging.info("...COMPLETED preprocessing")
    TARGET_UPDATE_FREQ = 100
    EPSILON_DECAY = 0.0001
    EPSILON = 0.1
    EPSILON_END = 0.000001
    BUFFER_SIZE = 100
    BATCH_SIZE = 64 
    LR = 0.0001
    GAMMA = 0.99
    total_steps  = 0
    
    agent = Agent.Agent(env, num_actions, BUFFER_SIZE, BATCH_SIZE, EPSILON, LR, GAMMA)
    
    for episode in range(0,max_episodes):
        state = f.preprocess_observation(env.reset()[0])
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state,env)
            
            next_state, reward, done, truncated, info = env.step(action)
            next_state = f.preprocess_observation(next_state)
            
            agent.memory.push((state, action, reward, next_state, float(done)))
            
            total_reward += reward
            agent.train()

            total_steps += 1
            print(f'episode: {episode}, total_steps: {total_steps}, action: {action}, total reward: {total_reward}')
            
            if total_steps % TARGET_UPDATE_FREQ == 0:
                print("...updating network...")
                agent.update_target_network()
            
            if(total_reward < 0):
                break
            
            state = next_state 
      
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, epsilon: {agent.epsilon}")

        mean, std = agent.evaluate(env)
        means.append(mean)
        
    logging.info(f'episode: {episode} completed')
        


