import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2



# Preprocess observations (convert to grayscale, resize, normalize)
def preprocess_observation(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = np.array(obs, dtype=np.float32) / 255.0
    return obs

# Stack frames to give the agent a sense of motion
class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames, axis=0)

    def append(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis=0)

# Define the CNN Q-network
class CNN_QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(CNN_QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
num_episodes = 1000
batch_size = 32
gamma = 0.99
learning_rate = 1e-4
replay_memory_size = 100000
target_update_frequency = 1000  # Steps
initial_epsilon = 1.0
final_epsilon = 0.1
epsilon_decay = 1000000  # Steps
frame_stack_size = 4
update_frequency = 4  # Steps between gradient updates

# Create the environment
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
num_actions = env.action_space.n

# Initialize the policy and target networks
policy_net = CNN_QNetwork(num_inputs=frame_stack_size, num_actions=num_actions).to(device)
target_net = CNN_QNetwork(num_inputs=frame_stack_size, num_actions=num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Replay memory
memory = deque(maxlen=replay_memory_size)

# Epsilon for epsilon-greedy policy
epsilon = initial_epsilon
epsilon_decay_step = (initial_epsilon - final_epsilon) / epsilon_decay
steps_done = 0

def select_action(state):
    global steps_done, epsilon
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

# Training loop
total_rewards = []

for episode in range(num_episodes):
    obs_raw, _ = env.reset()
    obs_preprocessed = preprocess_observation(obs_raw)
    frame_stack = FrameStack(frame_stack_size)
    state = frame_stack.reset(obs_preprocessed)
    episode_reward = 0
    done = False

    while not done:
        action = select_action(state)
        obs_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs_preprocessed = preprocess_observation(obs_raw)
        next_state = frame_stack.append(obs_preprocessed)
        episode_reward += reward

        # Store transition in replay memory
        memory.append((state, action, reward, next_state, done))

        state = next_state
        steps_done += 1

        # Epsilon decay
        if epsilon > final_epsilon:
            epsilon -= epsilon_decay_step

        # Perform optimization step
        if steps_done % update_frequency == 0 and len(memory) >= batch_size:
            transitions = random.sample(memory, batch_size)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

            batch_state = torch.from_numpy(np.stack(batch_state)).float().to(device)
            batch_action = torch.tensor(batch_action).long().to(device)
            batch_reward = torch.tensor(batch_reward).float().to(device)
            batch_next_state = torch.from_numpy(np.stack(batch_next_state)).float().to(device)
            batch_done = torch.tensor(batch_done).float().to(device)

            # Compute Q(s_t, a)
            q_values = policy_net(batch_state)
            state_action_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)

            # Compute V(s_{t+1}) for all next states using Double DQN
            with torch.no_grad():
                next_q_values_policy = policy_net(batch_next_state)
                next_actions = next_q_values_policy.argmax(dim=1)
                next_q_values_target = target_net(batch_next_state)
                next_state_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                expected_state_action_values = batch_reward + (gamma * next_state_values * (1 - batch_done))

            # Compute loss
            loss = criterion(state_action_values, expected_state_action_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        # Update the target network
        if steps_done % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

    total_rewards.append(episode_reward)
    if episode % 10 == 0:
        avg_reward = np.mean(total_rewards[-10:])
        print(f"Episode {episode}, average reward (last 10 episodes): {avg_reward}, epsilon: {epsilon:.3f}")

env.close()
