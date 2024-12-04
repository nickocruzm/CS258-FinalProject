import tensorflow as tf
import torch
import logging
from datetime import date
import numpy as np

today = date.today()
formatted_date = today.strftime("%d-%m-%Y")

logging.basicConfig(filename=f'Logs/Pong/{formatted_date}_version0.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def preprocess_observation(obs, new_size=[64, 64]):
    """
    Resize frames to new_size (e.g., [64, 64]) and normalize pixel values to [0, 1].
    Adds a batch dimension to the processed observation.
    """
    if isinstance(obs, torch.Tensor):
        obs = obs.numpy()
        logging.info(f"Converted PyTorch tensor to NumPy. New shape: {obs.shape}")
        
    elif not isinstance(obs, (tf.Tensor, np.ndarray)):
        obs = np.array(obs)
        logging.info(f"Converted input to NumPy array. Shape: {obs.shape}")

    resized_obs = tf.image.resize(obs, new_size).numpy()

    # Normalize pixel values to [0, 1]
    normalized_obs = resized_obs / 255.0


    batched_image = tf.expand_dims(normalized_obs, axis=0).numpy()
    print(f"Completed preprocessing. Final shape: {batched_image.shape}")

    # Return as a PyTorch tensor (if using PyTorch for modeling)
    return torch.from_numpy(batched_image).permute(0,3,1,2).float() 


def plot():
    mean, std = agent.evaluate(env)
    if best_agent is None or mean > best_score:
        best_score = mean
        best_agent = agent

    plt.figure(figsize=(6, 4))
    plt.plot(ao_10_score)
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title(f'Training Curve for Seed {seed} (Mean Reward: {mean:.2f} +/- {std:.2f})')
    plt.savefig(f'plot_s{seed}.png',format="png",dpi=300)
    plt.show()

    best_agent.save_checkpoint(f'drive/MyDrive/q1-model.s{seed}.pt')
    agent = best_agent

    agent = Agent(STATE_SIZE, ACTION_SIZE)


def get_description(tensor):
    print("Shape:", tensor.shape)             # Output: Shape: torch.Size([2, 3, 4])
    print("Number of dimensions:", tensor.ndimension())  # Output: 3
    print("Data type:", tensor.dtype)         # Output: torch.float32
    print("Device:", tensor.device)           # Output: cuda:0
    print("Requires grad:", tensor.requires_grad)  # Output: True
    print("Number of elements:", tensor.numel()) 