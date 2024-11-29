import tensorflow as tf
import torch
import logging
from datetime import date

today = date.today()
formatted_date = today.strftime("%d-%m-%Y")

logging.basicConfig(filename=f'Logs/Pong/{formatted_date}_version0.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def preprocess_observation(obs, new_size=[64,64]):
    """
        Resize frames: (px_height, px_width)
        Normalize pixel values [0, 1]
        
    """
    
    if(not torch.is_tensor(obs)):
        log = f'obs: {type(obs)} converted to tensor'
        log = log + f' obs.shape: {obs.shape}'
        obs_tensor = torch.from_numpy(obs)
        log = log + f' obs_tensor.shape: {obs_tensor.shape}'
        obs = obs_tensor
        logging.info(log)
    
    
    logging.info(f'START preprocessing...new size: {new_size}')

        
    #mean_obs = np.mean(obs, axis=-1)

    resized_obs = tf.image.resize(obs, new_size).numpy()
    
    normalized_obs = resized_obs/255.0
    
    logging.info("...COMPLETED preprocessing")
    return normalized_obs


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