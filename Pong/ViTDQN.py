import numpy as np
from skimage.transform import resize
from transformers import ViTModel, ViTConfig
import torch
import torch.nn as nn
import logging
from datetime import date
import os


today = date.today()
formatted_date = today.strftime("%m-%d-%Y")


log_file_path = f'{os.getcwd()}/glob_{formatted_date}.log'

# Configure a Vision Transformer
config = ViTConfig(image_size=210, patch_size=16, num_channels=3)
vit_model = ViTModel(config)


logging.basicConfig(filename=log_file_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class ViTDQN(nn.Module):

    def __init__(self, vit_model, num_actions):
        super(ViTDQN, self).__init__()
        self.vit = vit_model
        self.fc = nn.Sequential(
            nn.Linear(vit_model.config.hidden_size, 210),
            nn.ReLU(),
            nn.Linear(210, num_actions)
        )

    def forward(self, x):
        logging.info(f'VIT: forward()')
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        
        embeddings = self.vit(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token
        return self.fc(embeddings)


