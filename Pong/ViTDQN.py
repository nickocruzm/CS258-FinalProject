import numpy as np
from skimage.transform import resize
from transformers import ViTModel, ViTConfig
import torch
import torch.nn as nn
import logging




# Configure a Vision Transformer
config = ViTConfig(image_size=64, patch_size=16, num_channels=3, num_labels=256)
vit_model = ViTModel(config)


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
        logging.info(f'...forward called... \t x: {type(x)}, {x.shape}')
        embeddings = self.vit(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token
        return self.fc(embeddings)


