from collections import deque
import random

class ReplayBuffer:

    def __init__(self, bufferCap):
        self.memory = deque(maxlen=bufferCap)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        """
            Grabs a random batch of experiences from memory.
            batch_size: number of experiences to sample.
        """

        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)