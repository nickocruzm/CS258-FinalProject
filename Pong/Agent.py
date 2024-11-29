import ViTDQN as vit
import functions as f
import ReplayBuffer as RB
import torch
import torch.optim as optim

# if torch.backends.mps.is_available():
#     device = torch.device("mps")  # Use Metal Performance Shaders
# else:
#     device = torch.device("cpu")  # Fallback to CPU
# print(f"Using device: {device}") 

class Agent:
    def __init__(self, action_size, buffer_size, eps, LR):
        self.epsilon = eps
        self.action_size = action_size
        self.memory = RB.ReplayBuffer(buffer_size)
        self.Q_network = vit.ViTDQN(vit.vit_model, action_size)
        self.Target_network = vit.ViTDQN(vit.vit_model,action_size)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=LR)
        self.update_target_network()

    def update_target_network(self):
        self.Target_network.load_state_dict(self.Q_network.state_dict())

    def greedy_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            q_values = self.Q_network(state)
            return q_values.argmax().item()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.Q_network(state)
                return q_values.argmax().item()

    def train(self):
        if len(self.memory) < BATCH_SIZE: return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(np.array(batch[0]))
        actions = torch.LongTensor(batch[1]).unsqueeze(1)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(batch[3]))
        dones = torch.FloatTensor(batch[4]).unsqueeze(1)

        predicted_Q = self.Q_network(states).gather(1, actions)
        next_q = self.Target_network(next_states).max(1)[0].detach().unsqueeze(1)

        # dones tensor contains done flags that are stored as 1, if episode ended (a terminal state has been reached or 0 if episode has not ended (terminal state not reached) to only consider the future reward when a terminal state is not reached.

        expected_Q = rewards + (GAMMA * next_q * (1 - dones))

        # Mean Squared Error Loss between predicted and expected Q-values
        loss = nn.MSELoss()(predicted_Q, expected_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, env, n_rollouts=1):
        """
        Returns the mean + std score obtained by executing the greedy policy for a fixed number of rollouts.

        Params
        ======
            env: the environment
            n_rollouts (int): the number of rollouts to be performed
        """
        rewards = []
        for _ in range(n_rollouts):
            state, done = env.reset(), False
            rewards.append(0)
            while not done:
                action = self.greedy_action(state)
                state, reward, done, _ = env.step(action)
                rewards[-1] += reward
        return np.mean(rewards), np.std(rewards)

    def save_checkpoint(self, filename):
        """
        Saves a model to a file

        Parameters
        ----------
        model: your Q network
        filename: the name of the checkpoint file
        """
        torch.save(self.Q_network.state_dict(), filename)

    def load_checkpoint(self, model_name):
        """
        Loads a previously trained model.

        Params
        ======
            model_name (str): the name of the trained model
        """

        self.Q_network.load_state_dict(torch.load(model_name))
