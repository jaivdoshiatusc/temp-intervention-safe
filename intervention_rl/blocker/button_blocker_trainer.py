import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

class BlockerModel(nn.Module):
    def __init__(self, obs_size=60, action_dim=2, hidden1=64, hidden2=32, output_classes=3):
        super(BlockerModel, self).__init__()
        self.fc1 = nn.Linear(obs_size + action_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.Linear(hidden2, output_classes)

    def forward(self, obs, action):
        # obs shape: (batch_size, 60)
        # action shape: (batch_size, 2)
        x = torch.cat([obs, action], dim=1)  # shape: (batch_size, 62)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)  # shape: (batch_size, 3) for 3 categories

class BlockerDataset(Dataset):
    def __init__(self, observations, actions, labels):
        """
        observations: List (or np.array) of shape (N, 60)
        actions: List (or np.array) of shape (N, 2)  (continuous)
        labels: List/array of shape (N,) with values in {0, 1, 2}
        """
        self.observations = observations
        self.actions = actions
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        obs = torch.tensor(self.observations[idx], dtype=torch.float32)
        act = torch.tensor(self.actions[idx], dtype=torch.float32)  # shape (2,)
        lab = torch.tensor(self.labels[idx], dtype=torch.long)      # 0, 1, or 2
        return obs, act, lab

class ButtonBlockerTrainer:
    def __init__(self, device='cpu', obs_size=60, action_dim=2, num_classes=3, lr=1e-3):
        self.device = device
        self.model = BlockerModel(
            obs_size=obs_size,
            action_dim=action_dim,
            output_classes=num_classes
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.observations = []
        self.actions = []
        self.labels = []

        self.positive_labels = 0
        self.negative_labels = 0

    def map_to_label(self, array_label):
        if array_label == [2, 2]:
            return 0   # 'none'
        elif array_label == [-1, 0]:
            return 1   # 'backward'
        elif array_label == [1, 0]:
            return 2   # 'forward'
        else:
            raise ValueError("Unrecognized label array.")

    def map_from_label(self, int_label):
        # Reverse mapping (for debugging or interpretability):
        if int_label == 0:
            return [2, 2]
        elif int_label == 1:
            return [-1, 0]
        elif int_label == 2:
            return [1, 0]
        else:
            raise ValueError("Unrecognized integer label.")

    def store(self, obs, action, label):
        converted_label = self.map_to_label(label)
        if converted_label != 0:
            self.positive_labels += 1
        else:
            self.negative_labels += 1

        self.observations.append(obs)
        self.actions.append(action)
        self.labels.append(converted_label)

    def train(self, epochs=4, batch_size=32):
        if len(self.labels) == 0:
            return  # No data to train on

        # Create dataset and dataloader
        dataset = BlockerDataset(self.observations, self.actions, self.labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(epochs):
            for batch_obs, batch_actions, batch_labels in dataloader:
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_obs, batch_actions)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

    def should_block(self, obs, action, expert_label=None):    
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(obs_tensor, action_tensor)  # shape: (1, 3)
            prob = F.softmax(output, dim=-1)    # shape: (1, 3)
            prob = torch.clamp(prob, min=1e-6)  # numerical stability
            entropy = -(prob * prob.log()).sum().item()

            # The predicted label is 0,1, or 2
            blocker_model_decision = torch.argmax(prob, dim=-1).item()

            # If we have an expert label, compute "disagreement_prob"
            converted_expert_label = self.map_to_label(expert_label)
            if converted_expert_label is not None:
                # Probability that the model gives to the expert’s class
                model_prob_for_expert = prob[0, converted_expert_label].item()
                # Disagreement = everything else
                disagreement_prob = 1.0 - model_prob_for_expert
            else:
                disagreement_prob = 0.0

        converted_label = self.map_from_label(blocker_model_decision)

        return converted_label, entropy, disagreement_prob

    def get_labels(self):
        return self.positive_labels, self.negative_labels