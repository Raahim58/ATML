"""
Client class for Federated Learning
Handles local training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from scriptsfl.federated_utils import get_model_weights, set_model_weights


class Client:
    """
    Federated Learning Client

    Handles local training with vanilla SGD (can be extended for other methods)
    """

    def __init__(self, client_id, data_loader, model, device='cpu'):
        """
        Initialize client

        Args:
            client_id: unique client identifier
            data_loader: DataLoader for this client's data
            model: PyTorch model (will be copied)
            device: device to train on
        """
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device

        # Create a local copy of the model
        self.model = copy.deepcopy(model).to(device)

        # Data size
        self.data_size = len(data_loader.dataset)

    def train_local(self, epochs, lr, momentum=0.9):
        """
        Train locally using vanilla SGD

        Args:
            epochs: number of local epochs
            lr: learning rate
            momentum: SGD momentum

        Returns:
            dict with training stats
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        epoch_losses = []

        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * data.size(0)

            epoch_loss = running_loss / self.data_size
            epoch_losses.append(epoch_loss)

        stats = {
            'client_id': self.client_id,
            'epochs': epochs,
            'final_loss': epoch_losses[-1],
            'epoch_losses': epoch_losses
        }

        return stats

    def get_weights(self):
        """
        Get current model weights

        Returns:
            list of tensors
        """
        return get_model_weights(self.model)

    def set_weights(self, weights):
        """
        Set model weights

        Args:
            weights: list of tensors
        """
        set_model_weights(self.model, weights)

    def get_data_size(self):
        """
        Get number of training samples

        Returns:
            int: number of samples
        """
        return self.data_size
