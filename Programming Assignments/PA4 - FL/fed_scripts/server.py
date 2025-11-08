"""
Server class for Federated Learning
Handles client selection, broadcasting, and aggregation
"""

import torch
import numpy as np
import copy
from scriptsfl.federated_utils import get_model_weights, set_model_weights, aggregate_weights, evaluate_model


class Server:
    """
    Federated Learning Server

    Orchestrates the federated training process
    """

    def __init__(self, global_model, clients, test_loader, device='cpu'):
        """
        Initialize server

        Args:
            global_model: PyTorch model
            clients: list of Client objects
            test_loader: DataLoader for test set
            device: device to run evaluation on
        """
        self.global_model = global_model.to(device)
        self.clients = clients
        self.test_loader = test_loader
        self.device = device

        # Training history
        self.history = {
            'rounds': [],
            'test_accuracy': [],
            'test_loss': [],
            'train_loss': [],
            'selected_clients': [],
            'divergence': []
        }

    def select_clients(self, fraction=1.0, num_clients=None):
        """
        Randomly select clients for this round

        Args:
            fraction: fraction of clients to select (0 < fraction <= 1)
            num_clients: exact number of clients to select (overrides fraction)

        Returns:
            list of selected Client objects
        """
        if num_clients is None:
            num_clients = max(1, int(len(self.clients) * fraction))

        selected = np.random.choice(self.clients, size=num_clients, replace=False)
        return list(selected)

    def broadcast_weights(self, clients=None):
        """
        Send global model weights to clients

        Args:
            clients: list of clients (default: all clients)
        """
        if clients is None:
            clients = self.clients

        global_weights = get_model_weights(self.global_model)

        for client in clients:
            client.set_weights(global_weights)

    def aggregate(self, clients):
        """
        Aggregate client models into global model

        Args:
            clients: list of Client objects that participated

        Returns:
            average divergence of client models from global
        """
        # Get weights from all clients
        client_weights = [client.get_weights() for client in clients]
        client_sizes = [client.get_data_size() for client in clients]

        # Aggregate
        aggregated_weights = aggregate_weights(client_weights, client_sizes)

        # Update global model
        set_model_weights(self.global_model, aggregated_weights)

        # Compute divergence (optional, for analysis)
        from scriptsfl.federated_utils import compute_weight_divergence
        divergence = compute_weight_divergence(client_weights, aggregated_weights)

        return divergence

    def evaluate(self):
        """
        Evaluate global model on test set

        Returns:
            accuracy, loss
        """
        accuracy, loss = evaluate_model(self.global_model, self.test_loader, self.device)
        return accuracy, loss

    def train_round(self, local_epochs, lr, client_fraction=1.0, momentum=0.9):
        """
        Execute one round of federated training

        Args:
            local_epochs: number of local epochs per client
            lr: learning rate
            client_fraction: fraction of clients to select
            momentum: SGD momentum

        Returns:
            dict with round statistics
        """
        # Select clients
        selected_clients = self.select_clients(fraction=client_fraction)

        # Broadcast global model
        self.broadcast_weights(selected_clients)

        # Local training
        client_stats = []
        for client in selected_clients:
            stats = client.train_local(epochs=local_epochs, lr=lr, momentum=momentum)
            client_stats.append(stats)

        # Aggregate
        divergence = self.aggregate(selected_clients)

        # Evaluate global model
        test_acc, test_loss = self.evaluate()

        # Compute average training loss
        avg_train_loss = np.mean([s['final_loss'] for s in client_stats])

        round_stats = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'train_loss': avg_train_loss,
            'divergence': divergence,
            'num_clients': len(selected_clients),
            'client_ids': [c.client_id for c in selected_clients]
        }

        return round_stats

    def train(self, num_rounds, local_epochs, lr, client_fraction=1.0, momentum=0.9, verbose=True):
        """
        Train for multiple rounds

        Args:
            num_rounds: number of communication rounds
            local_epochs: number of local epochs per round
            lr: learning rate
            client_fraction: fraction of clients to select each round
            momentum: SGD momentum
            verbose: whether to print progress

        Returns:
            training history dict
        """
        for round_idx in range(num_rounds):
            round_stats = self.train_round(local_epochs, lr, client_fraction, momentum)

            # Update history
            self.history['rounds'].append(round_idx + 1)
            self.history['test_accuracy'].append(round_stats['test_accuracy'])
            self.history['test_loss'].append(round_stats['test_loss'])
            self.history['train_loss'].append(round_stats['train_loss'])
            self.history['selected_clients'].append(round_stats['client_ids'])
            self.history['divergence'].append(round_stats['divergence'])

            if verbose:
                print(f"Round {round_idx + 1}/{num_rounds} | "
                      f"Test Acc: {round_stats['test_accuracy']:.2f}% | "
                      f"Test Loss: {round_stats['test_loss']:.4f} | "
                      f"Train Loss: {round_stats['train_loss']:.4f} | "
                      f"Divergence: {round_stats['divergence']:.4f}")

        return self.history
