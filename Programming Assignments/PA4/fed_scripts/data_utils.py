"""
Data utilities for Federated Learning
Handles CIFAR-10/MNIST loading and IID/non-IID splits
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict


def load_dataset(dataset_name='cifar10', data_path='./data'):
    """
    Load CIFAR-10 or MNIST dataset
    
    Args:
        dataset_name: 'cifar10' or 'mnist'
        data_path: path to store/load data
        
    Returns:
        train_dataset, test_dataset
    """
    if dataset_name.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use 'cifar10' or 'mnist'")
    
    return train_dataset, test_dataset


def create_iid_split(dataset, num_clients):
    """
    Split dataset evenly (IID) across clients
    
    Args:
        dataset: PyTorch dataset
        num_clients: number of clients
        
    Returns:
        dict: {client_id: list of indices}
    """
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_size = num_samples // num_clients
    
    client_indices = {}
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < num_clients - 1 else num_samples
        client_indices[i] = indices[start_idx:end_idx].tolist()
    
    return client_indices


def create_dirichlet_split(dataset, num_clients, alpha, min_samples_per_client=10):
    """
    Create non-IID split using Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: number of clients
        alpha: concentration parameter (smaller = more skewed)
        min_samples_per_client: minimum samples each client should have
        
    Returns:
        dict: {client_id: list of indices}
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Initialize client indices
    client_indices = {i: [] for i in range(num_clients)}
    
    # For each class, sample proportions from Dirichlet and distribute
    for class_id in range(num_classes):
        indices = np.array(class_indices[class_id])
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()  # Normalize
        
        # Calculate number of samples per client for this class
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        
        # Split indices according to proportions
        splits = np.split(indices, proportions)
        
        # Assign to clients
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())
    
    # Ensure minimum samples per client
    for client_id in range(num_clients):
        if len(client_indices[client_id]) < min_samples_per_client:
            print(f"Warning: Client {client_id} has only {len(client_indices[client_id])} samples")
    
    return client_indices


def get_dataloaders(dataset, client_indices, batch_size=32, shuffle=True):
    """
    Create DataLoader for each client
    
    Args:
        dataset: PyTorch dataset
        client_indices: dict of {client_id: list of indices}
        batch_size: batch size for training
        shuffle: whether to shuffle data
        
    Returns:
        dict: {client_id: DataLoader}
    """
    client_loaders = {}
    for client_id, indices in client_indices.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        client_loaders[client_id] = loader
    
    return client_loaders


def analyze_data_distribution(dataset, client_indices, num_classes=10):
    """
    Analyze and print the label distribution for each client
    
    Args:
        dataset: PyTorch dataset
        client_indices: dict of {client_id: list of indices}
        num_classes: number of classes
    """
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    print("\n" + "="*80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for client_id, indices in client_indices.items():
        client_labels = labels[indices]
        class_counts = np.bincount(client_labels, minlength=num_classes)
        
        print(f"\nClient {client_id} (Total: {len(indices)} samples):")
        print(f"  Class distribution: {class_counts.tolist()}")
        print(f"  Percentages: {(class_counts / len(indices) * 100).round(2).tolist()}")
    
    print("="*80 + "\n")
