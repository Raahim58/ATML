"""
Federated Learning utility functions
Aggregation, evaluation, and drift metrics
"""

import torch
import numpy as np
import copy


def get_model_weights(model):
    """
    Extract weights from a PyTorch model
    
    Args:
        model: PyTorch model
        
    Returns:
        list of tensors (detached and on CPU)
    """
    return [param.detach().cpu().clone() for param in model.parameters()]


def set_model_weights(model, weights):
    """
    Load weights into a PyTorch model
    
    Args:
        model: PyTorch model
        weights: list of tensors
    """
    with torch.no_grad():
        for param, weight in zip(model.parameters(), weights):
            param.copy_(weight.to(param.device))


def aggregate_weights(client_weights, client_sizes):
    """
    Weighted averaging of client model weights
    
    Args:
        client_weights: list of [weights for each client]
        client_sizes: list of number of samples for each client
        
    Returns:
        aggregated weights (list of tensors)
    """
    total_size = sum(client_sizes)
    
    # Initialize with zeros
    aggregated = [torch.zeros_like(client_weights[0][i]) for i in range(len(client_weights[0]))]
    
    # Weighted sum
    for client_weight, size in zip(client_weights, client_sizes):
        weight_factor = size / total_size
        for i, param in enumerate(client_weight):
            aggregated[i] += param * weight_factor
    
    return aggregated


def compute_weight_divergence(client_weights, global_weights):
    """
    Compute the average L2 distance between client models and global model
    
    Args:
        client_weights: list of [weights for each client]
        global_weights: global model weights
        
    Returns:
        average divergence (float)
    """
    divergences = []
    
    for client_weight in client_weights:
        divergence = 0.0
        for client_param, global_param in zip(client_weight, global_weights):
            divergence += torch.norm(client_param - global_param).item() ** 2
        divergences.append(np.sqrt(divergence))
    
    return np.mean(divergences)


def evaluate_model(model, test_loader, device='cpu', criterion=None):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: device to run evaluation on
        criterion: loss function (default: CrossEntropyLoss)
        
    Returns:
        accuracy, loss
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Loss
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            # Accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    
    return accuracy, avg_loss


def flatten_weights(weights):
    """
    Flatten a list of weight tensors into a single vector
    
    Args:
        weights: list of tensors
        
    Returns:
        flattened 1D tensor
    """
    return torch.cat([w.flatten() for w in weights])


def unflatten_weights(flat_weights, shapes):
    """
    Unflatten a 1D tensor back to list of tensors with given shapes
    
    Args:
        flat_weights: 1D tensor
        shapes: list of shapes for each parameter
        
    Returns:
        list of tensors
    """
    weights = []
    offset = 0
    for shape in shapes:
        numel = np.prod(shape)
        weights.append(flat_weights[offset:offset+numel].view(shape))
        offset += numel
    return weights


def compute_gradient_from_weights(initial_weights, updated_weights, lr):
    """
    Compute effective gradient from weight update
    gradient = (initial_weights - updated_weights) / lr
    
    Args:
        initial_weights: initial model weights
        updated_weights: updated model weights
        lr: learning rate used
        
    Returns:
        list of gradient tensors
    """
    gradients = []
    for init_w, upd_w in zip(initial_weights, updated_weights):
        grad = (init_w - upd_w) / lr
        gradients.append(grad)
    return gradients
