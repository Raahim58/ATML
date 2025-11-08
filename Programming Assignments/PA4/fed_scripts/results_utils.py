"""
Utilities for saving and loading experimental results
"""

import pickle
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def save_results(results, filename, results_dir='./results'):
    """
    Save experimental results to pickle file
    
    Args:
        results: dict with experimental results
        filename: name of file (without extension)
        results_dir: directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{filename}.pkl")
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filepath}")


def load_results(filename, results_dir='./results'):
    """
    Load experimental results from pickle file
    
    Args:
        filename: name of file (with or without .pkl extension)
        results_dir: directory containing results
        
    Returns:
        dict with results
    """
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    return results


def plot_comparison(results_dict, metric='test_accuracy', title=None, save_path=None):
    """
    Plot comparison of multiple experiments
    
    Args:
        results_dict: dict of {experiment_name: results}
        metric: metric to plot ('test_accuracy' or 'test_loss')
        title: plot title
        save_path: path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    
    for exp_name, results in results_dict.items():
        if 'history' in results:
            history = results['history']
        else:
            history = results
        
        rounds = history.get('rounds', range(1, len(history[metric]) + 1))
        values = history[metric]
        
        plt.plot(rounds, values, marker='o', label=exp_name, linewidth=2)
    
    plt.xlabel('Communication Rounds', fontsize=12)
    
    if metric == 'test_accuracy':
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        if title is None:
            title = 'Test Accuracy vs Communication Rounds'
    elif metric == 'test_loss':
        plt.ylabel('Test Loss', fontsize=12)
        if title is None:
            title = 'Test Loss vs Communication Rounds'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_summary(results_dict):
    """
    Print summary statistics for multiple experiments
    
    Args:
        results_dict: dict of {experiment_name: results}
    """
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    for exp_name, results in results_dict.items():
        if 'history' in results:
            history = results['history']
        else:
            history = results
        
        final_acc = history['test_accuracy'][-1]
        best_acc = max(history['test_accuracy'])
        final_loss = history['test_loss'][-1]
        
        print(f"\n{exp_name}:")
        print(f"  Final Test Accuracy: {final_acc:.2f}%")
        print(f"  Best Test Accuracy:  {best_acc:.2f}%")
        print(f"  Final Test Loss:     {final_loss:.4f}")
        
        # Print configuration if available
        if 'config' in results:
            print(f"  Configuration:")
            for key, value in results['config'].items():
                print(f"    {key}: {value}")
    
    print("="*80 + "\n")
