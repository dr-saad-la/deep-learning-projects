"""
    Evaluation utility functions
"""
import os


import numpy as np

import tensorflow as tf

# Print Evaluation Metrics
def print_comparative_metrics(model_adaptive, model_static, test_data,
                            adaptive_history, static_history):
    """Print detailed comparison metrics for both models.

    Args:
        model_adaptive: Model trained with ReduceLROnPlateau
        model_static: Model trained with static learning rate
        test_data: Tuple of (X_test, y_test)
        adaptive_history: Training history of adaptive model
        static_history: Training history of static model
    """
    X_test, y_test = test_data

    # Test set evaluation
    adaptive_results = model_adaptive.evaluate(X_test, y_test, verbose=0)
    static_results = model_static.evaluate(X_test, y_test, verbose=0)
    metrics = ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']

    print("\nComparative Model Performance Summary")
    print("=" * 50)

    # Print test metrics comparison
    print("\nTest Set Metrics:")
    print("-" * 20)
    for i, metric in enumerate(metrics):
        print(f"{metric:>10}:  Adaptive: {adaptive_results[i]:.4f}  "
              f"Static: {static_results[i]:.4f}")

    # Training statistics
    print("\nTraining Statistics:")
    print("-" * 20)
    for history, name in [(adaptive_history, "Adaptive"), (static_history, "Static")]:
        best_epoch = np.argmin(history['val_loss']) + 1
        print(f"\n{name} Model:")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
        print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
        print(f"Best validation AUC: {max(history['val_auc']):.4f}")

    # Learning rate summary for adaptive model
    print("\nLearning Rate Summary:")
    print("-" * 20)
    lr_changes = sum(1 for i in range(1, len(adaptive_history['learning_rate']))
                    if adaptive_history['learning_rate'][i] <
                    adaptive_history['learning_rate'][i-1])
    print(f"Number of learning rate reductions: {lr_changes}")
    print(f"Initial learning rate: {adaptive_history['learning_rate'][0]:.6f}")
    print(f"Final learning rate: {adaptive_history['learning_rate'][-1]:.6f}")
