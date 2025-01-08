"""
    Visualization utilities
"""
import os

import matpplotlib.pyplot as plt


# The visualization tools
def plot_training_history(history, save_path=None):
    """Visualize training metrics and learning rate adaptation.

    Args:
        history: Training history from model.fit()
        save_path (str, optional): Path to save the plot. If None, display only.
    """
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Find epochs where learning rate changed
    lr_changes = [i for i in range(1, len(history.history['learning_rate']))
                 if history.history['learning_rate'][i] != history.history['learning_rate'][i-1]]

    # Plot loss with LR change indicators
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    for change in lr_changes:
        ax1.axvline(x=change, color='r', linestyle='--', alpha=0.3)
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy with LR change indicators
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    for change in lr_changes:
        ax2.axvline(x=change, color='r', linestyle='--', alpha=0.3)
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Plot learning rate
    ax3.plot(history.history['learning_rate'], label='Learning Rate')
    ax3.set_title('Learning Rate Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)

    # Plot AUC
    ax4.plot(history.history['auc'], label='Training AUC')
    ax4.plot(history.history['val_auc'], label='Validation AUC')
    for change in lr_changes:
        ax4.axvline(x=change, color='r', linestyle='--', alpha=0.3)
    ax4.set_title('Model AUC')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('AUC')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()
