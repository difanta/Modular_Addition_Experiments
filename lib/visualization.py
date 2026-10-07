import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_learning_acc_and_loss(loss_tr, acc_tr, loss_val, acc_val, norm=None, gradient_norm=None, block=True):
    """  
    plot the training and validation losses as well as the training and validation accuracies  
    """  
    n_plots = 2
    if norm is not None:
        n_plots += 1
    if gradient_norm is not None:
        n_plots += 1
  
    if n_plots == 4:
        _, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
    else:
        _, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
        axes = axes.flatten() if n_plots > 1 else [axes]  
  
    ax1 = axes[0]  
    ax1.grid()  
    ax1.plot(range(len(acc_tr)), acc_tr, label='acc_training')  
    ax1.plot(range(len(acc_val)), acc_val, label='acc_validation')  
    ax1.set_xlabel('Epochs')  
    ax1.set_ylabel('Accuracy')  
    ax1.legend(loc='best')  
  
    ax2 = axes[1]  
    ax2.grid()  
    ax2.plot(range(len(loss_tr)), loss_tr, label='loss_training')  
    ax2.plot(range(len(loss_val)), loss_val, label='loss_validation')  
    ax2.set_xlabel('Epochs')  
    ax2.set_ylabel('Loss')  
    ax2.legend(loc='best')  
  
    idx = 2
    if norm is not None:  
        ax3 = axes[idx]  
        ax3.grid()  
        ax3.plot(range(len(norm)), norm, label='norm', color='green')    
        ax3.set_xlabel('Epochs')    
        ax3.set_ylabel('Norm')    
        ax3.legend(loc='best')
        idx += 1
  
    if gradient_norm is not None:  
        ax4 = axes[idx]  
        ax4.grid()  
        ax4.plot(range(len(gradient_norm)), gradient_norm, label='gradient_norm', color='blue')    
        ax4.set_xlabel('Epochs')    
        ax4.set_ylabel('Gradient Norm')    
        ax4.legend(loc='best')
        idx += 1
  
    plt.tight_layout()  
    plt.show(block=block)

def visualize_2D_matrix(matrix, title):
    # Create a white-to-blue colormap
    cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])

    fig, ax = plt.subplots(figsize=(6, 5))
    # Display the matrix; vmin=0 ensures 0 maps to white
    im = ax.imshow(matrix, cmap=cmap, vmin=np.min(matrix) if matrix.min() < 0 else 0, vmax=np.max(matrix) if matrix.max() > 0 else 0)

    rows, cols = matrix.shape
    
    # Calculate a dynamic step size for ticks (aim for ~30 ticks max per axis)
    x_step = max(1, cols // 10)
    y_step = max(1, rows // 30)

    # Set major ticks with the calculated step size
    ax.set_xticks(np.arange(0, cols, x_step))
    ax.set_yticks(np.arange(0, rows, y_step))

    # Set up minor ticks for grid lines (offset by -0.5 to center on cell edges)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)
    ax.tick_params(which="minor", length=0) # Hide the minor tick marks themselves

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Value")
    plt.tight_layout()
    plt.show(block=False)