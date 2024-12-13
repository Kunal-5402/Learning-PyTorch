import torch
from torch import nn
import matplotlib.pyplot as plt

# Function to visualize weights and biases using Matplotlib scatter plot
def plot_weights_and_biases_scatter(model, filename):
    num_layers = sum(1 for layer in model.network_layers if isinstance(layer, nn.Linear))
    
    fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4 * num_layers))

    if num_layers == 1:
        axes = [axes]

    layer_index = 0
    for idx, layer in enumerate(model.network_layers):
        if isinstance(layer, nn.Linear):
            # Plot Weights
            weight = layer.weight.detach().numpy()
            for i in range(weight.shape[0]):
                axes[layer_index][0].scatter(
                    x=list(range(weight.shape[1])),
                    y=weight[i, :],
                    alpha=0.6,
                    s=20,
                    label=f"Weight {i}"
                )

            axes[layer_index][0].set_title(f"Layer {layer_index} Weights")
            axes[layer_index][0].set_xlabel("Neuron Index")
            axes[layer_index][0].set_ylabel("Weight Value")
            axes[layer_index][0].grid(True, linestyle='--', alpha=0.5)

            # Plot Biases
            bias = layer.bias.detach().numpy()
            axes[layer_index][1].scatter(
                x=list(range(len(bias))),
                y=bias,
                alpha=0.8,
                s=40,
                marker='x',
                label="Bias"
            )

            axes[layer_index][1].set_title(f"Layer {layer_index} Biases")
            axes[layer_index][1].set_xlabel("Neuron Index")
            axes[layer_index][1].set_ylabel("Bias Value")
            axes[layer_index][1].grid(True, linestyle='--', alpha=0.5)

            layer_index += 1

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.show()


def plot_weights_and_biases_heatmap(model, filename):

    fig, axes = plt.subplots(nrows=len(model.network_layers)//2 + 1, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    layer_index = 0
    for idx, layer in enumerate(model.network_layers):
        if isinstance(layer, nn.Linear):
            # Plot weights
            weight = layer.weight.detach().numpy()
            ax_w = axes[layer_index]
            im_w = ax_w.imshow(weight, aspect='auto', cmap='viridis')
            ax_w.set_title(f'Layer {idx} Weights')
            plt.colorbar(im_w, ax=ax_w, orientation='vertical')
            
            # Plot biases
            bias = layer.bias.detach().numpy().reshape(-1, 1)
            ax_b = axes[layer_index + 1]
            im_b = ax_b.imshow(bias, aspect='auto', cmap='viridis')
            ax_b.set_title(f'Layer {idx} Biases')
            plt.colorbar(im_b, ax=ax_b, orientation='vertical')
            
            layer_index += 2
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.show()

# plot_weights_and_biases_scatter(model)
# plot_weights_and_biases_heatmap(model)