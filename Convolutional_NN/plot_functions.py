import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_sample(output,target):

    plt.style.use('default')

    plt.rcParams["figure.figsize"] = (6,4)

    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams["font.size"] = 12

    plt.rcParams["text.usetex"] = False

    plt.rcParams["axes.titlesize"] = 11

    # Convertir los tensores a numpy y asegurarse de que están en CPU
    try:
        output_np = output.squeeze().cpu().detach().numpy()
        target_np = target.squeeze().cpu().detach().numpy()
    except:
        output_np=output
        target_np=target
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Asegurarse de que las celdas de la grilla sean lo suficientemente grandes para el texto
    fig.tight_layout(pad=3.0)
    
    # Output de la red
    axs[1].imshow(output_np, cmap='viridis', interpolation='nearest')
    axs[1].title.set_text('Output')
    for i in range(output_np.shape[0]):
        for j in range(output_np.shape[1]):
            text = axs[1].text(j, i, f'{output_np[i, j]:.0f}',
                            ha="center", va="center", color="w", fontsize=6)

    # Target
    axs[0].imshow(target_np, cmap='viridis', interpolation='nearest')
    axs[0].title.set_text('Target')
    for i in range(target_np.shape[0]):
        for j in range(target_np.shape[1]):
            text = axs[0].text(j, i, f'{target_np[i, j]:.0f}',
                            ha="center", va="center", color="w", fontsize=6)

    # Calcular la diferencia
    diferencia_np =np.abs( output_np - target_np)
    
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    
    # Asegurarse de que las celdas de la grilla sean lo suficientemente grandes para el texto
    fig2.tight_layout(pad=3.0)
    
    # Diferencia entre el output de la red y el target
    cax2 = ax2.imshow(diferencia_np, cmap='viridis', interpolation='nearest')
    ax2.title.set_text('Absolute Error')

    fig.colorbar(cax2)
    plt.show()


def visualizar_valores_vectoreslatentes(output, target):

    plt.style.use('default')

    plt.rcParams["figure.figsize"] = (6,4)

    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams["font.size"] = 12

    plt.rcParams["text.usetex"] = False

    plt.rcParams["axes.titlesize"] = 11
    
    # Convert tensors to numpy and make sure they're on CPU
    if type(output) == torch.Tensor:
        output_np = output.cpu().detach().numpy()
    else:
        output_np = output
    if type(target) == torch.Tensor:
        target_np = target.cpu().detach().numpy()
    else: 
        target_np = target
    
    # Create a plot with 2 subplots: one for the output and one for the target
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the output vector
    axs[1].plot(list(range(1,10)),output_np.flatten(), marker='.', linestyle='--', color='darkcyan')
    axs[1].title.set_text('Output')
    axs[1].grid(False)
    
    # Plot the target vector
    axs[0].plot(list(range(1,10)),target_np.flatten(), marker='.', linestyle='-', color='crimson')
    axs[0].title.set_text('Target')
    axs[0].grid(False)

    # Show the plot
    plt.show()

    # Create a plot with 2 subplots: one for the output and one for the target
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    
    # Plot the output vector
    axs.plot(list(range(1,10)),output_np.flatten(), marker='.', linestyle='--', color='darkcyan')
    axs.plot(list(range(1,10)),target_np.flatten(), marker='.', linestyle='--', color='crimson')
    axs.grid(False)

    
    # Plot the target vector
    plt.show()