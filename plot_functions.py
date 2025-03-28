import torch
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

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


#%%
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
    
#%%

def plot_loss_evolution(train_loss, test_loss):
    
    """
    Plot the training and validation loss
    
    Args:
    train_loss (list): list with the training loss
    test_loss (list): list with the test loss
    
    """
    
    epochs = np.arange(1, len(train_loss)+1)
    
    
    plt.plot(epochs, train_loss, label='Train', color='black')
    plt.plot(epochs, test_loss, label='Test', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    
    # Ajustar los ticks del eje x dinámicamente
    max_ticks = 10  # Número máximo de ticks que deseas mostrar
    if len(epochs) > max_ticks:
        tick_positions = np.linspace(1, len(epochs), num=max_ticks, dtype=int)
        plt.xticks(tick_positions)  # Mostrar solo los ticks seleccionados
    else:
        plt.xticks(epochs)  # Mostrar todos los ticks si hay pocos
        
    # Ajustar los límites del eje x al rango de valores de las épocas
    plt.xlim(epochs[0], epochs[-1])
    
    plt.show()
    

#%%
def plot_error_map(y_pred, y_true, i=0, t=0):
    """
    Muestra el mapa de temperaturas reales, predichas y el error (por pixel) en un instante concreto.
    Parámetros:
        y_pred: tensor con shape (B, T, 1, H, W)
        y_true: tensor con shape (B, T, 1, H, W)
        i: índice de la muestra
        t: timestep dentro de la secuencia
    """
    real = y_true[i, t].squeeze().detach().cpu().numpy()
    pred = y_pred[i, t].squeeze().detach().cpu().numpy()
    error = pred - real
    abs_error = abs(error)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    
    im0 = axs[0].imshow(real, cmap='hot')
    axs[0].set_title("Temperatura real")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred, cmap='hot')
    axs[1].set_title("Temperatura predicha")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(abs_error, cmap='viridis')
    axs[2].set_title("Absolute error")  
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()
    
#%%
def plot_se_map(y_pred, y_true, time=0, dt=1, show_pred=True, return_mse=False):
    """
    Muestra el mapa de temperaturas reales, predichas y el Squared Error (por pixel) en un instante concreto.
    
    Parámetros:
        y_pred: array con shape (T, H, W)
        y_true: tensor con shape (T, H, W)
        time: instante de tiempo real (en segundos)
        dt: intervalo de tiempo entre pasos
        show_pred: si es True, muestra el mapa de temperaturas predichas también
        return_mse: si es True, devuelve el valor del MSE
    """
    t = time // dt

    real = y_true[t, :, :]
    pred = y_pred[t, :, :]
    sq_diff = (pred - real) ** 2
    mse = np.mean(sq_diff)

    if show_pred:
        # Rango común de temperatura
        vmin = min(real.min(), pred.min())
        vmax = max(real.max(), pred.max())

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        
        im0 = axs[0].imshow(real, cmap='hot', vmin=vmin, vmax=vmax)
        axs[0].set_title("Real temperature [K]")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(pred, cmap='hot', vmin=vmin, vmax=vmax)
        axs[1].set_title("Predicted temperature [K]")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(sq_diff, cmap='viridis')
        axs[2].set_title("Squared error [K²]")
        plt.colorbar(im2, ax=axs[2])
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        
        im = ax.imshow(sq_diff, cmap='viridis')
        ax.set_title("Squared error [K²]")
        plt.colorbar(im, ax=ax)

    fig.suptitle(f'Temperature map at t = {time:.2f} s', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print(f"MSE: {mse:.8f} K^2")

    if return_mse:
        return mse
        
#%% 
def plot_nodes_evolution(y_pred, y_true, nodes_idx, dt=1, together=True, save_as_pdf=False, filename='nodes_evolution'):
    """
    Muestra la evolución temporal de las temperaturas reales y predichas en una serie de nodos.
    
    Parámetros:
        y_pred: array con shape (T, H, W)
        y_true: array con shape (T, H, W)
        nodes_idx: lista de índices de los nodos a mostrar [(idx1, idy1), (idx2, idy2), ...]
        dt: intervalo de tiempo entre cada paso de tiempo
        together: si es True, muestra todas las evoluciones en un solo gráfico
        save_as_pdf: si es True, guarda la figura como PDF en la carpeta 'figures'
        filename: nombre base del archivo (sin extensión)
    """
    time = np.arange(y_pred.shape[0]) * dt

    if together:
        plt.figure(figsize=(12, 6))
        
        for i, node_idx in enumerate(nodes_idx):
            color = plt.cm.tab10(i % 10)
            label = f'Node ({node_idx[0]}, {node_idx[1]})'

            y_true_node = y_true[:, node_idx[0], node_idx[1]]
            y_pred_node = y_pred[:, node_idx[0], node_idx[1]]

            plt.plot(time, y_true_node, label=f'{label} - Ground Truth', color=color)
            plt.plot(time, y_pred_node, 'x', label=f'{label} - Prediction', color=color)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [K]')
        plt.title('Time evolution of temperature in selected nodes')
        plt.xlim(time[0], time[-1])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_as_pdf:
            os.makedirs('figures', exist_ok=True)
            plt.savefig(f'figures/{filename}.pdf', format='pdf')
        plt.show()

    else:
        fig, axs = plt.subplots(len(nodes_idx), 1, figsize=(12, 3 * len(nodes_idx)), sharex=True)
        if len(nodes_idx) == 1:
            axs = [axs]

        for i, node_idx in enumerate(nodes_idx):
            axs[i].plot(time, y_true[:, node_idx[0], node_idx[1]], label='Ground truth', color='blue')
            axs[i].plot(time, y_pred[:, node_idx[0], node_idx[1]], 'x', label='Prediction', color='orange')
            axs[i].set_title(f"Node ({node_idx[0]}, {node_idx[1]})")

            axs[i].set_ylabel('Temperature [K]')
            axs[i].set_xlim(time[0], time[-1])

            if i == len(nodes_idx) - 1:
                axs[i].set_xlabel('Time [s]')
            if i == 0:
                axs[i].legend(loc='upper right')

        fig.suptitle('Time evolution of temperature in selected nodes', fontsize=16)
        fig.align_ylabels()
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_as_pdf:
            os.makedirs('figures', exist_ok=True)
            fig.savefig(f'figures/{filename}.pdf', format='pdf')
        plt.show()