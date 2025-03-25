import numpy as np
import matplotlib.pyplot as plt

#%%
def plot_error_map(y_pred, y_true, i=0, t=500):
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
    axs[2].set_title("Error absoluto")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()
    

#%%

def plot_loss_evolution(train_loss, val_loss):
    
    """
    Plot the training and validation loss
    
    Args:
    train_loss (list): list with the training loss
    val_loss (list): list with the validation loss
    
    """
    
    train_loss_plt = np.array(train_loss)
    val_loss_plt = np.array([v for v in val_loss])
    epochs = np.arange(1, len(train_loss)+1)
    
    
    plt.plot(epochs, train_loss, label='Train')
    plt.plot(epochs, val_loss, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
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