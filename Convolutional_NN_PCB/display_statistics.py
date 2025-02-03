import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def display_statistics(model_name:str,finetuning_model_name:str=None):
    
    # Paths de los diferentes directorios
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path,'Models')

    # Paths de los diferentes directorios
    dir_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(dir_path,'Models')
    if finetuning_model_name == None:
        folder_path = os.path.join(models_path,'UNET',model_name)
    else:
        folder_path = os.path.join(models_path,'UNET_finetuning',model_name,finetuning_model_name)

    path = os.path.join(folder_path,'results')

    train_loss_df = pd.read_csv(os.path.join(path,'train_loss.csv'))
    train_results_df = pd.read_csv(os.path.join(path,'train_results.csv'))
    test_results_df = pd.read_csv(os.path.join(path,'test_results.csv'))

    # Plot loss statistics train and test.
    
    train_results_rmse = list(train_results_df['rmse_loss'])
    train_results_max = list(train_results_df['max_loss'])
    test_results_rmse = list(test_results_df['rmse_loss'])
    test_results_max = list(test_results_df['max_loss'])

    colors = ['deepskyblue','orange']
    fig,ax = plt.subplots()
    ax.hist([train_results_rmse,test_results_rmse],bins=11,
            weights=[100*np.ones(len(train_results_rmse))/len(train_results_rmse),100*np.ones(len(test_results_rmse))/len(test_results_rmse)],
            edgecolor='black',alpha=0.8,color=colors)
    ax.set_title('RMSE for the train set')
    ax.set_xlabel('RMSE error')
    ax.set_ylabel(r'% of smaples')
    ax.legend(['train','test'])
    # ax.set_yscale('log')
    plt.savefig(os.path.join(path,'RMSE_error.png'),dpi=300,bbox_inches='tight')
    plt.show()

    fig,ax = plt.subplots()
    ax.hist([train_results_max,test_results_max],bins = 11,
            weights=[100*np.ones(len(train_results_max))/len(train_results_max),100*np.ones(len(test_results_max))/len(test_results_max)],
            edgecolor='black',alpha=0.8,color=colors)
    ax.set_title('max error for the test set')
    ax.set_xlabel('Max error')
    ax.set_ylabel(r'% of samples')
    ax.legend(['train','test'])
    # ax.set_yscale('log')
    plt.savefig(os.path.join(path,'max_error.png'),dpi=300,bbox_inches='tight')
    plt.show()

    # Plot loss evolution during train
    train_loss_train = list(train_loss_df['train_loss'])
    train_loss_test = list(train_loss_df['test_loss'])
    epochs = list(range(train_loss_train.__len__()))
    fig,ax = plt.subplots()
    plt.plot(epochs,train_loss_train)
    plt.plot(epochs,train_loss_test)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('error')
    ax.legend(['train','test'])
    # ax.set_yscale('log')
    plt.savefig(os.path.join(path,'error_evolution.png'),dpi=300,bbox_inches='tight')
    plt.show()
    

display_statistics(epochs=250,n_train=100,finetuning=False,physics=False,phy_param=1.0)
    