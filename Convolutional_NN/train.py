from UNet_train_functions import *
from UNet_finetuning_train_functions import *

n_train_cases = [50,100,250,500,1000,2000,5000]
n_train_finetuning_cases = [10,20,30,40,50,60,70,80,90,100]

for nt in n_train_cases:
    UNET_train(epochs=500,n_train=nt,overwrite=True)
    UNET_train_physics(epochs=500,n_train=nt,phy_param=1.0,overwrite=True)

for nt in n_train_cases:
    for nt2 in n_train_finetuning_cases:
        UNET_finetuning_train(pretrained_model='case_e_500_nt_{}_bs_5'.format(nt),epochs=500,
                              n_train=nt2,lr0=1E-6,overwrite=True,batch_size=1)
        UNET_finetuning_train(pretrained_model='case_e_500_nt_{}_bs_5'.format(nt),epochs=500,
                              n_train=nt2,lr0=1E-6,overwrite=True,batch_size=1,only_last_layers=True)