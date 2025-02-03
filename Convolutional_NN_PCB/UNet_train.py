from UNet_train_functions import *

n_train_cases = [50,100,250,500,1000,2000,5000]

for nt in n_train_cases:
    UNET_train(epochs=500,n_train=nt,overwrite=True)
    UNET_train_physics(epochs=500,n_train=nt,phy_param=1.0,overwrite=True)
