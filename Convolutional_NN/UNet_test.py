from UNet_test_functions import *

UNET_test_test(model_name='case_e_20_nt_50_bs_5')
UNET_test_train(model_name='case_e_20_nt_50_bs_5',n_train=50)
UNET_plot_sample_test(model_name='case_e_20_nt_50_bs_5')
UNET_plot_sample_train(model_name='case_e_20_nt_50_bs_5',n_train=50)



UNET_test_test(model_name='case_e_20_nt_50_bs_5',finetuning_model_name='case_e_20_nt_10_bs_1')
UNET_test_train(model_name='case_e_20_nt_50_bs_5',n_train=20,finetuning_model_name='case_e_20_nt_10_bs_1')
UNET_plot_sample_test(model_name='case_e_20_nt_50_bs_5',finetuning_model_name='case_e_20_nt_10_bs_1')
UNET_plot_sample_train(model_name='case_e_20_nt_50_bs_5',n_train=20,finetuning_model_name='case_e_20_nt_10_bs_1')
