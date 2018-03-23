
<br><br><br>

# Improvements on CycleGAN
Steps to run the WGAN with horse2zebra dataset

bash ./datasets/download_cyclegan_dataset.sh apple2orange

#### Parameters:

Flags to setup wgan are available in models/cycle_gan_model.py. These are set to enable WGAN by default with the bounds for clamping as -0.01, 0.01 (controlled by params: wgan_upbound and wgan_lowbound). 

#### Train Model: 

python train.py --dataroot ./datasets/apple2orange --name anyname --model cycle_gan --no_dropout

#### Test Model: 

python test.py --dataroot ./datasets/apple2orange --name anyname --model cycle_gan --phase test --no_dropout

######3 Cycle train:
Rename train_options_cycle to train_options in the options folder. 
Rename base_options_cycle to train_options in the options folder. 
Rename cycle_gan_model_cycle to cycle_gan_model in the models folder
Rename networks_cycle to networks in the models folder
Download jackfruit dataset from ImageNet 
python train.py --dataroot ./datasets/apple2orange --name anyname --model cycle_gan --no_dropout
## Running Progressive GAN
Rename train_options_progressive to train_options in the options folder. 
Rename cycle_gan_model_progressive to cycle_gan_model in the models folder
Rename networks_progressive to networks in the models folder
python train.py --dataroot ./datasets/apple2orange --name anyname --model cycle_gan --no_dropout


## Acknowledgments
Code is inspired by [Image-to-image translation in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


