import torch
from models import CycleGAN
import os

dataroot = './data/apple2orange'
sub_dirs = ['trainA', 'trainB', 'testA', 'testB']
batch_sizes = [128, 128, 4, 4]
workers = 2
lr = 0.0002
betas=(0.5, 0.999)
epochs = 1
gpu_ids = [0, 7]

# check cuda
print(torch.cuda.is_available())
print("Current device ", torch.cuda.current_device())
print("How many device? ", torch.cuda.device_count())
# torch.cuda.set_device(6)
# print("Current device ", torch.cuda.current_device())


# init cycleGAN instance
cg = CycleGAN(dataroot, sub_dirs, batch_sizes, workers, lr, betas, gpu_ids)

print("ready to train")
# print(os.environ['CUDA_VISIBLE_DEVICES'])

cg.train(epochs)
