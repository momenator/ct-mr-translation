import torch
from models import CycleGAN

dataroot = './data/final'
sub_dirs = ['trainA', 'trainB', 'testA', 'testB']
batch_sizes = [1, 1, 3, 3]
workers = 2
lr = 0.0002
betas=(0.5, 0.999)
epochs = 200
gpu_ids = [0]
ckpt_dir = './ckpt/visceral'
results_dir = './results/visceral'

# check cuda
# print(torch.cuda.is_available())
# print("Current device ", torch.cuda.current_device())
# print("How many device? ", torch.cuda.device_count())
# torch.cuda.set_device(6)
# print("Current device ", torch.cuda.current_device())

# init cycleGAN instance
cg = CycleGAN(dataroot, sub_dirs, batch_sizes, 
  workers, lr, betas, gpu_ids, ckpt_dir, results_dir)

cg.train(epochs, 10)
