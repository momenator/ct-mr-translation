import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import os
import functools
import numpy as np
import matplotlib.pyplot as plt


class NpzDataset(data.Dataset):
    def __init__(self, root):
        self.root_folder = root
        self.image_paths = os.listdir(root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Select sample
        image_path = self.image_paths[index]

        # Load data and get label
        image = np.load(self.root_folder + '/' + image_path)['data']
        return image


def create_dataset_npz(root):
    return NpzDataset(root)


def create_dataset(root):
    return dset.ImageFolder(root=root,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = list(paths)
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def create_dir_map(root_dir, target_dirs):
    """
        Create a directory map for ImageFolder
    """
    
    dir_map = {}
    
    for target_dir in target_dirs:
        dir_map[target_dir] = os.path.join(root_dir, 'link_' + target_dir)
        
    mkdir(dir_map.values())

    for key in dir_map:
        try:
            os.remove(os.path.join(dir_map[key], '0'))
        except:
            pass
        os.symlink(os.path.abspath(os.path.join(root_dir, key)),
                   os.path.join(dir_map[key], '0'))
    return dir_map


def create_gan_datasets(dataroot, sub_dirs):
    """
        Create 4 datasets for cycle GAN model
        return (trainA, trainB, testA, testB)
    """
    dir_map = create_dir_map(dataroot, sub_dirs)
        
    trainA = create_dataset_npz(dir_map[sub_dirs[0]])
    trainB = create_dataset_npz(dir_map[sub_dirs[1]])
    testA = create_dataset_npz(dir_map[sub_dirs[2]])
    testB = create_dataset_npz(dir_map[sub_dirs[3]])
        
    return (trainA, trainB, testA, testB)


def create_gan_dataloaders(trainA, trainB, testA, testB, batch_sizes, workers):
    trainA_loader = torch.utils.data.DataLoader(trainA, batch_size=batch_sizes[0], shuffle=True, num_workers=workers)
    trainB_loader = torch.utils.data.DataLoader(trainB, batch_size=batch_sizes[1], shuffle=True, num_workers=workers)
    testA_loader = torch.utils.data.DataLoader(testA, batch_size=batch_sizes[2], shuffle=True, num_workers=workers)
    testB_loader = torch.utils.data.DataLoader(testB, batch_size=batch_sizes[3], shuffle=True, num_workers=workers)
    return (trainA_loader, trainB_loader, testA_loader, testB_loader)


def display_batch(loader, device, title="Images", num_row=8, num_col=8):
    sample_batch = next(iter(loader))
    plt.figure(figsize=(num_row, num_col))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(
        sample_batch[0].to(device)[:num_row * num_col], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


# CUDA helpers
def cuda_devices(gpu_ids):
    gpu_ids = [str(i) for i in gpu_ids]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)


def cuda(xs):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda()
        else:
            return [x.cuda() for x in xs]
    return xs

