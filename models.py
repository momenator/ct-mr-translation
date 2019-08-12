import torch
import torchvision
import torch.nn as nn
import functools
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import utils
import itertools
import copy
import os


def conv_norm(in_dim, out_dim, kernel_size, stride, padding=0, relu=nn.ReLU):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm2d(out_dim),
                         relu())


def deconv_norm(in_dim, out_dim, kernel_size, stride, padding=0, output_padding=0):
    return nn.Sequential(nn.ConvTranspose2d(in_dim,
                                            out_dim,
                                            kernel_size,
                                            stride,
                                            padding,
                                            output_padding,
                                            bias=False), 
                         nn.BatchNorm2d(out_dim),
                         nn.ReLU())


class Residual(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Residual, self).__init__()
        
        self.ls = nn.Sequential(nn.ReflectionPad2d(1),
                                conv_norm(in_dim, out_dim, 3, 1),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(out_dim, out_dim, 3, 1),
                                nn.BatchNorm2d(out_dim))
    def forward(self, x):
        return x + self.ls(x)


class Discriminator(nn.Module):

    def __init__(self, dim=64):
        super(Discriminator, self).__init__()

        lrelu = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        conv_lrelu = functools.partial(conv_norm, relu=lrelu)

        self.ls = nn.Sequential(nn.Conv2d(1, dim, 4, 2, 1),
                                nn.LeakyReLU(0.2),
                                conv_lrelu(dim * 1, dim * 2, 4, 2, 1),
                                conv_lrelu(dim * 2, dim * 4, 4, 2, 1),
                                conv_lrelu(dim * 4, dim * 8, 4, 1, (1, 2)),
                                nn.Conv2d(dim * 8, 1, 4, 1, (2, 1)))

    def forward(self, x):
        return self.ls(x)


class Generator(nn.Module):

    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.ls = nn.Sequential(nn.ReflectionPad2d(3),
                                conv_norm(1, dim * 1, 7, 1),
                                conv_norm(dim * 1, dim * 2, 3, 2, 1),
                                conv_norm(dim * 2, dim * 4, 3, 2, 1),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                Residual(dim * 4, dim * 4),
                                deconv_norm(dim * 4, dim * 2, 3, 2, 1, 1),
                                deconv_norm(dim * 2, dim * 1, 3, 2, 1, 1),
                                nn.ReflectionPad2d(3),
                                nn.Conv2d(dim, 1, 7, 1),
                                nn.Tanh())

    def forward(self, x):
        return self.ls(x)

class ItemPool(object):

    def __init__(self, max_num=50):
        self.max_num = max_num
        self.num = 0
        self.items = []

    def __call__(self, in_items):
        if self.max_num <= 0:
            return in_items
        return_items = []
        for in_item in in_items:
            if self.num < self.max_num:
                self.items.append(in_item)
                self.num = self.num + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_num)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items


class CycleGAN:
    
    def __init__(self, dataroot, sub_dirs, batch_sizes, workers, lr, betas, gpu_ids, ckpt_dir, results_dir):
        
        # prepare dataset
        (trA, trB, teA, teB) = utils.create_gan_datasets(dataroot, sub_dirs)        

        # prepare loaders
        (trA_l, trB_l, teA_l, teB_l) = utils.create_gan_dataloaders(trA, trB, teA, teB, batch_sizes, workers)
        self.trA_l = trA_l
        self.trB_l = trB_l
        self.teA_l = teA_l
        self.teB_l = teB_l

        # define the models
        self.Da = Discriminator()
        self.Db = Discriminator()

        # Generate x from y
        self.Ga = Generator()
        # Generate y from x
        self.Gb = Generator()

        # define the loss functions
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # define the optimizers here
        self.da_optimizer = optim.Adam(self.Da.parameters(), lr=lr, betas=betas)
        self.db_optimizer = optim.Adam(self.Db.parameters(), lr=lr, betas=betas)
        self.ga_optimizer = optim.Adam(self.Ga.parameters(), lr=lr, betas=betas)
        self.gb_optimizer = optim.Adam(self.Gb.parameters(), lr=lr, betas=betas)

        
        # GPU set up
        print("gpu ids", gpu_ids) 
        utils.cuda_devices(gpu_ids)
        utils.cuda([ self.Da, self.Db, self.Ga, self.Gb ])

        # train!
        self.a_real_test = Variable(iter(teA_l).next())
        self.b_real_test = Variable(iter(teB_l).next())
        
        self.a_real_test, self.b_real_test = utils.cuda([self.a_real_test, self.b_real_test])
        self.a_fake_pool = ItemPool()
        self.b_fake_pool = ItemPool()

        self.start_epoch = 0
        self.ckpt_dir = ckpt_dir
        self.results_dir = results_dir
    

    def load_ckpt(self, checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path)
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Ga.load_state_dict(ckpt['Ga'])
            self.Gb.load_state_dict(ckpt['Gb'])
            self.da_optimizer.load_state_dict(ckpt['da_optimizer'])
            self.db_optimizer.load_state_dict(ckpt['db_optimizer'])
            self.ga_optimizer.load_state_dict(ckpt['ga_optimizer'])
            self.gb_optimizer.load_state_dict(ckpt['gb_optimizer'])
        except:
            print("No checkpoint!")
            self.start_epoch = 0


    def save_ckpt(self, epoch, iteration, keep_ckpt=2):
        ckpt_paths = os.listdir(self.ckpt_dir)
        ckpt_paths = [ self.ckpt_dir + '/' + ckpt_path for ckpt_path in ckpt_paths ]
        
        if len(ckpt_paths) >= keep_ckpt:
            # sort by time of creation
            ckpt_paths = sorted(ckpt_paths, key=os.path.getmtime)
            # delete earliest ckpt
            os.remove(ckpt_paths[0])
        
        torch.save({
            'epoch': epoch + 1,
            'Da': self.Da.state_dict(),
            'Db': self.Db.state_dict(),
            'Ga': self.Ga.state_dict(),
            'Gb': self.Gb.state_dict(),
            'da_optimizer': self.da_optimizer.state_dict(),
            'db_optimizer': self.db_optimizer.state_dict(),
            'ga_optimizer': self.ga_optimizer.state_dict(),
            'gb_optimizer': self.gb_optimizer.state_dict()
        }, '%s/Epoch_(%d)_(%d).ckpt' % (self.ckpt_dir, epoch + 1, iteration))        


    def train(self, epochs=200, eval_steps=200):
        
        for epoch in range(self.start_epoch, epochs):
                        
            for i, (a_real, b_real) in enumerate(zip(self.trA_l, self.trB_l)):

                step = epoch * min(len(self.trA_l), len(self.trB_l)) + i + 1

                # train Gx and Gy
                self.Ga.train()
                self.Gb.train()

                # wraps a_real and b_real
                a_real = Variable(a_real)
                b_real = Variable(b_real)
                a_real, b_real = utils.cuda([a_real, b_real])
                
                a_fake = self.Ga(b_real)
                b_fake = self.Gb(a_real)
                
                a_rec = self.Ga(b_fake)
                b_rec = self.Gb(a_fake)
                                
                # calculate loss

                # predict labels of fake images
                a_f_dis = self.Da(a_fake)
                b_f_dis = self.Db(b_fake)

                # set all labels as 1 as all are fakes
                r_labels = utils.cuda(Variable(torch.ones(a_f_dis.size())))

                a_gen_loss = self.MSE(a_f_dis, r_labels)
                b_gen_loss = self.MSE(b_f_dis, r_labels)
                

                # recon loss
                a_rec_loss = self.L1(a_rec, a_real)
                b_rec_loss = self.L1(b_rec, b_real)

                # why do we multiple rec loss by 10?
                cycle_loss = a_rec_loss * 10 + b_rec_loss * 10

                # compute total loss
                cg_loss = a_gen_loss + b_gen_loss + cycle_loss

                # backward pass and gradient update for Ga and Gb
                self.Ga.zero_grad()
                self.Gb.zero_grad()
                
                cg_loss.backward()
                                
                self.ga_optimizer.step()
                self.gb_optimizer.step()
                
                del cg_loss

                # train Da and Db

                # get a_fake and b_fake from fake pools
                a_fake = Variable(torch.Tensor(self.a_fake_pool([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(self.b_fake_pool([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])        

                # create labels from real and fake images
                a_r_dis = self.Da(a_real)
                a_f_dis = self.Da(a_fake)
                b_r_dis = self.Db(b_real)
                b_f_dis = self.Db(b_fake)

                r_labels = utils.cuda(Variable(torch.ones(a_f_dis.size())))
                f_labels = utils.cuda(Variable(torch.zeros(a_f_dis.size())))

                # calculate d losses
                a_d_r_loss = self.MSE(a_r_dis, r_labels)
                a_d_f_loss = self.MSE(a_f_dis, f_labels)
                b_d_r_loss = self.MSE(b_r_dis, r_labels)
                b_d_f_loss = self.MSE(b_f_dis, f_labels)
                
                a_d_loss = a_d_r_loss + a_d_f_loss
                b_d_loss = b_d_r_loss + b_d_f_loss

                # backward pass and grad update
                self.Da.zero_grad()
                self.Db.zero_grad()
                
                a_d_loss.backward()
                b_d_loss.backward()
                
                self.ga_optimizer.step()
                self.gb_optimizer.step()
                
                del a_d_loss
                del b_d_loss
                
                if i % eval_steps == 0:
                    self.evaluate(epoch, i, True)
    

    def save_as_npz(self, output_dir, fake_a, fake_b, rec_a, rec_b):
        np.savez(output_dir,
                 fake_a=fake_a, 
                 fake_b=fake_b, 
                 rec_a=rec_a, 
                 rec_b=rec_b)

    
    def evaluate(self, epoch, iteration, save_ckpt=False):
        self.Ga.eval()
        self.Gb.eval()

        # generate fake As and Bs
        a_fake_test = self.Ga(self.b_real_test)
        b_fake_test = self.Gb(self.a_real_test)

        # create reconstructed images
        a_rec_test = self.Ga(b_fake_test)
        b_rec_test = self.Gb(a_fake_test)

        pics = (torch.cat([self.a_real_test, 
                          b_fake_test, 
                          a_rec_test, 
                          self.b_real_test, 
                          a_fake_test, 
                          b_rec_test], dim=0).data + 1) / 2.0
                
        torchvision.utils.save_image(pics, '%s/Epoch_(%d)_(%d).png' % (self.results_dir, epoch, iteration), nrow=3)
        save_path_npz = '%s/Epoch_(%d)_(%d).npz' % (self.results_dir, epoch, iteration)
        self.save_as_npz(save_path_npz, 
                         a_fake_test.cpu().data.numpy(), 
                         b_fake_test.cpu().data.numpy(), 
                         a_rec_test.cpu().data.numpy(), 
                         b_rec_test.cpu().data.numpy())
        
        if save_ckpt:
            self.save_ckpt(epoch, iteration)

