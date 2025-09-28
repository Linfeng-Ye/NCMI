# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

import torch.nn.functional as F

import math

class Center(nn.Module):
    def __init__(self, n_dim, CdecayFactor=0.999):
        super(Center, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_dim = n_dim
        self.Center = nn.Parameter(torch.zeros((1, self.n_dim)).to(self.device))
        self.CdecayFactor = CdecayFactor
        self.virgin = True
    def update(self, feature):
        if self.virgin:
            self.Center.data = feature.detach().mean(dim=tuple(range(feature.dim() - 1)))
            self.virgin = False
        else:
            self.Center.data = self.CdecayFactor*self.Center.data + \
                        (1-self.CdecayFactor)*feature.detach().mean(dim=tuple(range(feature.dim() - 1)))
    def forward(self):
        return self.Center

def sigmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Applies element-wise sigmoid to x and normalizes along the specified dim.
    Returns a probability vector (non-negative, sums to 1 along dim).

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to normalize over.
        eps (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Probability vector.
    """
    sigmoid_x = torch.sigmoid(x)
    norm = sigmoid_x.sum(dim=dim, keepdim=True)
    return sigmoid_x / (norm + eps)

# sigmax = torch.nn.functional.softmax
class Centroid(nn.Module):
    def __init__(self, n_classes, n_dim, samples_data, samples_tar, Ctemp, CdecayFactor=0.999):
        super().__init__()
        self.n_classes = n_classes
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_dim = n_dim
        self.centroids = torch.eye(self.n_classes)
        self.centroids = F.pad(self.centroids, (0, self.n_dim-self.n_classes))*10+1e-6
        self.centroids = self.centroids.to(self.device)
        self.centroids = nn.Parameter(self.centroids)
        self.CdecayFactor = CdecayFactor
        self.Ctemp = Ctemp
        self.samples_data = torch.tensor(samples_data).to(self.device)
        self.samples_tar = samples_tar
    
    def update_epoch(self, model,featrue_center , data_loader):
        self.centroids.data = torch.zeros_like(self.centroids.data)
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for image,target in tqdm(data_loader):
                image,target = image.to(device), target.to(device)
                logits = model(image)

                logits = torch.nn.functional.normalize(logits-featrue_center(),1)/self.Ctemp
                # logits = logits/self.Ctemp
                Classes =  target.unique()
                logits = logits
                output = sigmax(logits.float(), 1)
                for Class in Classes:
                    self.centroids.data[Class] += torch.sum(output[target == Class], axis = 0)
        self.centroids.data =  self.centroids.data/(self.centroids.data.sum(1)[:,None])
    def get_centroids(self, target):
        with torch.no_grad():
            return torch.index_select(self.centroids, 0, target).to(target.device)
        # with torch.no_grad():
        #     return torch.index_select(self.centroids.data, 0, target.detach()).to(target.device)



def pairwiseKL(Log_Prob, Log_Cent_All):

    Log_Prob    = Log_Prob.unsqueeze(1)
    Log_Cent_All = Log_Cent_All.unsqueeze(0) 
    D = (Log_Prob.exp()* (Log_Prob-Log_Cent_All)).sum(2)
    return D

# def Seperation(Log_Prob, Log_Cent_All):

#     Log_Prob    = Log_Prob.unsqueeze(1)
#     Log_Cent_All = Log_Cent_All.unsqueeze(0) 
#     D = (Log_Cent_All.exp()*(Log_Cent_All-Log_Prob)).sum(2)
#     return D


def pairwise_cross_entropy(Probability: torch.Tensor,
                           labels: torch.LongTensor) -> torch.Tensor:

    p      = Probability       
    log_p  = Probability.log().detach()       
    ce_ij = - (p.unsqueeze(1) * log_p.unsqueeze(0)).sum(dim=2)  
    label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  
    ce_ij = ce_ij.masked_fill(label_eq, 0.0)
    return ce_ij


def train(epoch, centroids, featrue_center, Uepoch,  args):
    start = time.time()
    net.train()
    batch_index = 0
    for  images, labels in tqdm(cifar100_training_loader):
        batch_index += 1
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        
        optimizer.zero_grad()
        feature = net(images)

        featrue_center.update(feature)
        # breakpoint()
        feature = torch.nn.functional.normalize(feature-featrue_center(), dim=1)/args.feature_Temp

        Probability = sigmax(feature, 1)
        log_prob = Probability.log()

        # Centroid = centroids.get_centroids(labels.clone()).cuda()
        Centroid =  centroids.centroids.data
        # breakpoint()

        Centroid = sigmax(Centroid, 1)
        log_cent = Centroid.log()
        # breakpoint()
        KL_div = torch.nan_to_num(pairwiseKL(log_prob, log_cent))
        # breakpoint()
        CMI = (KL_div[torch.arange(KL_div.size(0)), labels]+1e-10)

        KL_divs = torch.nan_to_num(pairwiseKL(log_cent, log_prob))
        
        KL_divs[labels, torch.arange(KL_divs.size(1))] = 0.0
        SEP = torch.nan_to_num(KL_divs).mean(0)
        # KL_div[torch.arange(KL_div.size(0)), labels] = 0.0
        
        loss =(CMI/SEP).mean()
        # loss =CMI/SEP

        # try:
        loss.backward()
        optimizer.step()
        if epoch>=Uepoch:
            opt_cent.step()
        # except:
        #     breakpoint()
        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]

    finish = time.time()

    # print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, centroids=None, featrue_center=None, tb=True):
    start = time.time()
    net.eval()
    cmi_correct = 0.0
    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        feature = net(images)
        
        feature = torch.nn.functional.normalize(feature-featrue_center(), dim=1)/args.feature_Temp
        # feature = feature/args.feature_Temp

        Log_feature = sigmax(feature, 1).log()
        # breakpoint()
        KL_divs = pairwiseKL(Log_feature, centroids.centroids.data.log())
        _, predicted = KL_divs.min(1)
        cmi_correct += predicted.eq(labels).sum().item()

    finish = time.time()

    # print('Test set: Epoch: {}, CMI Accuracy: {:.4f},Time consumed:{:.2f}s'.format(
    #     epoch,
    #     cmi_correct / len(cifar100_test_loader.dataset),
    #     finish - start
    # ))
    # print()

    #add informations to tensorboard

    return  cmi_correct / len(cifar100_test_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-wd', type=float, default=0.0, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')

    parser.add_argument('-centlr', type=float, default=1e-6, help='initial learning rate')
    parser.add_argument('-centmomentum', type=float, default=0.9, help='initial learning rate')
    parser.add_argument('-feature_Temp', type=float, default=1.0, help='initial learning rate')

    args = parser.parse_args()
    # print(args.lr, args.feature_Temp)


    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=10,
        batch_size=args.b,
        shuffle=True
    )
    
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=10,
        batch_size=args.b,
        shuffle=True
    )

    samples_data, samples_tar = [], []
    train_label_list = torch.tensor(cifar100_training_loader.dataset.targets)
    for Class in tqdm(range(100)):
        idx = (train_label_list == Class).nonzero().squeeze().numpy()
        sampler = SubsetRandomSampler(idx)
        class_loader = DataLoader(cifar100_training_loader.dataset, batch_size = 500, sampler = sampler, pin_memory = True)
        img, tar = next(iter(class_loader))
        samples_data.append(img.numpy())
        samples_tar.append(tar)
    samples_data = np.array(samples_data)
    samples_tar = torch.cat(samples_tar, 0)

    net = get_network(args)

    centroids = Centroid(n_classes= 100, n_dim=net.fc.in_features, samples_data=samples_data,
                         samples_tar = samples_tar, Ctemp=args.feature_Temp, CdecayFactor=0.9999).cuda()
    VALcentroids =  Centroid(n_classes= 100, n_dim=net.fc.in_features, samples_data=samples_data,
                         samples_tar = samples_tar, Ctemp=args.feature_Temp, CdecayFactor=0.9999).cuda()

    featrue_center = Center(n_dim=net.fc.in_features, CdecayFactor=0.9999)

    net.fc = nn.Identity()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    opt_cent = optim.SGD(centroids.parameters(), lr=args.centlr, momentum=args.centmomentum, weight_decay=0)
    # opt_cent = torch.optim.Adam(centroids.parameters(), lr=args.lr/100)

    # opt_cent = torch.optim.Adam(centroids.parameters(), lr=args.lr/100)

    # train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCH) #learning rate decay
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    cent_scheduler  = optim.lr_scheduler.MultiStepLR(opt_cent, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log

    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()


    #create checkpoint folder to save model
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_cmi_acc =  0.0

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
            cent_scheduler.step(epoch)
        net.train()
        
        train(epoch, centroids ,featrue_center, 30, args)
        net.eval()
        # print("val acc: centroids")
        # cmi_acc = eval_training(epoch=epoch, centroids=centroids, featrue_center=featrue_center)
        # print("val acc: VALcentroids")
        VALcentroids.update_epoch(net,featrue_center, cifar100_training_loader)
        cmi_acc = eval_training(epoch=epoch, centroids=VALcentroids, featrue_center=featrue_center)

        if best_cmi_acc<cmi_acc:
            # if cmi_acc<cmi_acc_val_cents:
            #     tmp_acc = cmi_acc_val_cents
            # else:
            #     tmp_acc = cmi_acc
            best_cmi_acc=cmi_acc
            # weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            # print('saving weights file to {}'.format(weights_path))
            # torch.save({'epoch': epoch,
            #             'state_dict': net.state_dict(),
            #             'center_dict': featrue_center.state_dict(),
            #             'centroid': VALcentroids.state_dict()},
            #             weights_path)

    print("Settings: lr:{}, Temp:{}, WD:{}, centlr:{}, centmomentum:{}".format(args.lr, args.feature_Temp, args.wd, args.centlr, args.centmomentum),
          "Best CMI acc --- : {}".format(best_cmi_acc))
