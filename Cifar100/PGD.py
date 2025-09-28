import torch
import torch.nn.functional as F
from tqdm import tqdm
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
import math
Nmean = torch.tensor(settings.CIFAR100_TRAIN_MEAN).view(-1, 1, 1).cuda()
NSTD = torch.tensor(settings.CIFAR100_TRAIN_STD).view(-1, 1, 1).cuda()
def denormalize(tensor):
    return tensor * NSTD + Nmean
normalize = transforms.Normalize(mean=settings.CIFAR100_TRAIN_MEAN, std=settings.CIFAR100_TRAIN_STD)


def pgd_attack(model, centroids, featrue_center, images, labels, eps=0.3, alpha=2/255, iters=40, clamp_min=0.0, clamp_max=1.0):
    """
    Perform PGD attack on a batch of images.
    
    Args:
        model: The neural network to attack.
        images: Input images (tensor) of shape (B, C, H, W).
        labels: True labels for the input images.
        eps: Maximum perturbation (L-infinity norm).
        alpha: Step size per iteration.
        iters: Number of PGD iterations.
        clamp_min: Minimum pixel value.
        clamp_max: Maximum pixel value.
    
    Returns:
        adversarial_images: Perturbed adversarial examples.
    """
    # Make a copy of the original images
    original_images = denormalize(images.clone().detach())

    # Start from a random point within the epsilon-ball
    perturbed_images = original_images + torch.empty_like(images).uniform_(-eps, eps)
    perturbed_images = torch.clamp(perturbed_images, clamp_min, clamp_max)
    perturbed_images.requires_grad = True

    for _ in tqdm(range(iters)):
        feature = model(normalize(perturbed_images))
        feature = torch.nn.functional.normalize(feature-featrue_center(), dim=1)/args.feature_Temp
        
        Log_feature = sigmax(feature, 1).log()
        KL_divs = pairwiseKL(Log_feature, centroids.centroids.cuda().log())
        # breakpoint()
        CMI = KL_divs[torch.arange(KL_divs.size(0)), labels]
        KL_divs[torch.arange(KL_divs.size(0)), labels] = 0.0
        SEP = KL_divs.mean(-1)
        loss =torch.nan_to_num(CMI/SEP).mean()
     

        # Compute gradient
        loss.backward()
        grad = perturbed_images.grad.data

        # PGD update and projection
        perturbed_images = perturbed_images + alpha * grad.sign()
        perturbed_images = torch.max(torch.min(perturbed_images, original_images + eps), original_images - eps)
        perturbed_images = torch.clamp(perturbed_images, clamp_min, clamp_max).detach()
        perturbed_images.requires_grad = True

    return normalize(perturbed_images)




def eval_Robustness(epoch=0, centroids=None, featrue_center=None, tb=True):
    start = time.time()
    net.eval()
    cmi_correct = 0.0
    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        eps = 8/255
        alpha = eps/3
        perturbed_images = pgd_attack(net, centroids, featrue_center, images, labels, eps=eps, alpha=alpha, iters=5, clamp_min=0.0, clamp_max=1.0).detach()
        feature = net(perturbed_images)
        Log_feature = sigmax(feature, 1).log()
        KL_divs = pairwiseKL(Log_feature, centroids.centroids.cuda().log())
        _, predicted = KL_divs.min(1)
        cmi_correct += predicted.eq(labels).sum().item()
    finish = time.time()

    print('Test set: Epoch: {}, Robust CMI Accuracy: {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        cmi_correct / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    return  cmi_correct / len(cifar100_test_loader.dataset)


@torch.no_grad()
def eval_training(epoch=0, centroids=None, featrue_center=None, tb=True):
    start = time.time()
    net.eval()
    cmi_correct = 0.0
    for (images, labels) in tqdm(cifar100_test_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        feature = net(images)

        feature = torch.nn.functional.normalize(feature-featrue_center(), dim=1)/args.feature_Temp
        # feature = feature/args.feature_Temp

        Log_feature = sigmax(feature, 1).log()

        KL_divs = pairwiseKL(Log_feature, centroids.centroids.cuda().log())
        _, predicted = KL_divs.min(1)
        cmi_correct += predicted.eq(labels).sum().item()
    # breakpoint()

    finish = time.time()

    print('Test set: Epoch: {}, CMI Accuracy: {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        cmi_correct / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    return  cmi_correct / len(cifar100_test_loader.dataset)

class Center(nn.Module):
    def __init__(self, n_dim, CdecayFactor=0.999):
        super(Center, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_dim = n_dim
        self.Center = torch.zeros((1, self.n_dim)).to(self.device)
        self.CdecayFactor = CdecayFactor
        self.virgin = True
    def update(self, feature):
        if self.virgin:
            self.Center = feature.detach().mean(dim=tuple(range(feature.dim() - 1)))
            self.virgin = False
        else:
            self.Center = self.CdecayFactor*self.Center + \
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


class Center(nn.Module):
    def __init__(self, n_dim, CdecayFactor=0.999):
        super(Center, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_dim = n_dim
        self.Center = nn.Parameter(torch.zeros(self.n_dim).to(self.device))
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
        return self.Center.data

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


class Centroid(nn.Module):
    def __init__(self, n_classes, n_dim, samples_data, samples_tar, Ctemp, CdecayFactor=0.999):
        super().__init__()
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.centroids = torch.eye(self.n_classes)
        self.centroids = F.pad(self.centroids, (0, self.n_dim-self.n_classes))+0.01
        self.centroids = nn.Parameter(self.centroids/(self.centroids.sum(-1)[:,None]))
        self.CdecayFactor = CdecayFactor
        self.Ctemp = Ctemp
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.samples_data = None
        self.samples_tar = None
    def update_batch(self, model, targets,featrue_center , sample_size=8):      
        model.train()
        with torch.no_grad():
            uni_targets = torch.unique(targets)
            idx = torch.randperm(500)[:sample_size].to(self.device)
            img = torch.index_select(self.samples_data, 1, idx).view(-1,3,32,32)
            logits = model(img)
            logits = logits.detach()
            featrue_center.update(logits)
            logits = torch.nn.functional.normalize((logits-featrue_center()), 1)/(self.Ctemp)
            # logits = logits/self.Ctemp
            output = sigmax(logits.float(), 1)
            for Class in range(self.n_classes):
                self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
                    (1- self.CdecayFactor) * torch.mean(output[Class * sample_size : (Class + 1) * sample_size], axis = 0).detach().cpu()
        self.centroids.data =  self.centroids.data/(self.centroids.data.sum(1)[:,None])

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
                Classes =  target.cpu().unique()
                logits = logits.cpu()
                output = sigmax(logits.float(), 1)
                for Class in Classes:
                    self.centroids.data[Class] += torch.sum(output[target.cpu() == Class], axis = 0)
        self.centroids.data =  self.centroids.data/(self.centroids.data.sum(1)[:,None])
    def get_centroids(self, target):
        return torch.index_select(self.centroids.data, 0, target.cpu()).to(target.device)


def pairwiseKL(Log_Prob, Log_Cent_All):

    Log_Prob    = Log_Prob.unsqueeze(1)
    Log_Cent_All = Log_Cent_All.unsqueeze(0) 
    D = (Log_Prob.exp()* (Log_Prob-Log_Cent_All)).sum(2)
    return D



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-ckpt', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-step_size', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')


    parser.add_argument('-feature_Temp', type=float, default=1.0, help='initial learning rate')

    args = parser.parse_args()



    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=10,
        batch_size=args.b,
        shuffle=True
    )

    net = get_network(args)


    centroids = Centroid(n_classes= 100, n_dim=net.fc.in_features, samples_data=None,
                         samples_tar = None, Ctemp=args.feature_Temp, CdecayFactor=0.9999)
    featrue_center = Center(n_dim=net.fc.in_features, CdecayFactor=0.9999)
    net.fc = nn.Identity()
    # breakpoint()
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.ckpt)
    ckpt = torch.load(checkpoint_path)
    net.load_state_dict(ckpt['state_dict'])
    featrue_center.load_state_dict(ckpt['center_dict'])
    centroids.load_state_dict(ckpt['centroid'])
    net.eval()

    cmi_acc = eval_training(epoch=0, centroids=centroids, featrue_center=featrue_center)
    print(cmi_acc)

    cmi_acc = eval_Robustness(epoch=0, centroids=centroids, featrue_center=featrue_center)
    
    print(cmi_acc)


