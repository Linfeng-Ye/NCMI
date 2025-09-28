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


def pgd_attack(model,  images, labels, eps=0.3, alpha=2/255, iters=40, clamp_min=0.0, clamp_max=1.0):
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
        loss = F.cross_entropy(feature, labels)

        # Compute gradient
        loss.backward()
        grad = perturbed_images.grad.data

        # PGD update and projection
        perturbed_images = perturbed_images + alpha * grad.sign()
        perturbed_images = torch.max(torch.min(perturbed_images, original_images + eps), original_images - eps)
        perturbed_images = torch.clamp(perturbed_images, clamp_min, clamp_max).detach()
        perturbed_images.requires_grad = True

    return normalize(perturbed_images)




def eval_Robustness(epoch=0, tb=True):
    start = time.time()
    net.eval()
    correct = 0.0
    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        eps = 16/255
        alpha = eps/3
        perturbed_images = pgd_attack(net, images, labels, eps=eps, alpha=alpha, iters=5, clamp_min=0.0, clamp_max=1.0).detach()
        feature = net(perturbed_images)
        
        _, preds = feature.max(1)
        correct += preds.eq(labels).sum()
        
    finish = time.time()

    print('Test set: Epoch: {}, Robust CMI Accuracy: {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        correct / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    return  correct / len(cifar100_test_loader.dataset)


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()
    correct = 0.0
    for (images, labels) in tqdm(cifar100_test_loader):
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)

        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    # breakpoint()

    finish = time.time()

    print('Test set: Epoch: {}, CMI Accuracy: {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        correct / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    return  correct / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-ckpt', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-step_size', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')


   

    args = parser.parse_args()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=10,
        batch_size=args.b,
        shuffle=True)

    net = get_network(args)

    # breakpoint()
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, args.ckpt)
    ckpt = torch.load(checkpoint_path)
    net.load_state_dict(ckpt)

    net.eval()

    
    cmi_acc = eval_training(epoch=0)
    print(cmi_acc)

    cmi_acc = eval_Robustness(epoch=0)
    
    print(cmi_acc)


