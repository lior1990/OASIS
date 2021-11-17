from enum import Enum

import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vggloss import VGG19


class TargetMode(Enum):
    REAL = "real"
    FAKE = "fake"
    OTHER = "other"


class losses_computer():
    def __init__(self, opt):
        self.opt = opt
        if not opt.no_labelmix:
            self.labelmix_function = torch.nn.MSELoss()

    def loss(self, input, label, target_mode: "TargetMode"):
        #--- n+2 loss ---
        target = get_n2_target(self.opt, input, label, target_mode)
        loss = F.cross_entropy(input, target, reduction='none')
        if target_mode == TargetMode.REAL:
            # --- balancing classes ---
            weight_map = get_class_balancing(self.opt, input, label)
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)


def get_class_balancing(opt, input, label):
    if not opt.no_balancing_inloss:
        class_occurence = torch.sum(label, dim=(0, 2, 3))
        if opt.contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=1, keepdim=True)
        if opt.contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = torch.ones_like(input[:, :, :, :])
    return weight_map


def get_n2_target(opt, input, label, target_mode: "TargetMode"):
    target_is_real = target_mode in [TargetMode.REAL, TargetMode.OTHER]

    if target_mode == TargetMode.REAL:
        integers = torch.argmax(label, dim=1)
        return integers + 2  # convert labels to start from 2
    elif target_mode in [TargetMode.FAKE, TargetMode.OTHER]:
        # other -> label 1
        # fake -> label 0
        input_spatial_dim = (input.shape[0], input.shape[2], input.shape[3])
        return get_target_tensor(opt, input_spatial_dim, target_is_real)
    else:
        raise NotImplementedError


def get_target_tensor(opt, input, target_is_real):
    if opt.gpu_ids != "-1":
        if target_is_real:
            return torch.cuda.LongTensor(1).fill_(1.0).requires_grad_(False).expand(input)
        else:
            return torch.cuda.LongTensor(1).fill_(0.0).requires_grad_(False).expand(input)
    else:
        if target_is_real:
            return torch.LongTensor(1).fill_(1.0).requires_grad_(False).expand(input)
        else:
            return torch.LongTensor(1).fill_(0.0).requires_grad_(False).expand(input)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if gpu_ids:
            self.vgg = self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
