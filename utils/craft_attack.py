"""The function in this script is modified from Han, et al.
(https://github.com/XintianHan/ADV_ECG) """

import torch
import torch.nn.functional as F
import torch.nn as nn
import sys


def craft_attack(inputs, lengths, targets, model, max_length=18000,
                 eps=None, step_alpha=None, num_steps=None,
                 smooth_attack=False, sizes=None, weights=None,
                 loss_fn=nn.CrossEntropyLoss(), minimize=False):
    """This function creates white-box adversarial attacks.
    inputs (tensor): batch of N inputs
    lengths (tensor): length N tensor with each (original) input length
    targets (tensor): correct label for each input
    model (nn.module): NN for attack to target
    eps: Attack strength epsilon
    step_alpha: step size for iterative attach optimization
    num_steps: Number of optimization steps for each attack
    smooth_attack (bool): crafts smooth adversarial perturbations is True, otherwise, crafts regular PGD attacks
    sizes: sizes of the smoothing kernels used (SAP only)
    weights: weights for the smoothing kernels used (SAP only)
    loss_fn: loss function used for optimization
    maximize (bool): states whether to minimize (True) or maximize (False) the loss_fn."""
    step_alpha = -step_alpha if minimize else step_alpha
    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())

    for i in range(num_steps):
        output = model(crafting_input)
        loss = loss_fn(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward()
        added = torch.sign(crafting_input.grad.data)
        step_output = crafting_input + step_alpha * added
        total_adv = step_output - inputs
        total_adv = torch.clamp(total_adv, -eps, eps)
        crafting_output = inputs + total_adv
        crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)

    if smooth_attack:
        added = crafting_output - inputs
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        for i in range(num_steps*2):
            temp = smooth_perturbation(added, weights, sizes)
            output = model(inputs + temp)
            loss = loss_fn(output, targets)
            loss.backward()
            added = added + step_alpha * torch.sign(added.grad.data)
            added = torch.clamp(added, -eps, eps)
            added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        temp = smooth_perturbation(added, weights, sizes)
        crafting_output = inputs + temp.detach()

    crafting_output_clamp = crafting_output.clone()
    if lengths is not None:
        for i in range(crafting_output_clamp.size(0)):
            remainder = max_length - lengths[i]
            if remainder > 0:
                crafting_output_clamp[i][0][:int(remainder / 2)] = 0
                crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    sys.stdout.flush()
    return crafting_output_clamp.detach()


def smooth_perturbation(added, weights, sizes):
    """Smooths noise (added) using kernels as defined by weights"""
    temp = 0
    num_channels = len(added[0])
    for j in range(len(sizes)):
        smooth_kernel = weights[j].repeat(num_channels, 1, 1)
        temp += F.conv1d(added, smooth_kernel, padding=sizes[j] // 2, groups=num_channels)
    temp = temp / float(len(sizes))
    return temp
