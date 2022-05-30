import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
import sys
import os

data_dirc = 'data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy')
PERMUTATION = np.load(data_dirc+'random_permutation.npy')
BATCH_SIZE = 80
MAX_SENTENCE_LENGTH = 18000
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
file_name = 'best_model'
epsilons = [10, 50, 75, 100, 150]
stp_alphas = [1, 5, 7.5, 10, 15]

data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
data = data[PERMUTATION]
RAW_LABELS = RAW_LABELS[PERMUTATION]
mid = int(len(data)*0.9)
val_data = data[mid:]
val_label = RAW_LABELS[mid:]
val_dataset = ECGDataset(val_data, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, collate_fn=ecg_collate_func, shuffle=False)

model = CNN(num_classes=4)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('saved_model/{}.pth'.format(file_name), map_location=lambda storage, loc: storage))


def pgd_conv(inputs, lengths, targets, model, criterion, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())
    for i in range(num_steps):
        output = model(crafting_input)
        loss = criterion(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward()
        added = torch.sign(crafting_input.grad.data)
        step_output = crafting_input + step_alpha * added
        total_adv = step_output - inputs
        total_adv = torch.clamp(total_adv, -eps, eps)
        crafting_output = inputs + total_adv
        crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
    added = crafting_output - inputs
    added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    for i in range(num_steps*2):
        temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
        for j in range(len(sizes)-1):
            temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
        temp = temp/float(len(sizes))
        output = model(inputs + temp)
        loss = criterion(output, targets)
        loss.backward()
        added = added + step_alpha * torch.sign(added.grad.data)
        added = torch.clamp(added, -eps, eps)
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
    for j in range(len(sizes)-1):
        temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
    temp = temp/float(len(sizes))
    crafting_output = inputs + temp.detach()
    crafting_output_clamp = crafting_output.clone()
    for i in range(crafting_output_clamp.size(0)):
        remainder = MAX_SENTENCE_LENGTH - lengths[i]
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    sys.stdout.flush()
    return  crafting_output_clamp


def save_adv_samples(data_loader, model, eps=1, step_alpha=1, num_steps=20):
    model.eval()
    pred_nat = np.array([])
    pred_adv = np.array([])
    data_adv = np.array([])
    total = 0.0

    for bi, (inputs, lengths, targets) in enumerate(data_loader):
        inputs_batch, lengths_batch, targets_batch = inputs.to(device), lengths.to(device), targets.to(device)
        crafted_clamp = pgd_conv(inputs_batch, lengths_batch, targets_batch, model, F.cross_entropy, eps,
                                 step_alpha, num_steps, sizes=crafting_sizes, weights=crafting_weights)
        pred = model(inputs_batch)
        pred_clamp = model(crafted_clamp)
        crafted_clamp = crafted_clamp.detach().cpu().numpy()
        pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
        pred_clamp = torch.argmax(pred_clamp, dim=1).detach().cpu().numpy()
        total += np.sum(pred == targets.numpy())
        print(total/len(val_label))
        data_adv = np.vstack([data_adv, crafted_clamp]) if data_adv.size else crafted_clamp
        pred_adv = np.hstack([pred_adv, pred_clamp]) if pred_adv.size else pred_clamp
        pred_nat = np.hstack([pred_nat, pred]) if pred_nat.size else pred

    path = 'adv_exp/pgd_conv_'+str(eps)
    if os.path.isdir(path) is False:
        os.mkdir(path)
    np.save(path+'/data_adv.npy', data_adv)
    np.save(path+'/pred_nat.npy', pred_nat)
    np.save(path+'/pred_adv.npy', pred_adv)
    return None


sizes = [5, 7, 11, 15, 19]
sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
crafting_sizes = []
crafting_weights = []
for size in sizes:
    for sigma in sigmas:
        crafting_sizes.append(size)
        weight = np.arange(size) - size//2
        weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
        weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
        crafting_weights.append(weight)

for epsilon, stp_alpha in zip(epsilons, stp_alphas):
    save_adv_samples(val_loader, model, eps=epsilon, step_alpha=stp_alpha, num_steps=20)
print('Done')
sys.stdout.flush()
