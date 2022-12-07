"""This script evaluates and saves the prediction accuracy (correct/incorrect) and
 uncertainty for each sample is a dataset for an ensemble"""
import numpy as np
import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
import torch.nn as nn
import sys
sys.path.append('../')
from tqdm import tqdm
import utils.global_variables as gv
from model.CNN import CNN
from utils.create_data import create_data
from utils.filters import filter1, filter2


BATCH_SIZE = 500
MODEL_DIR = '../saved_model/'
ADV_DIR = '../adv_exp/'
MODELS = ['base_model', 'fdec_model1', 'fdec_model2']
FILTERS = [None, filter1, filter2]
SAVE_DIR = 'fdec/'
ADV_SETS = ['pgd_conv_10', 'pgd_conv_50', 'pgd_conv_75', 'pgd_conv_100', 'pgd_conv_150',
            'pgd_10', 'pgd_50', 'pgd_75', 'pgd_100', 'pgd_150']
ETA = 1e-30 #constant for log stability
device = gv.device


def save_pred(score, I, dir, name):
    if os.path.isdir(dir) is False:
        os.mkdir(dir)
    np.savez(dir+name+'.npz', score=score, I=I)
    return None


def get_output(model, data, label):
    model.eval()
    dataset = torch.utils.data.TensorDataset(data, label)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    output = np.array([])
    with torch.no_grad():
        for inputs, _ in tqdm(loader):
            inputs = inputs.to(device)
            out = F.softmax(model(inputs), dim=1).detach().cpu().numpy()
            output = np.vstack([output, out]) if output.size else out
    return output


def get_uncertainty(data, label):
    outputs = []
    T = len(MODELS)
    for model_name, filter in zip(MODELS, FILTERS):
        model = CNN(num_classes=4, f_filter=filter)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(MODEL_DIR + model_name + '.pth', map_location=lambda storage, loc: storage))
        output = get_output(model, data, label)
        outputs.append(output)
    outputs = np.array(np.array(outputs))
    outputs = np.transpose(outputs, (1, 0, 2))
    y_hat = np.sum(outputs, axis=1) / T
    score = np.argmax(y_hat, axis=1) == label.numpy()
    H_hat = -1*np.sum(y_hat * np.log(y_hat + ETA), axis=1)
    I = H_hat + np.sum(outputs * np.log(outputs + ETA), axis=(1, 2))/T
    return score, I


'''Load Data'''
data = np.load('../data/raw_data.npy', allow_pickle=True)
raw_labels = np.load('../data/raw_labels.npy')
permutation = np.load('../data/random_permutation.npy')
train_data, train_label, val_data, val_label = create_data(data, raw_labels, permutation, 19, 'zero',
                                                           18000, True, 'two')

val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""Get Imin, Imax from training set"""
_, I = get_uncertainty(train_data, train_label)
Imin, Imax = I.min(), I.max()
"""evaluate on natural"""
score, I = get_uncertainty(val_data, val_label)
#Imin, Imax = I.min(), I.max()
I = I-Imin/(Imax-Imin)
save_pred(score, I, SAVE_DIR, 'natural')
"""evaluate on adversarial"""
#for i in ADV_SETS:
#    adv_data = torch.tensor(np.load(ADV_DIR+i+'/data_adv.npy'))
#    score, I = get_uncertainty(adv_data, val_label)
   # Imin, Imax = I.min(), I.max()
#    I = I - Imin / (Imax - Imin)
#    save_pred(score, I, SAVE_DIR, i)
