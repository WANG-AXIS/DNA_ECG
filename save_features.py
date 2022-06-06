import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import sys
import utils.global_variables as gv
sys.path.append('utils')
from model.CNN import CNN
from utils.create_data import create_data
import os
from utils.filters import filter1, filter2
'''Data settings'''
AUGMENTED = True
RATIO = 19
PREPROCESS = 'zero'
data_dirc = 'data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy')
PERMUTATION = np.load(data_dirc+'random_permutation.npy')
BATCH_SIZE = 32
MAX_SENTENCE_LENGTH = 18000
device = gv.device
PADDING = 'two'
file_name = 'best_model'
f_filter = None
#f_filter = filter1


def get_features(file_name):
    print('#### Extracting Features ####')
    data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
    train_data, train_label, val_data, val_label = create_data(data, RAW_LABELS, PERMUTATION, RATIO, PREPROCESS,
                                                               MAX_SENTENCE_LENGTH, AUGMENTED, PADDING)
    dataset = torch.utils.data.TensorDataset(train_data, train_label)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = CNN(num_classes=4, f_filter=f_filter)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load('saved_model/{}.pth'.format(file_name), map_location=lambda storage, loc: storage))
    model.eval()
    features = np.array([])
    for i, (data, labels) in enumerate(loader):
        data_batch, label_batch = data.to(device),  labels.to(device)
        _, feature_batch = model(data_batch, return_features=True)
        feature_batch = feature_batch.detach().cpu().numpy()
        features = np.vstack([features, feature_batch]) if features.size else feature_batch
    if os.path.isdir('saved_features/') is False:
        os.mkdir('saved_features')
    path = 'saved_features/' + file_name + '.npy'
    np.save(path, features)
    print('#### Features Saved to {} ####'.format(path))
    print(features.shape)
    print(train_data.shape)


get_features(file_name)
