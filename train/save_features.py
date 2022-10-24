import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import sys
from model.CNN import CNN
from utils.create_data import create_data
import os
sys.path.append('utils')


def get_features(file_name, active_filter, batch_size=32):
    print('#### Extracting Features ####')
    data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
    train_data, train_label, val_data, val_label = create_data(data, RAW_LABELS, PERMUTATION, RATIO, PREPROCESS,
                                                               MAX_SENTENCE_LENGTH, AUGMENTED, PADDING)
    dataset = torch.utils.data.TensorDataset(train_data, train_label)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    model = CNN(num_classes=4, f_filter=active_filter)
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
