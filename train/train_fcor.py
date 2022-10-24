import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import sys
import utils.global_variables as gv
from utils.test_model import cal_F1
from model.CNN import CNN
from utils.create_data import create_data
from save_features import get_features
from utils.filters import filter1, filter2
sys.path.append('utils')

'''Training settings'''
AUGMENTED = True
RATIO = 19
PREPROCESS = 'zero'
data_dirc = '../data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy')
PERMUTATION = np.load(data_dirc+'random_permutation.npy')
BATCH_SIZE = 80
MAX_SENTENCE_LENGTH = 18000
device = gv.device
LEARNING_RATE = 0.001
NUM_EPOCHS = 200  # number epoch to train
PADDING = 'two'
FILE_NAME = 'fcor_model2'
active_filter = filter2


def train_model():
    print('#### Start Training {} ####'.format(FILE_NAME))
    data = np.load(data_dirc+'raw_data.npy', allow_pickle=True)
    train_data, train_label, val_data, val_label = create_data(data, RAW_LABELS, PERMUTATION, RATIO, PREPROCESS, MAX_SENTENCE_LENGTH, AUGMENTED, PADDING)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    model = CNN(num_classes=4, f_filter=active_filter)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_acc = 0.0
    acc_history = np.zeros([NUM_EPOCHS])
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            model.train()
            data_batch, label_batch, features = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, model_feat = model(data_batch, return_features=True)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            torch.cuda.empty_cache()
        train_loss /= len(train_loader.sampler)
        # validate
        val_acc, val_F1 = cal_F1(val_loader, model)
        if val_acc > best_acc:
            best_acc = val_acc
            best_F1 = val_F1
            torch.save(model.state_dict(), '../saved_model/' + FILE_NAME + '.pth')
        acc_history[epoch] = val_acc
        np.savez('../saved_model/histories/' + FILE_NAME + '.npz', acc_history=acc_history)
        print('Epoch: [{}/{}], Val Acc: {}, Val F1: {}, Train Loss: {}'.format(
            epoch + 1, NUM_EPOCHS, val_acc, val_F1, train_loss))
        sys.stdout.flush()
    print('#### End Training ####')
    print('best val acc:', best_acc)
    print('best F1:', best_F1)


train_model()
get_features(FILE_NAME, active_filter)
