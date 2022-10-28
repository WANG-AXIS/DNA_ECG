import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
import utils.global_variables as gv
from utils.filters import filter1, filter2
MAX_SENTENCE_LENGTH = 18000

data_dirc = 'data/'
RAW_LABELS = np.load(data_dirc + 'raw_labels.npy')
PERMUTATION = np.load(data_dirc + 'random_permutation.npy')
BATCH_SIZE = 128
use_cuda = torch.cuda.is_available()
device = gv.device

data = np.load(data_dirc + 'raw_data.npy', allow_pickle=True)
data = data[PERMUTATION]
RAW_LABELS = RAW_LABELS[PERMUTATION]
mid = int(len(data) * 0.9)
data = data[mid:]
label = RAW_LABELS[mid:]
nat_loader = torch.utils.data.DataLoader(dataset=ECGDataset(data, label),
                                         batch_size=BATCH_SIZE,
                                         collate_fn=ecg_collate_func,
                                         shuffle=False)
names = ['cor', 'dec', 'fcor', 'fdec']
filters = [[None, None], [None, None], [filter1, filter2], [filter1, filter2]]
epsilons = [10, 50, 75, 100, 150]
attacks = ['pgd', 'pgd_conv']


def get_predictions(data_loader, model):
    model.eval()
    pred = np.array([])
    for (inputs, _, _) in data_loader:
        inputs_batch = inputs.to(device)
        output = model(inputs_batch)
        output = torch.argmax(output, dim=1).detach().cpu().numpy()
        pred = np.hstack([pred, output]) if pred.size else output
    return pred


nat_columns = ['name', 'Avg', 'P(1 model)', 'P(2 model)', 'P(3 model)']
adv_columns = ['name', 'Attack Type', r'$\epsilon$', 'Avg', 'P(1 model)', 'P(2 model)', 'P(3 model)']
nat_df = pd.DataFrame(columns=nat_columns)
adv_df = pd.DataFrame(columns=adv_columns)

for name, f_filter in zip(names, filters):
    nat_pred = [np.load('adv_exp/{}_{}/pred_nat.npy'.format(attacks[0], epsilons[0]))]
    model_dir = 'saved_model/{}_model'.format(name)
    for kdx, attack in enumerate(attacks):
        for jdx, epsilon in enumerate(epsilons):
            adv_dirc = 'adv_exp/{}_{}/'.format(attack, epsilon)
            adv_pred = [np.load(adv_dirc + 'pred_adv.npy')]
            data = torch.tensor(np.load(adv_dirc + 'data_adv.npy'))
            dataset = torch.utils.data.TensorDataset(data, MAX_SENTENCE_LENGTH*torch.ones(len(data)), torch.tensor(label))
            adv_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
            for idx, i in enumerate(['1.pth', '2.pth']):
                model = nn.DataParallel(CNN(num_classes=4, f_filter=f_filter[idx])).to(device)
                model.load_state_dict(torch.load(model_dir + i, map_location=lambda storage, loc: storage))
                if jdx == 0 and kdx == 0:
                    nat_pred.append(get_predictions(nat_loader, model))
                adv_pred.append(get_predictions(adv_loader, model))
            if jdx == 0 and kdx == 0:
                n0 = nat_pred[0] == label
                n1 = nat_pred[1] == label
                n2 = nat_pred[2] == label
                nat_score = np.sum(np.vstack([n0, n1, n2]), axis=0)
                navg = np.mean(nat_score) / 3 * 100
                pn3 = np.sum(nat_score >= 3) / len(nat_score) * 100
                pn2 = np.sum(nat_score >= 2) / len(nat_score) * 100
                pn1 = np.sum(nat_score >= 1) / len(nat_score) * 100
                nat_df = nat_df.append(pd.DataFrame([[name, navg, pn1, pn2, pn3]], columns=nat_columns))
            a0 = adv_pred[0] == label
            a1 = adv_pred[1] == label
            a2 = adv_pred[2] == label
            a0 = a0[n0]
            a1 = a1[n0]
            a2 = a2[n0]
            adv_score = np.sum(np.vstack([a0, a1, a2]), axis=0)
            aavg = np.mean(adv_score)/3 * 100
            pa3 = np.sum(adv_score >= 3) / len(adv_score) * 100
            pa2 = np.sum(adv_score >= 2) / len(adv_score) * 100
            pa1 = np.sum(adv_score >= 1) / len(adv_score) * 100
            adv_df = adv_df.append(pd.DataFrame([[name, attack, epsilon, aavg, pa1, pa2, pa3]], columns=adv_columns))
            print([name, attack, epsilon, aavg, pa1, pa2, pa3])


nat_df.to_pickle('nat_df.pkl')
adv_df.to_pickle('adv_df.pkl')