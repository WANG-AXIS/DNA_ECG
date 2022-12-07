"""This script should be run after evalute_ensemble.py has been run for all models and samples.
This script calculates and saves the Uncertainty accuracy (UA), correct-certain ratio (Rcc) and
incorrect-uncertain ratio (Riu) for the specified networks and attacks."""

import numpy as np
import pandas as pd

MODELS = ['cor', 'dec', 'fcor', 'fdec']
SAMPLES = ['natural', 'pgd_10', 'pgd_50', 'pgd_75', 'pgd_100', 'pgd_150']


def get_curves(score, I, T):
    Rcc, Riu, UA = np.zeros(len(T)), np.zeros(len(T)), np.zeros(len(T))
    I[I<0] = 0
    I[I>1] = 1
    #I = (I-I.min())/(I.max()-I.min())
    for idx, t in enumerate(T):
        certain = I <= t
        Ncc = np.sum(score & certain)
        Nic = np.sum(~score & certain)
        Niu = np.sum(~score & ~certain)
        Ncu = np.sum(score & ~certain)
        Rcc[idx] = Ncc/(Ncc+Nic+.001)
        Riu[idx] = Niu/(Niu+Nic+.001)
        UA[idx] = (Ncc+Niu)/(Ncc+Niu+Ncu+Nic)
    return Rcc, Riu, UA


T = np.linspace(0, 1, num=100, endpoint=True)
columns = ['Model', 'Metric', 'Sample', 'Score']
data = pd.DataFrame([], columns=columns)


'''Create table'''
columns = ['Model', 'Metric', '0', '10', '50', '75', '100', '150']
data = pd.DataFrame([], columns=columns)
for model in MODELS:
    s1 = np.array([[model, model, model, model]]).transpose(1, 0)
    s2 = np.array([['Acc', 'Rcc', 'Riu', 'UA']]).transpose(1, 0)
    s3 = []
    for sample in SAMPLES:
        score = np.load('{}/{}.npz'.format(model, sample))
        score, I = score['score'], score['I']
        Rcc, Riu, UA = get_curves(score, I, T)
        s3.append([score.mean(), Rcc.mean(), Riu.mean(), UA.mean()])
    s3 = (100*np.array(s3)).transpose(1, 0).round(2)
    s = np.hstack((s1, s2, s3))
    s = pd.DataFrame(s, columns=columns)
    data = data.append(s, ignore_index=True)
data = data.sort_values(by='Metric')
data.to_csv('pgd.csv', index=False)