import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

MODELS = ['cor', 'dec', 'fcor', 'fdec']
SAMPLES = ['natural', 'pgd_conv_10', 'pgd_conv_50', 'pgd_conv_75', 'pgd_conv_100', 'pgd_conv_150']


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
fig, ax = plt.subplots(len(SAMPLES), 3, figsize=(12, 18))
for row, sample in enumerate(SAMPLES):
    for model in MODELS:
        score = np.load('{}/{}.npz'.format(model, sample))
        score, I = score['score'], score['I']
        Rcc, Riu, UA = get_curves(score, I, T)
        ax[row, 0].plot(T, Rcc, label=model)
        ax[row, 1].plot(T, Riu, label=model)
        ax[row, 2].plot(T, UA, label=model)
    ax[row, 0].set_ylabel(sample)
ax[0, 2].legend()
ax[0, 0].set_title(r'$R_{cc}$')
ax[0, 1].set_title(r'$R_{iu}$')
ax[0, 2].set_title(r'$UA$')
ax[5, 0].set_xlabel(r'$I_{T}$')
ax[5, 1].set_xlabel(r'$I_{T}$')
ax[5, 2].set_xlabel(r'$I_{T}$')

plt.savefig('PGD_results.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

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
    s3 = np.array(s3).transpose(1, 0)
    s = np.hstack((s1, s2, s3))
    s = pd.DataFrame(s, columns=columns)
    data = data.append(s, ignore_index=True)