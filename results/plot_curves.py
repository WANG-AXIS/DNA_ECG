"""This runs experiment with mixed dataset to see the number of misclassified samples."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# China 2018
epsilons = np.array([0, 1, 2, 4, 6, 7])/40  # from yaml files
ensemble_names_china = ['cor_china', 'dec_china', 'fcor_china', 'fdec_china',
                        'dverge_china', 'cor_adv_china', 'dec_adv_china']
table_names_china = ['baseline', 'dec', 'part', 'dec+part',
                     'dverge', 'adv', 'dec+adv']

# Physionet 2017
epsilons = np.array([0, 10, 50, 75, 100, 150])  # from yaml files
ensemble_names = ['cor_test_run', 'dec_test_run', 'fcor_test', 'fdec_test',
                  'dverge', 'adversarial_training_test', 'dec_adv']
table_names = ['baseline', 'dec', 'part', 'dec+part',
               'dverge', 'adv', 'dec+adv']

# Define probability mass function for probability of each attack strength
# attack_pmf = np.array([0.5, 0.25, 0.15, .1, 0, 0])
attack_pmf = np.array([1., 0, 0, 0, 0, 0])  # Natural samples only
attack_cdf = np.cumsum(attack_pmf)


def plot_curves(ensemble_names, ax, set_legend=False):
    correct_masks = []
    for ensemble_name in ensemble_names:
        data_dir = f'../adv_exp_new/{ensemble_name}_sap'
        labels = np.load(f'{data_dir}/labels.npy')
        predictions = np.load(f'{data_dir}/predictions.npy')
        uncertainties = np.load(f'{data_dir}/uncertainties.npy')

        n = len(labels)
        u = np.random.rand(n)
        attack_indices = np.searchsorted(attack_cdf, u)
        predictions = np.array([predictions[attack_indices[i], i] for i in range(n)])
        uncertainties = np.array([uncertainties[attack_indices[i], i] for i in range(n)])
        order = np.argsort(uncertainties)#[::-1]
        predictions, uncertainties, labels = predictions[order], uncertainties[order], labels[order]

        correct_mask = predictions == labels
        correct_masks.append(correct_mask)

    correct_masks = np.array(correct_masks).transpose(1, 0)
    num_incorrect = np.cumsum(~correct_masks, axis=0)
    aucs = np.sum(num_incorrect/n, axis=0)/n
    for ensemble_name, auc in zip(ensemble_names, aucs):
        print(f'{ensemble_name}: {auc}')

    ax.plot(np.linspace(0, 100, n), 100*num_incorrect/n)
    ax.set_xlabel('Cases Deferred to DNN (%)')
    if set_legend:
        ax.legend(table_names, fontsize=9)


sns.set_theme()
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
print('Physionet 2017')
plot_curves(ensemble_names, ax[0])
print('CPSC 2018')
plot_curves(ensemble_names_china, ax[1], set_legend=True)
ax[0].set_ylabel('Misclassified Cases (%)')


plt.savefig('plot_curve.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
