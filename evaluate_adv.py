"""This script is used for crafting adversarial attacks against a model,
evaluating them with an ensemble, and saving the results."""
import numpy as np
import sys
import torch
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
from train.ensemble import Ensemble
from utils.craft_attack import craft_attack

save_dir = 'adv_results/sample_run'
ensemble_dir = 'runs/sample_run'
ensemble_kwargs = {'architecture': CNN,
                   'num_models': 3,
                   'input_channels': 1,
                   'num_classes': 4,
                   'filters': None}

# Physionet 2017 Settings:
data_dir = 'data/'
max_length = 18000
epsilons = [0, 10, 50, 75, 100, 150]
stp_alphas = [0, 1, 5, 7.5, 10, 15]
sizes = [5, 7, 11, 15, 19]
sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]

# CPSC 2018 Settings:
# data_dir = 'data/china_2018/'
# max_length = 24000
# sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
# epsilons = np.array([0, 1, 2, 4, 6, 7]) / 40
# stp_alphas = np.array([0, 1, 2, 4, 6, 7]) / 400
# sizes = [9, 11, 15, 19, 21]
# sigmas = [5.0, 7.0, 10.0, 13.0, 17.0]

num_steps = 20
smooth_attack = False
device_str = 'cuda:0'
batch_size = 15


def main():
    data = np.load(f'{data_dir}raw_data.npy', allow_pickle=True)
    labels = np.load(f'{data_dir}raw_labels.npy')
    permutation = np.load(f'{data_dir}random_permutation.npy')
    data, labels = data[permutation], labels[permutation]
    mid = int(len(data)*0.9)
    val_data, val_label = data[mid:], labels[mid:]
    val_dataset = ECGDataset(val_data, val_label, max_length)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             collate_fn=lambda x: ecg_collate_func(x, max_length),
                                             shuffle=False)
    device = torch.device(device_str)
    ensemble = Ensemble(**ensemble_kwargs)
    ensemble.load_models(ensemble_dir)

    crafting_sizes = []
    crafting_weights = []
    for size in sizes:
        for sigma in sigmas:
            crafting_sizes.append(size)
            weight = np.arange(size) - size // 2
            weight = np.exp(-weight ** 2.0 / 2.0 / (sigma ** 2)) /\
                np.sum(np.exp(-weight ** 2.0 / 2.0 / (sigma ** 2)))
            weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
            crafting_weights.append(weight)

    predictions = []
    I = []
    for epsilon, stp_alpha in zip(epsilons, stp_alphas):
        eps_predictions = np.array([], dtype=np.int32)
        eps_I = np.array([], dtype=np.float32)
        for (inputs, lengths, targets) in val_loader:
            inputs, lengths, targets = inputs.to(device), lengths.to(device), targets.to(device)

            if epsilon != 0:
                inputs = craft_attack(inputs, lengths, targets,
                                      ensemble.models[0].to(device), max_length,
                                      epsilon, stp_alpha, num_steps,
                                      smooth_attack, crafting_sizes, crafting_weights)

            batch_pred, batch_I = ensemble.infer(inputs, batch_size=batch_size, device=device)
            eps_predictions = np.hstack([eps_predictions, batch_pred]) if eps_predictions.size else batch_pred
            eps_I = np.hstack([eps_I, batch_I]) if eps_I.size else batch_I

        predictions.append(eps_predictions)
        I.append(eps_I)
        print(f'Done computing epsilon={epsilon}.')
        sys.stdout.flush()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(f'{save_dir}/predictions.npy', np.array(predictions))
    np.save(f'{save_dir}/uncertainties.npy', np.array(I))
    np.save(f'{save_dir}/labels.npy', val_label)

    all_settings = {'ensemble_dir': ensemble_dir,
                    'ensemble_kwargs': ensemble_kwargs,
                    'data_dir': data_dir,
                    'epsilons': epsilons,
                    'stp_alphas': stp_alpha,
                    'smooth_attack': smooth_attack,
                    'sizes': sizes,
                    'sigmas': sigmas}
    with open(f'{save_dir}/settings.yaml', 'w') as file:
        yaml.dump(all_settings, file)

    print('Done')
    sys.stdout.flush()


if __name__ == "__main__":
    main()
