"""This script is training an ensemble model"""
import numpy as np
import sys
import yaml
from pathlib import Path
from ensemble import Ensemble
sys.path.append('../')
from utils.create_data import create_data
from model.CNN import CNN


'''Training settings
save_dir: directory in which to save trained ensemble
data_dir: directory location for data
max_training_minutes: set maximum number of training minutes per model in ensemble.
If None, then training will run for the full number of epochs.'''

save_dir = 'runs/sample_run'
data_dir = '../data/data_dir'
max_training_minutes = None

ensemble_kwargs = {'architecture': CNN,
                   'num_models': 3,
                   'input_channels': 1,
                   'num_classes': 4,
                   'filters': None,
                   'save_dir': save_dir}

load_model_kwargs = None #Set to None if training from scratch
# load_model_kwargs = {'load_dir': 'None',
#                      'model_nums': None}

data_kwargs = {'ratio': 19,
               'preprocess': 'zero',
               'max_length': 18000,
               'augmented': True,
               'padding': 'two'}

# training_kwargs = None
training_kwargs = {'batch_size': 64,
                   'epochs': 80,
                   'learning_rate': 0.001,
                   'optimizer_name': 'Adam',
                   'loss_fn': 'CrossEntropyLoss',
                   'device': 'cuda',
                   'decorrelate_weight': 0.0,
                   'projection_dim': 32,
                   'train_model_numbers': None,
                   'data_parallel': True}

adversarial_training_kwargs = None  # Set to none for no adversarial training
# adversarial_training_kwargs = {'max_length': 24000,
#                                'eps': 10,
#                                'step_alpha': 1./10,
#                                'num_steps': 20,
#                                'smooth_attack': False,
#                                'sizes': None,
#                                'weights': None}

dverge_kwargs = None  # Set to none for no DVERGE training
# dverge_kwargs = {'batch_size': 64,
#                  'epochs': 80,
#                  'learning_rate': 0.001,
#                  'optimizer_name': 'Adam',
#                  'loss_fn': 'CrossEntropyLoss',
#                  'device': 'cuda',
#                  'epsilon': 1/40,
#                  'step_alpha': 1/400,
#                  'num_steps': 20,
#                  'data_parallel': True}


def main():
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    all_settings = {'data_kwargs': data_kwargs,
                    'training_kwargs': training_kwargs,
                    'ensemble_kwargs': ensemble_kwargs,
                    'load_models_kwargs': load_model_kwargs,
                    'data_dir': data_dir,
                    'adversarial_training_kwargs': adversarial_training_kwargs,
                    'dverge_kwargs': dverge_kwargs,
                    'max_training_minutes': max_training_minutes}
    with open(f'{save_dir}/settings.yaml', 'w') as file:
        yaml.dump(all_settings, file)

    data = np.load(f'{data_dir}/raw_data.npy', allow_pickle=True)
    raw_labels = np.load(f'{data_dir}/raw_labels.npy')
    permutation = np.load(f'{data_dir}/random_permutation.npy')
    train_data, train_labels, validate_data, validate_labels = create_data(data,
                                                                           raw_labels,
                                                                           permutation,
                                                                           **data_kwargs)
    ensemble = Ensemble(**ensemble_kwargs)
    if load_model_kwargs is not None:
        ensemble.load_models(**load_model_kwargs)
    if dverge_kwargs is not None:
        ensemble.train_dverge(train_data,
                              train_labels,
                              **dverge_kwargs,
                              max_training_minutes=max_training_minutes)
    else:
        ensemble.train(train_data,
                       train_labels,
                       validate_data,
                       validate_labels,
                       **training_kwargs,
                       adversarial_training_kwargs=adversarial_training_kwargs,
                       max_training_minutes=max_training_minutes)


if __name__ == "__main__":
    main()
