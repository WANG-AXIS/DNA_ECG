"""This is the class definition for the ensemble model"""
import os
import csv
import time
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
sys.path.append('../')
from utils import filters as f_filters
from utils.decorrelation_func import decorrelation_fn
from utils.craft_attack import craft_attack


class Ensemble:
    def __init__(self,
                 architecture,
                 num_models,
                 input_channels=1,
                 num_classes=4,
                 save_dir=None,
                 filters=None):
        self.num_models = num_models
        self.save_dir = save_dir
        if filters is None:
            self.models = [architecture(num_classes, input_channels) for _ in range(num_models)]
        else:
            assert len(filters) == self.num_models, \
                "Number of filters must equal number of models"
            filters = [getattr(f_filters, i) if i is not None else None for i in filters]
            self.models = [architecture(num_classes, input_channels, f_filter=i) for i in filters]

    def train(self, training_data, training_labels, validate_data, validate_labels,
              batch_size, epochs, learning_rate, optimizer_name, loss_fn, device,
              decorrelate_weight, projection_dim, track_validation_R=True,
              train_model_numbers=None, data_parallel=False,
              adversarial_training_kwargs=None, max_training_minutes=None):
        if train_model_numbers is None:
            train_model_numbers = range(0, self.num_models)
        loss_fn = getattr(nn, loss_fn)()

        for model_id in train_model_numbers:
            print(f"Training model {model_id + 1}/{self.num_models}")

            model = self.models[model_id]
            device = torch.device(device)
            model.to(device)
            if data_parallel:
                device = torch.device('cuda')
                model = nn.DataParallel(model.to(device))

            optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)
            model_best_path = os.path.join(self.save_dir, f"model_{model_id}_best.pth")
            csv_file_path = os.path.join(self.save_dir, f"model_{model_id}_training_history.csv")

            with open(csv_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Epoch', 'Validation Accuracy', 'Validation Loss',
                                     'Validation R', 'Training Time (min)'])

                if decorrelate_weight > 0 and model_id > 0:
                    train_features = np.array([np.load(f'{self.save_dir}/train_features/model_{i}.npy')
                                              for i in range(0, model_id)]).transpose(1, 0, 2)
                    train_features = torch.tensor(train_features)
                else:
                    train_features = torch.zeros(len(training_labels))
                train_dataset = torch.utils.data.TensorDataset(training_data, training_labels, train_features)
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True)
                if track_validation_R and model_id > 0:
                    val_features = np.array([np.load(f'{self.save_dir}/val_features/model_{i}.npy')
                                            for i in range(0, model_id)]).transpose(1, 0, 2)
                    val_features = torch.tensor(val_features)
                else:
                    val_features = torch.zeros(len(validate_labels))
                validate_dataset = torch.utils.data.TensorDataset(validate_data, validate_labels, val_features)
                validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=False)

                start_time = time.time()
                best_epoch = 0
                best_validation_accuracy = 0
                best_validation_loss = np.inf
                stop_training = False

                for epoch in range(1, epochs+1):
                    model.train()
                    for batch_idx, (inputs, labels, features) in enumerate(train_loader):
                        inputs, labels, features = inputs.to(device), labels.to(device), features.to(device)

                        if adversarial_training_kwargs is not None:
                            inputs = craft_attack(inputs, None, labels, model, **adversarial_training_kwargs)
                            model.train()

                        optimizer.zero_grad()
                        outputs, current_features = model(inputs, return_features=True)
                        loss = loss_fn(outputs, labels)

                        if decorrelate_weight > 0 and model_id > 0:
                            features = torch.transpose(features, 1, 0)
                            decorrelation_loss = decorrelation_fn(current_features, features,
                                                                         projection_dim, return_R=False)
                            loss += decorrelate_weight * decorrelation_loss
                            print(f'Decorrelation Loss: {decorrelate_weight * decorrelation_loss}')

                        loss.backward()
                        optimizer.step()
                        print(f'Epoch:{epoch}/{epochs} Batch:{batch_idx+1}/{len(train_loader)} Train Loss:{loss:.4f}')
                        sys.stdout.flush()
                        if max_training_minutes is not None:
                            current_time = (time.time() - start_time) / 60.0
                            if current_time > max_training_minutes:
                                state_dict = model.module.state_dict() if data_parallel else model.state_dict()
                                path = os.path.join(self.save_dir, f"model_{model_id}_last.pth")
                                torch.save(state_dict, path)
                                print(f'Time limit reached. Saving model {model_id}.')
                                stop_training = True
                        if stop_training:
                            break
                    if stop_training:
                        break

                    validation_accuracy, validation_loss, validation_R = self.validate(model, validate_loader,
                                                                                       device, loss_fn,
                                                                                       decorrelate_weight,
                                                                                       projection_dim,
                                                                                       model_id, track_validation_R)
                    if validation_loss < best_validation_loss:
                        best_validation_accuracy = validation_accuracy
                        best_validation_loss = validation_loss
                        best_epoch = epoch
                        state_dict = model.module.state_dict() if data_parallel else model.state_dict()
                        torch.save(state_dict, model_best_path)

                    current_time = (time.time() - start_time) / 60.0  # Convert to minutes
                    csv_writer.writerow([epoch, validation_accuracy, validation_loss, validation_R, current_time])
                    sys.stdout.flush()

            print(f"Model {model_id + 1} trained. Best validation accuracy: {best_validation_accuracy:.4f}"
                  f" at Epoch {best_epoch}. Total training time: {current_time:.4f} minutes.")

            for dataset, data_name in zip([training_data, validate_data], ['train', 'val']):
                save_path = f'{self.save_dir}/{data_name}_features'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                print(f'Extracting {data_name} features...')
                self.save_features(model, dataset, f'{save_path}/model_{model_id}', batch_size, device)

            model.to(torch.device('cpu'))
            torch.cuda.empty_cache()

    @staticmethod
    def validate(model, validate_loader, device, loss_fn, decorrelate_weight=0,
                 projection_dim=None, model_id=0, track_R=False):
        model.eval()
        correct = 0
        total = 0
        loss = 0
        R_squared = 0
        with torch.no_grad():
            for inputs, labels, features in validate_loader:
                inputs, labels, features = inputs.to(device), labels.to(device), features.to(device)
                outputs, current_features = model(inputs, return_features=True)
                loss += loss_fn(outputs, labels)
                if model_id > 0 and track_R:
                    features = torch.transpose(features, 1, 0)
                    decorrelation_loss, batch_R = decorrelation_fn(current_features, features,
                                                                   projection_dim, return_R=True)
                    loss += decorrelate_weight * decorrelation_loss
                    R_squared += batch_R.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct/total, loss.item()/total, R_squared/len(validate_loader)

    @staticmethod
    def save_features(model, data, save_path, batch_size, device):
        data = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset=data,
                                             batch_size=batch_size,
                                             shuffle=False)
        features = np.array([])
        model.eval()
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(device)
                _, batch_features = model(inputs, return_features=True)
                batch_features = batch_features.detach().cpu().numpy()
                features = np.vstack([features, batch_features]) if features.size else batch_features
        np.save(f'{save_path}.npy', features)
        return None

    def load_models(self, load_dir, model_nums=None):
        if model_nums is None:
            model_nums = range(self.num_models)
        for model_id in model_nums:
            model_path = os.path.join(load_dir, f"model_{model_id}_best.pth")
            try:
                self.models[model_id].load_state_dict(torch.load(model_path))
            except RuntimeError:
                model = nn.DataParallel(self.models[model_id])
                model.load_state_dict(torch.load(model_path))
                self.models[model_id] = model.module
            self.models[model_id].eval()

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def infer(self, x, batch_size, device):
        """ This method infers the classifications of inputs x as an ensemble
        Returns
        predictions: Predicted labels of each sample (int)
        I: Unnormalized uncertainty of each sample, as defined in Mobiny, et. al;
        DropConnect is Effective in Modelling Uncertainty of Bayesian Deep Networks
        (float) """
        eta = 1e-30  # constant for log stability
        device = torch.device(device)
        for model in self.models:
            model.eval()
            model.to(device)
        x = torch.tensor(x, dtype=torch.float32)
        predictions = np.array([], dtype=np.int32)
        I = np.array([], dtype=np.float32)
        with torch.no_grad():
            for x_batch in self.chunks(x, batch_size):
                x_batch = x_batch.to(device)
                ensemble_out = [F.softmax(model(x_batch), dim=1).detach().cpu().numpy()
                                for model in self.models]
                ensemble_out = np.array(ensemble_out).transpose(1, 0, 2)  # now ordered batch, model, class
                y_batch = np.mean(ensemble_out, axis=1)
                predictions_batch = np.argmax(y_batch, axis=1)
                I_batch = -np.sum(y_batch * np.log(y_batch+eta), axis=1) \
                    + np.sum(ensemble_out * np.log(ensemble_out+eta), axis=(1, 2))/self.num_models
                predictions = np.vstack([predictions, predictions_batch]) if predictions.size else predictions_batch
                I = np.vstack([I, I_batch]) if I.size else I_batch
        return predictions, I

    def train_dverge(self, training_data, training_labels, batch_size, epochs,
                     learning_rate, optimizer_name, loss_fn, device,
                     epsilon, step_alpha, num_steps,
                     data_parallel=False, max_training_minutes=None, train_model_numbers=None):
        if train_model_numbers is None:
            train_model_numbers = range(0, self.num_models)

        models = [self.models[i].to(device) for i in train_model_numbers]
        layer_list = models[0].layer_list
        if data_parallel:
            models = [nn.DataParallel(i) for i in models]

        optimizers = [getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate) for model in models]
        loss_fn = getattr(nn, loss_fn)()
        train_dataset = torch.utils.data.TensorDataset(training_data, training_labels)

        start_time = time.time()
        stop_training = False
        if max_training_minutes is not None:
            max_training_minutes = max_training_minutes * len(models)
        for epoch in range(1, epochs+1):
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            source_loader = iter(torch.utils.data.DataLoader(dataset=train_dataset,
                                                             batch_size=batch_size,
                                                             shuffle=True))

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                source_inputs, source_labels = next(source_loader)
                source_inputs, source_labels = source_inputs.to(device), source_labels.to(device)

                feature_layer = random.choice(layer_list)
                distilled_features = []
                for model in models:
                    model.eval()
                    if data_parallel:
                        forward_func = lambda x: model.module.get_features(x, feature_layer)
                    else:
                        forward_func = lambda x: model.get_features(x, feature_layer)
                    with torch.no_grad():
                        target_features = forward_func(inputs).detach()
                    distilled_feature = craft_attack(source_inputs, None, target_features, forward_func,
                                                     eps=epsilon, step_alpha=step_alpha, num_steps=num_steps,
                                                     loss_fn=nn.MSELoss(), minimize=True)
                    distilled_features.append(distilled_feature)
                for model_idx, model in enumerate(models):
                    model.train()
                    optimizers[model_idx].zero_grad()
                    total_loss = 0
                    for feature_idx, feature in enumerate(distilled_features):
                        if model_idx != feature_idx:
                            outputs = model(feature)
                            loss = loss_fn(outputs, source_labels)
                            loss.backward()
                            total_loss += loss.item()
                    optimizers[model_idx].step()
                    print(f'Epoch:{epoch}/{epochs} Batch:{batch_idx+1}/{len(train_loader)}'
                          f' Model:{model_idx} Train Loss:{total_loss:.4f}')
                    sys.stdout.flush()
                    torch.cuda.empty_cache()

                if max_training_minutes is not None:
                    current_time = (time.time() - start_time) / 60.0
                    if current_time > max_training_minutes:
                        print(f'Time limit reached.')
                        stop_training = True
                if stop_training:
                    break
            if stop_training:
                break
        print(f'Saving Models')
        for model_id, model in enumerate(models):
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            path = os.path.join(self.save_dir, f"model_{model_id}_best.pth")
            torch.save(state_dict, path)
