"""This function is from Han, et al. (https://github.com/XintianHan/ADV_ECG)
and is used to create training and validation set from the 2017 Physionet Data"""
import torch
import numpy as np
from torch.utils.data import Dataset
MAX_SENTENCE_LENGTH = 18000


class ECGDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list, target_list, max_length):
        """
        @param data_list: list of newsgroup tokens
        @param target_list: list of newsgroup targets
        @param max_length: maximum length (int) for output data

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """

        datum = self.data_list[key]
        if len(datum.shape) == 1:
            datum = np.expand_dims(datum, 0)
        if len(datum[0]) >= self.max_length:
            datum = datum[:, :self.max_length]

        label = self.target_list[key]
        return [datum, len(datum[0]), label]


def ecg_collate_func(batch, max_length):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []

    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        remainder = max_length - datum[1]
        padded_vec = np.pad(np.array(datum[0]), ((0, 0), (int(remainder/2), remainder-int(remainder/2))),
                            'constant', constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)).type(torch.FloatTensor),
            torch.LongTensor(length_list),
            torch.LongTensor(label_list)]
