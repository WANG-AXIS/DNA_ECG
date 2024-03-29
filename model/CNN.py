import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    """This is the CNN architecture used in Han, et al. Deep
       learning models for electrocardiograms are susceptible to adversarial attack.
       Definition has been modified to accommodate prefiltering and decorrelation"""
    def __init__(self, num_classes=4, input_channels=1, f_filter=None):
        super(CNN, self).__init__()
        if f_filter is None:
            self.filter = None
        else:
            self.filter = nn.parameter.Parameter(torch.tensor(f_filter, dtype=torch.float32), requires_grad=False)

        self.conv1 = nn.Conv1d(input_channels, 320, kernel_size=24, stride=1, padding=11, bias=True)
        self.bn1 = nn.BatchNorm1d(320)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(320, 256, kernel_size=16, stride=1, padding=15, dilation=2, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=16, stride=1, padding=30, dilation=4, bias=True)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=16, stride=1, padding=30, dilation=4, bias=True)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.3)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=16, stride=1, padding=30, dilation=4, bias=True)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.3)
        self.conv6 = nn.Conv1d(256, 128, kernel_size=8, stride=1, padding=14, dilation=4, bias=True)
        self.bn6 = nn.BatchNorm1d(128)
        self.maxpool6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation = 2)
        self.dropout6 = nn.Dropout(0.3)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout7 = nn.Dropout(0.3)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn8 = nn.BatchNorm1d(128)
        self.dropout8 = nn.Dropout(0.3)
        self.conv9 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn9 = nn.BatchNorm1d(128)
        self.dropout9 = nn.Dropout(0.3)
        self.conv10 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn10 = nn.BatchNorm1d(128)
        self.dropout10 = nn.Dropout(0.3)
        self.conv11 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=28, dilation=8, bias=True)
        self.bn11 = nn.BatchNorm1d(128)
        self.maxpool11 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=2)
        self.dropout11 = nn.Dropout(0.3)
        self.conv12 = nn.Conv1d(128, 64, kernel_size=8, stride=1, padding=28, dilation=8, bias=True)
        self.bn12 = nn.BatchNorm1d(64)
        self.dropout12 = nn.Dropout(0.3)
        self.conv13 = nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=28, dilation=8, bias=True)
        self.bn13 = nn.BatchNorm1d(64)
        self.dropout13 = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)
        self.features = None
        self.layer_list = list(range(1, 14))  # List of layer indices for DVERGE

    def forward(self, x, return_features=False):
        if self.filter is not None:
            x = torch.fft.fft(x)
            x = torch.multiply(x, self.filter)
            x = torch.fft.ifft(x).real
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool6(x)
        x = self.dropout6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.dropout8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.dropout9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.dropout10(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.maxpool11(x)
        x = self.dropout11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.dropout12(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x = self.dropout13(x)
        self.features = torch.mean(x, dim=2)
        x = self.fc(self.features)
        if return_features:
            return x, self.features
        else:
            return x

    def get_features(self, x, layer_num):
        """This method extracts the features distilled from batch x at layer number layer_num,
        which is used for DVERGE training"""
        if self.filter is not None:
            x = torch.fft.fft(x)
            x = torch.multiply(x, self.filter)
            x = torch.fft.ifft(x).real
        for i in range(1, layer_num+1):
            if i != 1:
                x = F.relu(x)
                if i in [2, 7, 12]:
                    x = getattr(self, f'maxpool{i-1}')(x)
                x = getattr(self, f'dropout{i-1}')(x)
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'bn{i}')(x)
        return x
