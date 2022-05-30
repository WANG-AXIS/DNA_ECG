import numpy as np
import torch
import utils.global_variables as gv
device = gv.device


def rand_project(X, k, D):
    R = torch.randn(D, k).to(device)/np.sqrt(D)
    return torch.matmul(X, R)


def get_Y_hat(X, Y, k):
    batch_size, D = X.shape
    if k is not None:
        X = rand_project(X, k, D)
    X = torch.cat([X, torch.ones([batch_size, 1]).to(device)], dim=1)
    Yhat = torch.matmul(torch.matmul(X, torch.linalg.pinv(X)), Y)
    return Yhat


def get_R(X, Y, k=None):
    if np.random.uniform()>0.5:
        X,Y = Y,X
    X = torch.flatten(X, start_dim=1)
    Y = torch.flatten(Y, start_dim=1)
    Yhat = get_Y_hat(X, Y, k)
    SSres = torch.sum(torch.square(Y - Yhat))
    SStot = torch.sum(torch.square(Y - torch.mean(Y, dim=0).unsqueeze(0)))
    eta = 0.00001 #constant for stability
    R = 1 - SSres/(SStot+eta)
   # print('SSres:{} SStot:{} R:{}'.format(SSres, SStot, R))
    return torch.log(SStot+eta) - torch.log(SSres+eta), R