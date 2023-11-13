import numpy as np
import torch
import logging


def rand_project(X, k, D):
    R = torch.randn(D, k).to(X.device)/np.sqrt(D)
    return torch.matmul(X, R)


def get_Y_hat(X, Y, k):
    batch_size, D = X.shape
    if k is not None:
        X = rand_project(X, k, D)
    X = torch.cat([X, torch.ones([batch_size, 1]).to(X.device)], dim=1)
    rows, columns = X.shape
    if rows < columns:
        logging.warning(f'Feature Matrix is underdetermined: {rows} rows and {columns} columns.'
                        f'Returning 0 for decorrelation loss and R^2.')
        return None
    else:
        Yhat = torch.matmul(torch.matmul(X, torch.linalg.pinv(X)), Y)
        return Yhat


def decorrelation_fn(X, Y_all, k=None, return_R=True):
    """Calculates the decorrelation across multiple sets of features from other models
    X (Tensor): Extracted features from the current model that is training. Shape (batch_size, feature_dim)
    Y_all (Tensor): Series of extracted features from previously trained models. Shape (model_num, batch_size, feature_dim)
    k (int): Dimension in which to randomly project features
    return_R (bool): Flag that controls whether to return average R^2 value. If false, only the component needed for the
     loss is returned."""
    eta = 0.00001  # constant for stability
    loss_R = 0
    R = 0
    for Y in Y_all:
        if np.random.uniform() > 0.5:
            X, Y = Y, X
        X = torch.flatten(X, start_dim=1)
        Y = torch.flatten(Y, start_dim=1)
        Yhat = get_Y_hat(X, Y, k)
        if Yhat is None:
            R += torch.tensor(0, dtype=torch.float32, device=X.device)
            loss_R += torch.tensor(0, dtype=torch.float32, device=X.device)
        else:
            SSres = torch.sum(torch.square(Y - Yhat))
            SStot = torch.sum(torch.square(Y - torch.mean(Y, dim=0).unsqueeze(0)))
            R += (1 - SSres/(SStot+eta))
            loss_R += (torch.log(SStot+eta) - torch.log(SSres+eta))
  #      print('SSres:{} SStot:{} R:{}'.format(SSres, SStot, R))
    loss_R /= len(Y_all)
    R /= len(Y_all)
    if return_R:
        return loss_R, R
    else:
        return loss_R
