import torch

def mape_score(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

def r2_score(y_pred, y_true):
    ss_res = torch.sum(torch.pow(y_true - y_pred, 2))
    ss_tot = torch.sum(torch.pow(y_true - torch.mean(y_true), 2))
    r2 = 1 - ss_res / ss_tot
    return r2

def mse_score(y_pred, y_true):
    return torch.mean(torch.pow(y_true - y_pred, 2))