import hashlib
import sys
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=12, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

def accuracy(network, loader, weights, device, algorithm):
    loss_func = F.binary_cross_entropy_with_logits
    p_all = np.array([], dtype=float)
    y_all = np.array([], dtype=float)
    network.eval()
    for i, batch in enumerate(loader):
        if network.hparams["dm_idx"]:
            x = batch[0].to(device)
            y = batch[1].to(device)
            d = batch[2].to(device)
            p = network.predict(x, d)

        else:
            x = batch[0].to(device)
            y = batch[1].to(device)
            p = network.predict(x)

        if torch.isnan(p).any():
            print("张量 p 中存在 NaN 元素:")
            print(p)
        p = p.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        p_all = np.concatenate((p_all, p))
        y_all = np.concatenate((y_all, y))

    y_true = torch.from_numpy(y_all).float()  # .reshape(-1,1)
    y_pred = torch.from_numpy(p_all).float()
    auc = roc_auc_score(y_all, p_all)
    log_losses = loss_func(y_pred.reshape(-1,1), y_true.reshape(-1,1),reduction='mean')
    # average_log_loss = float(np.mean(log_losses))
    network.train()
    return auc, log_losses.item()


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
