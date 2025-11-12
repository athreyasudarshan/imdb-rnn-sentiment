import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
from src.utils import *

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_metrics(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin, average="macro")
    return acc, f1
