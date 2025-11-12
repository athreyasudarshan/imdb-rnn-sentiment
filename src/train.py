# src/train.py
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.models import BaseRNN


def load_data(seq_len: int):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", f"imdb_{seq_len}.npz")
    data = np.load(path)
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def get_vocab_size() -> int:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base, "data", "vocab.txt"), "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def ensure_metrics_header(metrics_path: str):
    if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("Model,Activation,Optimizer,Seq Length,Grad Clipping,Accuracy,F1,Epoch Time (s)\n")


def train_model(args):
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    X_train, y_train, X_test, y_test = load_data(args.seq_len)
    vocab_size = get_vocab_size()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = BaseRNN(
        vocab_size=vocab_size,
        rnn_type=args.arch,
        bidirectional=(args.arch == "bilstm"),
        activation=args.act
    ).to(device)

    # Loss/optim (use logits for stability)
    criterion = nn.BCEWithLogitsLoss()
    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.long),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.long),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=32
    )

    # Results file
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    ensure_metrics_header(metrics_path)

    print(f"Training {args.arch.upper()} | act={args.act} | opt={args.opt} | len={args.seq_len} | clip={args.clip}")
    epoch_times = []

    # ---- Train
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)                 # raw logits
            loss = criterion(logits, yb)       # no sigmoid here
            loss.backward()
            if args.clip == "yes":
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss/len(train_loader):.4f} - Time: {epoch_time:.2f}s")

    # ---- Evaluate
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()  # convert logits -> probabilities
            preds_all.extend(probs)
            y_all.extend(yb.numpy())

    preds_all = np.array(preds_all)
    acc = accuracy_score(y_all, (preds_all >= 0.5))
    f1 = f1_score(y_all, (preds_all >= 0.5), average="macro")
    time_per_epoch = float(np.mean(epoch_times))

    # ---- Save metrics
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(f"{args.arch},{args.act},{args.opt},{args.seq_len},{args.clip},{acc:.4f},{f1:.4f},{time_per_epoch:.2f}\n")

    print(f"Done: Accuracy={acc:.4f}, F1={f1:.4f}, Time/epoch={time_per_epoch:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["rnn", "lstm", "bilstm"], required=True)
    parser.add_argument("--act", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--opt", choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--seq_len", type=int, choices=[25, 50, 100], default=50)
    parser.add_argument("--clip", choices=["yes", "no"], default="no")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_model(args)
