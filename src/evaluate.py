import os, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from src.models import BaseRNN
from utils import set_seed

def evaluate(seq_len=50, arch="lstm", act="relu"):
    set_seed(42)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = np.load(os.path.join(base, "data", f"imdb_{seq_len}.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    vocab_size = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseRNN(vocab_size, rnn_type=arch, bidirectional=(arch=="bilstm"),
                    activation=act).to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.long),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=32)

    preds, y_true = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = model(xb).squeeze().cpu().numpy()
            preds.extend(p)
            y_true.extend(yb.numpy())

    acc = accuracy_score(y_true, np.array(preds) >= 0.5)
    f1 = f1_score(y_true, np.array(preds) >= 0.5, average="macro")
    print(f"Test Accuracy={acc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    evaluate()
