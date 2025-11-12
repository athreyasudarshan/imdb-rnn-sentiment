# src/models.py
import torch
import torch.nn as nn

ACTS = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_size=64, num_layers=2,
                 dropout=0.4, rnn_type="rnn", bidirectional=False, activation="relu"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.activation = ACTS[activation.lower()]
        self.num_directions = 2 if bidirectional or rnn_type.lower()=="bilstm" else 1
        self.bidirectional = (bidirectional or rnn_type.lower()=="bilstm")

        if rnn_type.lower() in ["lstm", "bilstm"]:
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                               dropout=dropout, batch_first=True, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_size, num_layers=num_layers,
                              dropout=dropout, nonlinearity="tanh",
                              batch_first=True, bidirectional=self.bidirectional)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # NOTE: no Sigmoid here; we'll use BCEWithLogitsLoss
        )

    def forward(self, x):
        # x: (B, T) int ids with 0 as pad
        mask = (x != 0)  # True for real tokens
        emb = self.embed(x)  # (B, T, E)
        out, _ = self.rnn(emb)  # (B, T, H*D)

        # last non-pad index per sequence: lengths-1
        lengths = mask.sum(dim=1).clamp(min=1) - 1  # (B,)
        batch_idx = torch.arange(x.size(0), device=x.device)
        last_hidden = out[batch_idx, lengths, :]  # (B, H*D)

        logits = self.fc(last_hidden).squeeze(-1)  # (B,)
        return logits  # raw logits
