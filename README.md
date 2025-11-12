# IMDb Sentiment Analysis with RNNs

This project trains and evaluates **RNN-based sentiment classifiers** (RNN, LSTM, BiLSTM) on the IMDb movie review dataset using **PyTorch**.

---

## Setup Instructions

### Requirements
- Python **3.10+**
- Recommended: virtual environment (`venv` or `conda`)
- Install dependencies:

```bash
pip install -r requirements.txt

## Project Structure
imdb-rnn-sentiment/
│
├── data/
│ ├── IMDB Dataset.csv
│ ├── imdb_25.npz
│ ├── imdb_50.npz
│ ├── imdb_100.npz
│ └── vocab.txt
│
├── results/
│ ├── metrics.csv
│ └── plots/
│
├── src/
│ ├── preprocess.py
│ ├── models.py
│ ├── train.py
│ ├── evaluate.py
│ └── plot_results.py
│
└── README.md


## How to Run

### 1 Preprocess the Dataset
```bash
python -m src.preprocess

## 2️ Train the Model
python -m src.train --arch lstm --act relu --opt adam --seq_len 50 --clip no --epochs 5

Command-Line    Options
Flag	        Description	Example
--arch	        Model architecture (rnn, lstm, bilstm)	        --arch lstm
--act	        Activation function (relu, tanh, sigmoid)	    --act relu
--opt	        Optimizer (adam, sgd, rmsprop)	                --opt adam
--seq_len	    Sequence length (25, 50, 100)	                --seq_len 50
--clip	        Gradient clipping (yes or no)	                --clip yes
--epochs	    Number of training epochs	                    --epochs 5

## ⏱️ Expected Runtime and Outputs

| Sequence Length | Avg Time/Epoch (CPU) | Accuracy (Approx.) |
|------------------|----------------------|--------------------|
| 25                |                 ~8 s |           0.72 |
| 50                |                 ~12 s |            0.77 |
| 100               |                   ~19 s |             0.82 |

### Output Files
- `data/imdb_*.npz` → preprocessed train/test datasets  
- `results/metrics.csv` → summary of all experiment runs (Accuracy, F1, Time per epoch)


##Reproducibility

Random seeds set (42) for NumPy and PyTorch
TensorFlow 2.20.0 (for preprocessing)
PyTorch 2.2.0 (for training/evaluation)
Deterministic runs for consistent results