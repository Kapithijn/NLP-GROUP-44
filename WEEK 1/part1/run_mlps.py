"""Train the one-layer and two-layer MLPs on all three representations. Run from the part1 folder.

Usage: python run_mlps.py [number of folds, default 5]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.cwd().parent))
import part1

NUM_FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EPOCHS, BATCH_SIZE = 20, 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

data = load_dataset("google-research-datasets/paws", "labeled_final")["train"]
s1, s2, y = list(data["sentence1"]), list(data["sentence2"]), list(data["label"])
labels = torch.tensor(y, device=device)
tokenizer = part1.experiment.train_tokenizer(s1, 5000, 2)
splits = part1.experiment.cross_validation_split(list(zip(s1, s2, y)), k=5, seed=0)


def token_features(binary):
    """Sparse pair features: 5,000 columns for sentence 1, then 5,000 for sentence 2."""
    vocab = len(tokenizer.get_vocab())
    rows, cols = [], []
    for offset, sentences in enumerate((s1, s2)):
        for row, encoding in enumerate(tokenizer.encode_batch(sentences)):
            rows += [row] * len(encoding.ids)
            cols += [token + offset * vocab for token in encoding.ids]
    values = np.ones(len(rows), dtype=np.float32)
    X = sp.coo_matrix((values, (rows, cols)), shape=(len(s1), 2 * vocab)).tocsr()
    X.sum_duplicates()  # adds repeated tokens together, which gives the bag-of-words counts
    if binary:
        X.data[:] = 1
    return X


def batch(X, ids):
    rows = X[ids]
    return torch.from_numpy(rows if isinstance(rows, np.ndarray) else rows.toarray()).to(device)


def make_model(input_dim, hidden_layers):
    layers, size = [], input_dim
    for _ in range(hidden_layers):
        layers += [nn.Linear(size, 128), nn.ReLU(), nn.Dropout(0.1)]
        size = 128
    return nn.Sequential(*layers, nn.Linear(size, 2), nn.LogSoftmax(dim=-1)).to(device)


def train_fold(X, hidden_layers, train_ids, validation_ids, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(X.shape[1], hidden_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    best_loss, best_accuracy, best_epoch = float("inf"), None, None
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(train_ids)
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            criterion(model(batch(X, ids)), labels[ids]).backward()
            optimizer.step()
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(validation_ids), 512):
                ids = validation_ids[start:start + 512]
                logprobs = model(batch(X, ids))
                loss += nn.functional.nll_loss(logprobs, labels[ids], reduction="sum").item()
                correct += (logprobs.argmax(dim=1) == labels[ids]).sum().item()
        if loss < best_loss:
            best_loss, best_accuracy, best_epoch = loss, correct / len(validation_ids), epoch + 1
    return best_accuracy, best_epoch


features = {"indicator": token_features(True), "BoW": token_features(False),
            "BERT": np.load("bert_feats_bert-base-uncased.npy")}
systems = [("MLP", "BERT", 1), ("Two-layer MLP", "BERT", 2),
           ("MLP", "indicator", 1), ("Two-layer MLP", "indicator", 2),
           ("MLP", "BoW", 1), ("Two-layer MLP", "BoW", 2)]
scores = {}
for model_name, feature_name, hidden_layers in systems:
    name = f"{model_name} on {feature_name}"
    scores[name] = []
    for fold in range(NUM_FOLDS):
        start = time.time()
        train_ids = np.asarray(splits[fold]["training_ids"], dtype=int)
        validation_ids = np.asarray(splits[fold]["validation_ids"], dtype=int)
        X = features[feature_name]
        if feature_name == "BERT":  # scale inside each fold, using training rows only
            X = StandardScaler().fit(X[train_ids]).transform(X).astype(np.float32)
        accuracy, best_epoch = train_fold(X, hidden_layers, train_ids, validation_ids, fold)
        scores[name].append(accuracy)
        print(f"{name}, fold {fold + 1}: {accuracy:.4f} (best epoch {best_epoch}, {time.time() - start:.0f}s)", flush=True)
        with open("results/mlp_fold_scores.json", "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
