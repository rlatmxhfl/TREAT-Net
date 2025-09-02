import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
sys.path.append('/home/diane.kim/nature/model')
sys.path.append('/home/diane.kim/nature/model/baseline/tabpfn_mlp')

from utils.data_preprocessor import *
from utils.fix_seed import *
from tabpfn_mlp_bce import train_model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0  # fallback default if not passed

    print(f"Running with seed: {seed}")
    fix_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test, train_ids, val_ids, test_ids, preprocessor = load_data(binary=True, seed=seed, ref_df="/home/diane.kim/nature/data/final/MASTER_2262_wTTE.csv")

    model = train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                num_epochs=100, optimizer_="SGD", seed=seed, device="cuda")

    # import torch
    # from torch import nn
    # import numpy as np
    # from sklearn.metrics import balanced_accuracy_score

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # ---- pick a tiny subset
    # idx = torch.randperm(len(X_train))[:32]
    # X_small = X_train[idx].clone()          # shape [32, D]
    # y_small = y_train[idx].clone()          # tensor of 0/1

    # # ---- sanity check labels
    # print("labels uniq:", torch.unique(y_small, return_counts=True))

    # # ---- normalize (critical for stability)
    # mu  = X_small.mean(0, keepdim=True)
    # std = X_small.std(0, keepdim=True).clamp_min(1e-6)
    # X_small = (X_small - mu) / std

    # # ---- simple MLP with biases
    # def build_classifier(input_dim):
    #     return nn.Sequential(
    #         nn.Linear(input_dim, 192, bias=True),
    #         nn.ReLU(),
    #         nn.Linear(192, 1, bias=True),
    #     )

    # m   = build_classifier(X_small.shape[1]).to(device)
    # opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=0.0)
    # crit = nn.BCEWithLogitsLoss()

    # X_small = X_small.to(device)
    # y_small = y_small.float().to(device)

    # m.train()
    # for it in range(2000):
    #     opt.zero_grad()
    #     logits = m(X_small).squeeze(1)             # [32]
    #     loss   = crit(logits, y_small)             # y_small float in {0,1}
    #     loss.backward()
    #     opt.step()
    #     if (it+1) % 200 == 0:
    #         print(f"iter {it+1:4d} | loss {loss.item():.4f}")

    # m.eval()
    # with torch.no_grad():
    #     probs = torch.sigmoid(m(X_small).squeeze(1))
    #     preds = (probs >= 0.5).long().cpu().numpy()
    #     y_np  = y_small.long().cpu().numpy()
    #     bacc  = balanced_accuracy_score(y_np, preds)
    #     print("Tiny-set bAcc:", bacc)

