import os

import sys
sys.path.append('../../../model')
sys.path.append('/home/diane.kim/ACS/tabular/model/TabPFN/src')

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import random
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

from tabpfn.classifier import TabPFNClassifier
from tabpfn.config import ModelInterfaceConfig
from tabpfn.preprocessing import PreprocessorConfig

from src.evaluation import *

##################################### inspection/debug #####################################

def inspect_tensor(x, name="tensor", max_print=5):
    if isinstance(x, np.ndarray):
        arr = x
        dev = "numpy"
        dtype = arr.dtype
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
        dev = str(x.device)
        dtype = x.dtype
    else:
        print(f"[{name}] not a tensor/ndarray:", type(x))
        return
    
    print(f"[{name}] shape={arr.shape} dtype={dtype} device={dev} "
          f"nan={np.isnan(arr).any()} inf={np.isinf(arr).any()}")
    if arr.ndim > 0 and arr.size:
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr
        print(f"  min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g} mean={np.nanmean(arr):.6g} std={np.nanstd(arr):.6g}")
        print("  sample rows:")
        print(arr[:max_print] if arr.ndim <= 2 else arr[:max_print, :min(flat.shape[1], max_print)])

def inspect_labels(y, name="labels"):
    if torch.is_tensor(y):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)
    print(f"[{name}] shape={y_np.shape} uniq={np.unique(y_np, return_counts=True)} "
          f"dtype={y_np.dtype} nan={np.isnan(y_np).any() if y_np.dtype.kind in 'fc' else False}")

def check_no_nans(*tensors):
    for i, t in enumerate(tensors):
        if torch.is_tensor(t):
            bad = ~torch.isfinite(t)
            assert not bad.any().item(), f"Non-finite values in tensor #{i}"
        else:
            a = np.asarray(t)
            assert np.isfinite(a).all(), f"Non-finite values in array #{i}"

def check_gradients(model):
    total = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += float(g.abs().sum().cpu())
    print(f"[grads] total |grad| = {total:.6g}")
    return total

##################################### initiating & evaluating tabpfn #####################################

def init_tabpfn(seed=None, balance_probabilities=True):
    model = TabPFNClassifier(
        n_estimators=1,
        random_state=seed,
        balance_probabilities=balance_probabilities,
        inference_config=ModelInterfaceConfig(
          # OUTLIER_REMOVAL_STD=1000,
          REGRESSION_Y_PREPROCESS_TRANSFORMS=(None,),
          FINGERPRINT_FEATURE=False,
          PREPROCESS_TRANSFORMS=(PreprocessorConfig("none",),))
    )
    return model

def train_tabpfn(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    model_eval(y_true=y_test, y_pred=preds, y_proba=proba)
    return model

def get_tab_embeddings(model, X_train, X_val, X_test):
    emb_train = model.get_embeddings(X_train).mean(axis=0)
    emb_val = model.get_embeddings(X_val).mean(axis=0)
    emb_test = model.get_embeddings(X_test).mean(axis=0)

    print(emb_train.shape)

    emb_train = torch.tensor(emb_train, dtype=torch.float32)
    emb_val = torch.tensor(emb_val, dtype=torch.float32)
    emb_test = torch.tensor(emb_test, dtype=torch.float32)

    print("emb_train tensor shape: ", emb_train.shape)

    return emb_train, emb_val, emb_test

##################################### building MLP classifier #####################################

def build_classifier(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 192, bias=False),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(192, 1, bias=False),
    )

##################################### training tabpfn + MLP classifier #####################################

def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                num_epochs=100, optimizer_="Adam", seed=None, device="cuda"):

    def _train_model(X_train, y_train, X_val, y_val, X_test, y_test, seed=seed):
        # inspect_tensor(X_train, "X_train"); inspect_labels(y_train, "y_train")
        # inspect_tensor(X_val,   "X_val");   inspect_labels(y_val,   "y_val")
        # inspect_tensor(X_test,  "X_test");  inspect_labels(y_test,  "y_test")

        pfn = init_tabpfn(seed=seed, balance_probabilities=True)
        pfn = train_tabpfn(pfn, X_train, y_train, X_test, y_test)

        emb_train, emb_val, emb_test = get_tab_embeddings(pfn, X_train, X_val, X_test)
        input_dim = emb_train.shape[1]

        pos_freq = y_train.sum().item()
        neg_freq = len(y_train) - pos_freq
        pos_weight_val = neg_freq / (pos_freq + 1e-8)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, dtype=torch.float, device=device))

        # weights = (1. / torch.bincount(y_train.long()))[y_train.long()]
        # sampler = WeightedRandomSampler(weights=weights.cpu(), num_samples=len(weights), replacement=True)

        # inspect_tensor(emb_train, "emb_train")
        # inspect_tensor(emb_val, "emb_val")
        # inspect_tensor(emb_test, "emb_test")
        # check_no_nans(emb_train, emb_val, emb_test)

        model = build_classifier(input_dim).to(device)

        # print(model)
        # dummy = torch.randn(4, input_dim, device=device)
        # out = model(dummy)
        # inspect_tensor(out, "model(dummy) logits")

        # class_counts = torch.bincount(y_train)
        # class_weights = 1.0 / class_counts.float()
        # sample_weights = class_weights[y_train]

        # sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(TensorDataset(emb_train, y_train), batch_size=32,
                                #   sampler=sampler,
                                  sampler=None, shuffle=True
                                  )
        val_loader = DataLoader(TensorDataset(emb_val, y_val), batch_size=32, shuffle=False)
        test_loader = DataLoader(TensorDataset(emb_test, y_test), batch_size=32, shuffle=False)

        # criterion = nn.CrossEntropyLoss()

        if optimizer_ == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
        if optimizer_ == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, steps_per_epoch=len(train_loader),
                                                epochs=num_epochs, pct_start=0.3, anneal_strategy='cos')

        # x_batch, y_batch = next(iter(train_loader))
        # inspect_tensor(x_batch, "xb batch"); inspect_labels(y_batch, "yb batch")
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
                # if epoch == 0:
                #     print("[epoch0] pre-forward checkpoint")
                #     inspect_tensor(x_batch, "x_batch"); inspect_labels(y_batch, "y_batch")
                optimizer.zero_grad() #prevents gradients from accumulating from previous epochs
                logits = model(x_batch).squeeze(1)

                # inspect_tensor(logits, "logits") if (epoch == 0) else None

                loss = criterion(logits, y_batch)
                loss.backward()

                # gsum = check_gradients(model)

                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * len(y_batch) #mean loss multiplied by batch size to give the loss sum
            train_loss = total_loss/len(train_loader.dataset) #mean loss per sample across the entire dataset

            model.eval()
            with torch.no_grad():
                val_total_loss = 0
                correct = 0
                total_samples = 0
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
                    logits = model(x_batch).squeeze(1)
                    loss = criterion(logits, y_batch)
                    val_total_loss += loss.item() * len(y_batch)

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long()
                    correct += (preds == y_batch.long()).sum().item() #.item() when converting tensors to numbers/accumulating metrics
                    total_samples += y_batch.size(0)
                val_loss = val_total_loss/len(val_loader.dataset)
                val_acc = correct/total_samples
            
            print(f"Epoch {epoch:03d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} | val accuracy {val_acc:.3f}")
        
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"checkpoints/tabweights_BCE_final_epoch_seed{seed}.pt")
        print(f"Final epoch model weights saved for seed {seed}.")
        
        model.eval()
        with torch.no_grad():
            logits = model(emb_test.to(device)).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().numpy()
            y_test  = y_test.long().cpu().numpy()
        
        model_eval(y_true=y_test, y_pred=preds, y_proba=probs, positive_class=1, 
             specificities=(0.2, 0.4, 0.6), average="weighted", verbose=True, set_name="test")

        return model, emb_train
    

    model = _train_model(X_train, y_train, X_val, y_val, X_test, y_test, seed=seed)

    return model