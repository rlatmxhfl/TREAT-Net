import os

import wandb

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import seaborn as sns

from .misc import DummyRun
from .dataloader import *
from .models import *


# reported are balanced accuracy

def unpack_batch(batch):
    if len(batch) == 4:
        return batch
    elif len(batch) == 3:
        pt_ids, video_tensors, labels = batch
        views = [None] * len(pt_ids)
        return pt_ids, video_tensors, views, labels
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")


def train_one_epoch(model, dataloader, criterion, optimizer, device, label_mapping, mode="study",
                    lr_scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue

        if mode == "study":
            pt_ids, video_tensors, views, labels = batch
        elif mode == "cine":
            pt_ids, video_tensors, labels = batch

        video_tensors = video_tensors.to(device, non_blocking=True)
        labels = [label_mapping[label] for label in labels]
        labels = torch.tensor(labels, dtype=torch.long).to(device=device, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(video_tensors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).tolist())
        all_labels.extend(labels.tolist())

    epoch_loss = running_loss / len(dataloader)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, balanced_acc * 100


def evaluate(model, dataloader, criterion, device, label_mapping, log_path=None, mode="study", run=DummyRun(),
             set_name='val'):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_logits = []
    log_data = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    attn_weights  = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating [{set_name}]"):
            if batch is None:
                continue

            pt_ids, video_tensors, views, labels = unpack_batch(batch)

            video_tensors = video_tensors.to(device)
            label_indices = [label_mapping[label] for label in labels]
            labels_tensor = torch.tensor(label_indices, dtype=torch.long, device=device)

            logits, attn_weights = model(video_tensors)
            # logits = model(video_tensors)
            loss = criterion(logits, labels_tensor)

            preds = torch.argmax(logits, dim=1).tolist()
            truths = labels_tensor.tolist()

            all_preds.extend(preds)
            all_labels.extend(truths)
            all_logits.append(logits.cpu())

            for i in range(len(preds)):
                entry = {
                    "patient_id": pt_ids[i],
                    "true_label": truths[i],
                    "predicted_label": preds[i],
                }
                if attn_weights is not None:
                    entry["attention_weights"] = attn_weights[i].tolist()
                    entry["num_videos"] = attn_weights.shape[1] if attn_weights.dim() > 1 else 1
                log_data.append(entry)

            running_loss += loss.item()

    all_logits_tensor = torch.cat(all_logits, dim=0)
    y_prob = torch.softmax(all_logits_tensor, dim=1).numpy()
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap="Blues",
                xticklabels=["MANAG", "PTCA", "SURG"],
                yticklabels=["MANAG", "PTCA", "SURG"])
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    filename = f"confusion_matrix_{model}.png" if log_path is None else log_path.replace(".csv",
                                                                                         "_confusion_matrix.png")
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    run.log({f'confusion_matrix/{set_name}': wandb.Image(fig)}, commit=False)
    plt.close()

    class_accuracies = {
        f"Class {cls}": accuracy_score(
            y_true[y_true == cls], y_pred[y_true == cls]
        ) for cls in np.unique(y_true)
    }

    metrics = {
        "ROC AUC": roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Class Accuracy": class_accuracies,
    }
    for metric, value in metrics.items():
        run.log({f'{metric}/{set_name}': value}, commit=False)

    for metric, value in metrics.items():
        if isinstance(value, dict):
            print("  Class Accuracy:")
            for cls, acc in value.items():
                print(f"    {cls}: {acc:.4f}")
        else:
            print(f"  {metric}: {value:.4f}")
    print("\n")

    if log_path:
        pd.DataFrame(log_data).to_csv(log_path, index=False)
        print(f"Per-patient log saved to: {log_path}")

        metrics_path = log_path.replace(".csv", "_metrics.txt")
        with open(metrics_path, "w") as f:
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    f.write("Class Accuracy:\n")
                    for cls, acc in value.items():
                        f.write(f"  {cls}: {acc:.4f}\n")
                else:
                    f.write(f"{metric}: {value:.4f}\n")
            print(f"Evaluation metrics saved to: {metrics_path}")

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss, metrics["Balanced Accuracy"] * 100


def get_lr_scheduler(optimizer, max_lr, steps_per_epoch, epochs, pct_start=0.0):
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=max_lr,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pct_start=0.3,
    #     anneal_strategy='cos',
    #     div_factor=25.0,
    #     final_div_factor=1e4
    # )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr, epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        # pct_start=0.1,
        pct_start=pct_start,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=100000,  # 2.0,
        final_div_factor=100000,  # 10000.0,
        three_phase=False,
        last_epoch=-1)
    return scheduler


def train_model(model, train_loader, val_loader, test_loader, label_mapping, device,
                save_path, checkpoint_name, num_epochs=30, lr=1e-5, wd=1e-3,
                criterion=None, mode="study", run=DummyRun(), epoch_start=0, optim='adam'):
    os.makedirs(save_path, exist_ok=True)

    # encoder_params = model.module.encoder.parameters() if isinstance(model,
    #                                                                  torch.nn.DataParallel) else model.encoder.parameters()

    if optim == 'adam':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, list(model.parameters())),
                                      lr=lr, weight_decay=wd, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, list(model.parameters())),
                                    lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_scheduler = get_lr_scheduler(optimizer, lr, len(train_loader), num_epochs - epoch_start)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = -1
    best_checkpoint_path = None

    for epoch in range(epoch_start, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_bal_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, label_mapping,
                                                    mode=mode, lr_scheduler=lr_scheduler)
        val_loss, val_bal_acc = evaluate(
            model, val_loader, criterion, device, label_mapping,
            log_path=os.path.join(save_path, f"{checkpoint_name}_val_epoch{epoch + 1}.csv"),
            run=run, set_name='val',
        )
        test_loss, test_bal_acc = evaluate(
            model, test_loader, criterion, device, label_mapping,
            log_path=os.path.join(save_path, f"{checkpoint_name}_test_epoch{epoch + 1}.csv"),
            run=run, set_name='test',
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Train] Loss: {train_loss:.4f}, Balanced Accuracy: {train_bal_acc:.2f}%")
        print(f"[Val] Loss:   {val_loss:.4f}, Balanced Accuracy: {val_bal_acc:.2f}%")
        print(f"[Test] Loss:   {val_loss:.4f}, Balanced Accuracy: {test_bal_acc:.2f}%")

        # Check for best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_checkpoint_path = os.path.join(save_path, f"{checkpoint_name}_epoch_{epoch + 1}.pth")

        run.log(
            {
                "loss/training": train_loss,
                "loss/validation": val_loss,
                "loss/test": test_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
            },
        )

    # Save final model
    final_path = os.path.join(save_path, f"{checkpoint_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

    # Print best epoch info
    print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    if best_checkpoint_path:
        print(f"Checkpoint path: {best_checkpoint_path}")

    # # Save loss curve
    # plt.figure()
    # plt.plot(train_losses, label="Train")
    # plt.plot(val_losses, label="Val")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss Curve")
    # plt.legend()
    # plt.grid()
    # plt.savefig(os.path.join(save_path, "loss_curve.png"))
    # plt.close()

    # # Final test evaluation
    # print("\nFinal evaluation on test set...")
    # test_loss, test_bal_acc = evaluate(
    #     model, test_loader, criterion, device, label_mapping,
    #     log_path=os.path.join(save_path, f"{checkpoint_name}_test_predictions.csv")
    # )
    # print(f"Test Loss: {test_loss:.4f}, Balanced Accuracy: {test_bal_acc:.2f}%")