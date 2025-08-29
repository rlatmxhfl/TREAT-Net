import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
sys.path.append('/mnt/rcl-server/workspace/diane/nature')

import argparse
import warnings
from functools import partial

import torch
import wandb
import random
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from scipy.interpolate import interp1d

from src import *
from src.loss_functions import *
from src.loss_functions.anl_loss import SymmetricCrossEntropy

# from src import LABEL_MAPPING_DICT, NUM_CLASSES

VIEWS = ['AP2', 'AP4', 'PLAX']
# TARGET_NAMES = [f"CAD {i}" for i in range(1, 6)]
from src.dataloader import EMB_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("-s", "--suffix", type=str, default="")
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="embedding")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--target", type=str, default='cad', choices=['cad', 'tp'])
    parser.add_argument("--loss_fn", type=str, default='anl_ce', choices=['ce', 'anl_ce'])
    parser.add_argument("--dropped_class", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--save_embeddings", action='store_true')
    parser.add_argument("--use_view_tokens", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--total_epochs", type=int, default=50)
    parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-3)
    parser.add_argument("--subsample_frac", type=float, default=None)
    parser.add_argument("--subsample_n", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lpft", action="store_true")
    parser.add_argument("-nbb", "--non_balanced_batch", action="store_true")
    parser.add_argument("-ufe", "--unfreeze_encoder", action="store_true")
    parser.add_argument("-nw", "--no_wandb", action="store_true")
    parser.add_argument("-db", "--debug", action="store_true")
    parser.add_argument("--wdb_group", type=str, default=None)
    parser.add_argument("--pretrain_view", action="store_true", help="Pretrain transformer on view classification")
    parser.add_argument("--pretrain_epochs", type=int, default=10)
    parser.add_argument("--tab_weight", type=str, default=None)
    parser.add_argument("--mode", type=str, default='video+tab',
                        choices=['video', 'video+tab', 'late_fusion'])

    args = parser.parse_args()
    return args


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    lr_scheduler=None, run=DummyRun(), is_anl=False):
    model.train()
    running_loss = 0.0
    total = 0

    metrics = {
        'preds_cad': [], 'targets_cad': [], 'probs_cad': [],
        'preds_tp': [], 'targets_tp': [], 'probs_tp': [],
    }

    for batch in tqdm(dataloader, desc="Training", mininterval=33):
        videos, tabs, views, labels = batch[:4]
        videos = videos.to(device)
        tabs = tabs.to(device)
        views = views.to(device)

        optimizer.zero_grad()

        if len(labels) == 1:
            labels = labels[0].to(device)
            outputs = model(videos, tabs, views, training=True)
            if not is_anl:
                loss = criterion(outputs.squeeze(), labels.squeeze().float())
            else:
                loss = criterion(outputs, labels, model)

            preds = outputs.argmax(dim=1) if outputs.squeeze().ndim == 2 else (outputs > 0).long()
            metrics['preds_tp'].append(preds.detach().cpu())
            metrics['targets_tp'].append(labels.cpu())
            if outputs.squeeze().ndim <= 1:
                metrics['probs_tp'].append(torch.sigmoid(outputs.detach()).cpu())
            else:
                metrics['probs_tp'].append(torch.softmax(outputs.detach(), dim=1).cpu())
        else:
            labels_tp, labels_cad = [l.to(device) for l in labels]
            logits_tp, logits_cad = model(videos, tabs, views, training=True)

            loss_tp = criterion['tp'](logits_tp.squeeze(), labels_tp.squeeze().float())
            # loss_cad = criterion['cad'](logits_cad.squeeze(), labels_cad.squeeze().float())
            loss_cad = criterion['cad'](logits_cad, labels_cad)
            loss = loss_tp + loss_cad * 3

            preds_tp = (logits_tp > 0).long()
            probs_tp = torch.sigmoid(logits_tp)

            # preds_cad = (logits_cad > 0).long()
            # probs_cad = torch.sigmoid(logits_cad)
            preds_cad = logits_cad.argmax(dim=1)
            probs_cad = torch.softmax(logits_cad, dim=1)


            metrics['preds_tp'].append(preds_tp.cpu())
            metrics['targets_tp'].append(labels_tp.cpu())
            metrics['probs_tp'].append(probs_tp.detach().cpu())

            metrics['preds_cad'].append(preds_cad.cpu())
            metrics['targets_cad'].append(labels_cad.cpu())
            metrics['probs_cad'].append(probs_cad.detach().cpu())

        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss.item() * videos.size(0)
        total += videos.size(0)

    run_data = {
        "loss/train": running_loss / total,
    }

    for task in ['tp', 'cad']:  # reversed order
        if metrics[f'preds_{task}']:
            preds = torch.cat(metrics[f'preds_{task}']).numpy()
            targets = torch.cat(metrics[f'targets_{task}']).numpy()
            probs = torch.cat(metrics[f'probs_{task}']).numpy()

            run_data[f"acc/train_{task}"] = 100.0 * accuracy_score(targets, preds)
            run_data[f"acc_balanced/train_{task}"] = 100.0 * balanced_accuracy_score(targets, preds)
            run_data[f"worst_acc/train_{task}"] = 100.0 * compute_worst_class_accuracy(targets, preds)
            run_data[f"f1/train_{task}"] = 100.0 * f1_score(targets, preds, average='macro')

            run_data[f"auc/train_{task}"] = safe_auc_score(
                targets, probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs,
                average='macro', multi_class='ovr'
            )

    run.log(run_data, commit=False)
    return run_data["loss/train"], run_data.get("acc_balanced/train_tp")


def get_criterion(multitask=False, loss_fn='ce', is_eval=False, num_cad_classes=5):
    if multitask:
        criterion = {
            'tp': nn.BCEWithLogitsLoss(),
            # 'cad': nn.BCEWithLogitsLoss(),
            'cad': nn.CrossEntropyLoss(),
        }
    else:
        if loss_fn != 'ce':
            warnings.warn(f"{loss_fn} loss is not supported for single task.")
        criterion = nn.BCEWithLogitsLoss()
    if is_eval:
        return criterion

    configs = {'is_anl': False}
    if (loss_fn == 'anl_ce') and multitask:
        configs = {
            "name": "anl_ce",
            "alpha": 20.0,  # 20.0, 10.0
            "beta": 1.0,
            "delta": 5e-6,  # 5e-6, 5e-7
            "min_prob": 1e-7,
            "is_anl": True
        }
        criterion['cad'] = anl_ce(num_cad_classes, configs)
        warnings.warn(f"{loss_fn} loss is only supported for CAD prediction.")
    elif loss_fn != 'ce':
        raise ValueError('Loss function not implemented')

    return criterion, configs

def plot_sensitivity_over_epochs(stats_dict, set_name, save_path, run_name=''):
    """
    Plots sensitivity over epochs for different specificity targets.

    Args:
        stats_dict (dict): The main dictionary containing sensitivity stats.
        set_name (str): The dataset to plot ('val' or 'test').
        save_path (str): Directory where the plot image will be saved.
        run_name (str): An optional name for the run to include in the title.
    """
    # Select the data for the specified set (e.g., 'val' or 'test')
    if set_name not in stats_dict or not stats_dict[set_name]:
        print(f"No data available to plot for the '{set_name}' set.")
        return

    data_to_plot = stats_dict[set_name]
    
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the data for each specificity target
    for target, stats in data_to_plot.items():
        if not stats:
            continue  # Skip if there's no data for this target

        # Unzip the list of tuples into separate lists for epochs, sensitivities, etc.
        # stats is a list of tuples like: [(epoch, sensitivity, specificity, threshold), ...]
        epochs, sensitivities, _, _ = zip(*stats)

        # Plot sensitivities vs. epochs
        ax.plot(epochs, sensitivities, marker='o', linestyle='-', label=f'Spec. Target {int(target*100)}%')

    # Formatting the plot
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"Sensitivity Over Epochs for {set_name.upper()} Set\n{run_name}".strip())
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1.05) # Sensitivity is between 0 and 1
    
    # Ensure x-axis ticks are integers
    all_epochs = sorted({epoch for stats in data_to_plot.values() for epoch, _, _, _ in stats})
    if all_epochs:
        ax.set_xticks(np.unique(all_epochs))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


    # Save the figure to a file
    plot_filename = os.path.join(save_path, f"sensitivity_plot_{set_name}_{run_name}.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Sensitivity plot saved to: {plot_filename}")

def train_model(model, train_loader, val_loader, test_loader, label_mapping, device,
                save_path, checkpoint_name, num_epochs=30, lr=1e-5, wd=1e-3,
                loss_fn=None, mode="study", run=DummyRun(), epoch_start=0, optim='adam',
                num_classes=3, run_name='', freeze=False, multitask=False):
    os.makedirs(save_path, exist_ok=True)

    # Downstream task
    model.freeze_pretrain_head()
    model.unfreeze_task_head()
    if freeze:
        model.freeze_encoder()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        # optimizer = torch.optim.Adam(model.task_classifier.parameters(), lr=1e-4, weight_decay=0.)
        # optimizer = torch.optim.SGD(model.task_classifier.parameters(), lr=1e-2, nesterov=True, momentum=0.9)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1  # 10% warm-up
        )
    else:
        if optim == 'adam':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, weight_decay=wd,
                                          # amsgrad=True
                                          )
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        lr_scheduler = get_lr_scheduler(optimizer, lr, len(train_loader), num_epochs - epoch_start, pct_start=0.1)

    show_trainable(model)

    criterion, configs = get_criterion(multitask=multitask, loss_fn=loss_fn)
    criterion_eval = get_criterion(multitask=multitask, loss_fn=loss_fn, is_eval=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = -1
    best_checkpoint_path = None

    sensitivity_stats = {
    'val': {0.2: [], 0.4: [], 0.6: []},
    'test': {0.2: [], 0.4: [], 0.6: []}
    }

    for epoch in range(epoch_start, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_bal_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            lr_scheduler=lr_scheduler, run=run,
            is_anl=configs['is_anl'],
        )
        val_loss, val_bal_acc, val_sens_data = evaluate(
            model, val_loader, criterion_eval, device, label_mapping, epoch=epoch,
            # log_path=os.path.join(save_path, f"{checkpoint_name}_val_epoch{epoch + 1}.csv"),
            run=run, set_name='val',
        )
        test_loss, test_bal_acc, test_sens_data = evaluate(
            model, test_loader, criterion_eval, device, label_mapping, epoch=epoch,
            run=run, set_name='test',
        )

        for t in [0.2, 0.4, 0.6]:
            if t in val_sens_data:
                sensitivity_stats['val'][t].append((epoch+1, *val_sens_data[t]))
            if t in test_sens_data:
                sensitivity_stats['test'][t].append((epoch+1, *test_sens_data[t]))    

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Train] Loss: {train_loss:.4f}, Balanced Accuracy: {train_bal_acc:.2f}%")
        print(f"[Val]   Loss: {val_loss:.4f}, Balanced Accuracy: {val_bal_acc:.2f}%")
        print(f"[Test]  Loss: {test_loss:.4f}, Balanced Accuracy: {test_bal_acc:.2f}%")
        if len(run_name):
            print(run_name)

        # Log to wandb or DummyRun
        run.log({
            "loss/train": train_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1,
        }, commit=True)

        # Save best model by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1


    # Save final model
    final_path = os.path.join(save_path, f"{checkpoint_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    # print(f"\nFinal model saved: {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    for set_name in ['val', 'test']:
        print(f"\n[{set_name.upper()}] Sensitivity Summary:")
        for target in [0.2, 0.4, 0.6]:
            stats = sensitivity_stats[set_name][target]
            if stats:
                epochs, sens, spec, thresh = zip(*stats)
                avg_sens = np.mean(sens)
                avg_spec = np.mean(spec)
                avg_thresh = np.mean(thresh)
                min_sens_idx = np.argmin(sens)
                max_sens_idx = np.argmax(sens)
                print(f"  Specificity Target {int(target*100)}%:")
                print(f"    Avg Sensitivity: {avg_sens*100:.2f}%, Avg Specificity: {avg_spec*100:.2f}%, Avg Threshold: {avg_thresh:.4f}")
                print(f"    Min Sensitivity: {sens[min_sens_idx]*100:.2f}% (Epoch {epochs[min_sens_idx]})")
                print(f"    Max Sensitivity: {sens[max_sens_idx]*100:.2f}% (Epoch {epochs[max_sens_idx]})")

    print("\nGenerating sensitivity plots...")
    plot_sensitivity_over_epochs(sensitivity_stats, 'val', save_path, run_name)
    plot_sensitivity_over_epochs(sensitivity_stats, 'test', save_path, run_name)


def evaluate(model, dataloader, criterion, device, label_mapping, epoch,
             log_path=None, run=DummyRun(), set_name="val"):
    model.eval()
    running_loss = 0.0
    total = 0

    metrics = {
        'preds_cad': [], 'targets_cad': [], 'probs_cad': [],
        'preds_tp': [], 'targets_tp': [], 'probs_tp': [],
    }

    with torch.no_grad():
        for batch in dataloader:
            videos, tabs, views, labels = batch[:4]
            videos = videos.to(device)
            tabs = tabs.to(device)
            views = views.to(device)

            if len(labels) == 1:
                labels = labels[0].to(device)
                outputs = model(videos, tabs, views)
                loss = criterion(outputs.squeeze(), labels.squeeze().float())
                preds = outputs.argmax(dim=1) if outputs.squeeze().ndim == 2 else (outputs > 0).long()
                metrics['preds_tp'].append(preds.cpu())
                metrics['targets_tp'].append(labels.cpu())
                if outputs.squeeze().ndim <= 1:
                    metrics['probs_tp'].append(torch.sigmoid(outputs.detach()).cpu())
                else:
                    metrics['probs_tp'].append(torch.softmax(outputs.detach(), dim=1).cpu())
            else:
                labels_tp, labels_cad = [l.to(device) for l in labels]
                logits_tp, logits_cad = model(videos, tabs, views)

                loss_tp = criterion['tp'](logits_tp.squeeze(), labels_tp.squeeze().float())
                loss_cad = criterion['cad'](logits_cad, labels_cad)
                loss = loss_tp + loss_cad * 3

                preds_tp = (logits_tp > 0).long()
                probs_tp = torch.sigmoid(logits_tp)

                preds_cad = logits_cad.argmax(dim=1)
                probs_cad = torch.softmax(logits_cad, dim=1)

                metrics['preds_tp'].append(preds_tp.cpu())
                metrics['targets_tp'].append(labels_tp.cpu())
                metrics['probs_tp'].append(probs_tp.cpu())

                metrics['preds_cad'].append(preds_cad.cpu())
                metrics['targets_cad'].append(labels_cad.cpu())
                metrics['probs_cad'].append(probs_cad.cpu())

            running_loss += loss.item() * videos.size(0)
            total += videos.size(0)

    run_data = {
        f"loss/{set_name}": running_loss / total,
    }

    for task in ['tp', 'cad']:
        if metrics[f'preds_{task}']:
            preds = torch.cat(metrics[f'preds_{task}']).numpy()
            targets = torch.cat(metrics[f'targets_{task}']).numpy()
            probs = torch.cat(metrics[f'probs_{task}']).numpy()

            run_data[f"acc/{set_name}_{task}"] = 100.0 * accuracy_score(targets, preds)
            run_data[f"acc_balanced/{set_name}_{task}"] = 100.0 * balanced_accuracy_score(targets, preds)
            run_data[f"worst_acc/{set_name}_{task}"] = 100.0 * compute_worst_class_accuracy(targets, preds)
            run_data[f"f1/{set_name}_{task}"] = 100.0 * f1_score(targets, preds, average='macro')

            run_data[f"auc/{set_name}_{task}"] = safe_auc_score(
                targets, probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs,
                average='macro', multi_class='ovr'
            )

            spec_sens_results = {}
            if task == 'tp':
                # tn, fp, fn, tp_ = confusion_matrix(targets, preds).ravel()
                # run_data[f"sensitivity/{set_name}_{task}"] = 100.0 * tp_ / (tp_ + fn) if (tp_ + fn) > 0 else float('nan')
                # run_data[f"specificity/{set_name}_{task}"] = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else float('nan')

                # # Compute threshold-based sensitivities at specificities
                # probs_binary = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs
                # fpr, tpr, thresholds = roc_curve(targets, probs_binary)
                # for spec_target in [0.2, 0.4, 0.6]:
                #     spec_arr = 1 - fpr
                #     valid_indices = np.where(spec_arr >= spec_target)[0]
                #     if len(valid_indices) > 0:
                #         idx = valid_indices[-1]  # Choose the threshold with highest TPR while keeping specificity ≥ target
                #         thresh = thresholds[idx]
                #         preds_spec = (probs_binary >= thresh).astype(int)
                #         tn, fp, fn, tp_ = confusion_matrix(targets, preds_spec).ravel()
                #         sens = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else float('nan')
                #         spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
                #         key = f"spec{int(spec_target*100)}"
                #         run_data[f"sensitivity_{key}/{set_name}_{task}"] = 100.0 * sens
                #         run_data[f"specificity_{key}/{set_name}_{task}"] = 100.0 * spec
                #         run_data[f"threshold_{key}/{set_name}_{task}"] = thresh
                #         spec_sens_results[spec_target] = (sens, spec, thresh)

                #         print(f"[{set_name.upper()}] Specificity Target: {int(spec_target*100)}% → Sensitivity: {sens*100:.2f}%, Specificity: {spec*100:.2f}%, Threshold: {thresh:.4f}")

                # ───────────────────────── default metrics ─────────────────────────
                tn, fp, fn, tp_ = confusion_matrix(targets, preds).ravel()
                run_data[f"sensitivity/{set_name}_{task}"] = 100.0 * tp_ / (tp_ + fn) if (tp_ + fn) else float('nan')
                run_data[f"specificity/{set_name}_{task}"] = 100.0 * tn  / (tn  + fp) if (tn  + fp) else float('nan')

                # We will collect both:
                #   • sensitivity at fixed-specificity targets (0.2,0.4,0.6)
                #   • specificity at fixed-sensitivity targets (0.2,0.4,0.6)
                probs_binary = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs
                fpr, tpr, thresholds = roc_curve(targets, probs_binary)
                spec_arr = 1 - fpr

                # ---------- 1.  Sensitivity at fixed-specificity targets ----------
                for spec_target in [0.2, 0.4, 0.6]:
                    idxs = np.where(spec_arr >= spec_target)[0]
                    if not len(idxs):
                        continue
                    idx      = idxs[-1]                       # highest TPR while spec ≥ target
                    thresh   = thresholds[idx]
                    preds_sp = (probs_binary >= thresh).astype(int)
                    tn, fp, fn, tp_ = confusion_matrix(targets, preds_sp).ravel()
                    sens = tp_ / (tp_ + fn) if (tp_ + fn) else float('nan')
                    spec = tn  / (tn  + fp) if (tn  + fp) else float('nan')

                    key = f"spec{int(spec_target*100)}"
                    run_data[f"sensitivity_{key}/{set_name}_{task}"] = 100.*sens
                    run_data[f"specificity_{key}/{set_name}_{task}"] = 100.*spec
                    run_data[f"threshold_{key}/{set_name}_{task}"]   = thresh
                    spec_sens_results[spec_target] = (sens, spec, thresh)

                    print(f"[{set_name.upper()}] Specificity Target: {spec_target:4.2f} → "
                          f"Sensitivity: {sens*100:5.2f}%, Specificity: {spec*100:5.2f}%, "
                          f"Threshold: {thresh:.4f}")
                    
            show_confusion_matrix(targets, preds, f"{set_name}_{task}", run=run)

    run.log(run_data, commit=False)
    return run_data[f"loss/{set_name}"], run_data.get(f"acc_balanced/{set_name}_tp"), spec_sens_results


def save_predictions(studies, all_targets, all_preds, all_probs, log_path, set_name):
    patient_id = [study['patient_id'] for study in studies]
    df = pd.DataFrame()
    df['patient_id'] = patient_id
    df['predicted_label'] = all_preds
    df['label'] = all_targets

    for k, probs in enumerate(all_probs.T):
        df[f'prob_{k}'] = probs
    df.to_csv(f'{os.path.dirname(log_path)}/{set_name}_preds.csv', index=False)
    return df


def pretrain_model(model, train_loader, val_loader, test_loader, device, run, epochs=10, lr=1e-4, save_path=None):
    # Pretraining mode

    model.freeze_task_head()
    model.unfreeze_pretrain_head()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0,
    )
    lr_scheduler = get_lr_scheduler(optimizer, lr, len(train_loader), epochs, pct_start=0.2)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for videos, views, labels, _ in train_loader:
            videos, views, labels = videos.to(device), views.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos, views, training=True, head="view")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total
        print(f"[Train] Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%")

        # Evaluate
        val_acc = evaluate_view_classification(model, val_loader, device, set_name="val")
        test_acc = evaluate_view_classification(model, test_loader, device, set_name="test")

        # Log to wandb
        run.log({
            "pretrain/loss_train": avg_loss,
            "pretrain/acc_train": train_acc,
            "pretrain/acc_val": val_acc,
            "pretrain/acc_test": test_acc,
            "pretrain/lr": optimizer.param_groups[0]['lr'],
        }, commit=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        run.log({"pretrain_step": epoch + 1})

    print(f"[Pretraining] Best val acc: {best_val_acc:.2f}%")
    if save_path is not None:
        torch.save(best_model_state, os.path.join(save_path, "pretrain_best.pth"))
        print(f"[Pretraining] Saved best model at: {save_path}/pretrain_best.pth")

    # Restore best model and remove view classification head if needed
    model.load_state_dict(best_model_state)


def evaluate_view_classification(model, dataloader, device, set_name="val"):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for videos, views, labels, _ in dataloader:
            videos = videos.to(device)
            views = views.to(device)
            labels = labels.to(device)

            outputs = model(videos, views, training=False, head="view")
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * accuracy_score(torch.cat(all_labels), torch.cat(all_preds))
    print(f"[{set_name.capitalize()}] View classification accuracy: {acc:.2f}%")
    return acc


def save_embeddings(model, split, use_view_tokens, target, args, device):
    model.eval()
    model.to(device)

    file_suffix = '_grouped_by_patient' if use_view_tokens else ''
    path = f'{EMB_DIR}/echoprime_{split}{file_suffix}.pt'

    # Load grouped_by_patient video_info (study-level)
    video_info = torch.load(path, weights_only=False)
    video_info = filter_by_class(video_info, target, dropped_class=None)

    # Create dataset
    dataset = ViewLabelV1(organize_by_patient(video_info, target=target))
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn_v1, pretrain=False)
    )

    embeddings_dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting embeddings [{split}]"):
            videos, _, _, _, indices = batch
            videos = videos.to(device)

            # Extract CLS embedding (study-level)
            cls_embeddings = model.extract_cls_embedding(videos)  # [B, D]

            for i, idx in enumerate(indices):
                idx = int(idx)
                embeddings_dict[idx] = cls_embeddings[i].cpu()

    # Load grouped_by_patient video_info (study-level)
    path = f'{EMB_DIR}/echoprime_{split}_grouped_by_patient.pt'
    video_info = torch.load(path, weights_only=False)

    # Add embedding to video_info
    for idx in range(len(video_info)):
        if idx in embeddings_dict:
            video_info[idx]['embedding'] = embeddings_dict[idx]
        else:
            print(f"Warning: Missing embedding for index {idx}")

    # Save enriched data
    save_path = f'{EMB_DIR}/echoprime_{split}{file_suffix}_with_{args.target}_embeddings.pt'
    torch.save(video_info, save_path)
    print(f"✅ Saved study-level embeddings to {save_path}")


def main(args):
    if args.debug:
        args.no_wandb = True
        args.num_workers = 0

    fix_seed(args.seed, True)
    torch.cuda.empty_cache()

    if args.exp_dir is None:
        exp_dir = f"checkpoints/{args.prefix}{args.exp_name}{args.suffix}/"
    else:
        exp_dir = args.exp_dir
    run_name = args.prefix + args.exp_name + args.suffix

    if args.no_wandb:
        run = DummyRun()
    else:
        run = wandb.init(project="recom_therapy",
                         entity="rcl_stroke",
                         config=vars(args),
                         group=args.wdb_group,
                         dir=exp_dir,
                         name=run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrain_view:
        model = StudyClassifier(
            num_classes=NUM_CLASSES[args.target],
            num_layers=args.num_layers,
            nhead=args.nhead,
        )
        loaders = set_loaders(args, pretrain_view=True, target='view')
        print(f"[Pretraining] Running view classification pretraining for {args.pretrain_epochs} epochs...")

        # wandb.define_metric('pretrain_step')
        if not isinstance(run, DummyRun):
            wandb.define_metric("pretrain_step")
            wandb.define_metric("pretrain/*", step_metric="pretrain_step")

        pretrain_model(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            test_loader=loaders['test'],
            device=device,
            run=run,
            epochs=args.pretrain_epochs,
            save_path=exp_dir,
            lr=1e-4,
        )
        return

    if args.use_view_tokens:
        model = StudyClassifierFromTokens(emb_dim=512 * 3, num_classes=NUM_CLASSES[args.target])
        # model = StudyClassifierFromTokensInterView(embed_dim=512, num_classes=NUM_CLASSES[args.target])
    else:
        if args.multitask:
            study_classifier_init = StudyClassifierMultiTask
        else:
            match args.mode:
                case "video+tab":
                    study_classifier_init = StudyClassifierV1
                case "late_fusion":
                    study_classifier_init = StudyClassifierV1LateFusion
                case "video":
                    study_classifier_init = StudyClassifierVideoOnly
                case _:
                    raise ValueError(f"Unknown mode {args.mode}")
        model = study_classifier_init(
            num_layers=args.num_layers,
            nhead=args.nhead,
            num_classes=NUM_CLASSES['cad'],
        )
    model.to(device)

    if args.tab_weight is not None and args.mode == 'late_fusion':
        tab_state = torch.load(args.tab_weight, map_location="cpu")
        missing, unexpected = model.tab_classifier.load_state_dict(tab_state, strict=False)
        print("→ Loaded tab_classifier weights from", args.tab_weight,
            "\n   • missing keys:", missing,
            "\n   • unexpected keys:", unexpected)
        if not missing and not unexpected:
            print("Proceeding to model weight freezing.")
            for p in model.tab_classifier.parameters():
                p.requires_grad_(False)
    # After pretraining, train the task classifier

    set_loaders_fn = set_loaders_multitask if args.multitask else set_loaders
    loaders = set_loaders_fn(args, use_view_tokens=args.use_view_tokens, target=args.target,
                             dropped_class=args.dropped_class)

    train_model(model,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                test_loader=loaders['test'],
                label_mapping=LABEL_MAPPING_DICT[args.target],
                device=device,
                save_path=exp_dir,
                checkpoint_name=f"embedding_{args.mode}_balancedSampler",
                num_epochs=args.epochs,
                lr=args.learning_rate,
                wd=args.weight_decay,
                loss_fn=args.loss_fn,
                mode="study",
                run=run,
                epoch_start=0,
                optim=args.optim,
                num_classes=NUM_CLASSES[args.target],
                run_name=run_name,
                freeze=args.freeze,
                multitask=args.multitask,
                )

    if args.save_embeddings:
        for split in ['train', 'val', 'test']:
            save_embeddings(model, split, args.use_view_tokens,
                            args.target, args, device)


