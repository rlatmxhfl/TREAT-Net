import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
sys.path.append('..')

import argparse
import warnings
from functools import partial

import torch
import torch.nn as nn
import wandb
import random
import numpy as np
import pandas as pd
import matplotlib as plt
# from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from scipy.interpolate import interp1d
from tqdm import tqdm

from .misc import DummyRun, show_trainable
from .model_training import get_lr_scheduler
from .classifiers import StudyClassifierVideoOnly, StudyClassifierV1, StudyClassifierV1LateFusion
from .dataloader import EMB_DIR, LABEL_MAPPING_DICT, NUM_CLASSES, set_loaders
from .evaluation import *
from ..utils.visualization import *

# from model.utils.fix_seed import fix_seed

VIEWS = ['AP2', 'AP4', 'PLAX']

def fix_seed(seed, benchmark=False):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark

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
                    lr_scheduler=None, run=DummyRun()):
    model.train()
    running_loss, total = 0.0, 0

    metrics = {'preds': [], 'targets': [], 'probs': []}

    for batch in tqdm(dataloader, desc="Training", mininterval=33):
        videos, tabs, views, labels = batch[:4]

        videos = videos.to(device)
        tabs = tabs.to(device)
        views = views.to(device)

        optimizer.zero_grad()

        labels = labels[0].to(device)

        logits = model(videos, tabs, views, training=True)
        loss = criterion(logits.squeeze(), labels.squeeze().float()) #squeeze means removing extra dimensions (e.g., [32, 1] → [32])

        preds = (logits > 0).long()
        probs = torch.sigmoid(logits.detach())
        metrics['probs'].append(probs.detach().cpu())
        metrics['preds'].append(preds.detach().cpu())
        metrics['targets'].append(labels.cpu())

        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss.item() * videos.size(0) #accumulate the loss across mini-batches within an epoch
        total += videos.size(0)

    run_data = {"loss/train": running_loss / total}

    preds = torch.cat(metrics[f'preds']).numpy()
    targets = torch.cat(metrics[f'targets']).numpy()
    probs = torch.cat(metrics[f'probs']).numpy()

    epoch_result = model_eval(
        y_true=targets,
        y_pred=preds,
        y_proba=probs,
        specificities=(0.2, 0.4, 0.6),
        average="weighted",
        verbose=False,
        set_name="train"
    )

    epoch_metrics = epoch_result["metrics"]
    epoch_class_acc = epoch_metrics.get("class accuracy", {})

    run_data ={
        "loss/train": running_loss / total,
        "acc/train": 100.0 * epoch_metrics["Accuracy"],
        "acc_balanced/train": 100.0 * epoch_metrics["Balanced Accuracy"],
        "worst_acc/train": 100.0 * compute_worst_class_accuracy(targets, preds),
        "f1/train": 100.0 * epoch_metrics["F1 Score"],
        "auc/train": epoch_metrics["ROC AUC"],
    }

    run.log(run_data, commit=False)
    return run_data["loss/train"], run_data.get("acc_balanced/train")

def train_model(model, train_loader, val_loader, test_loader, label_mapping, device,
                save_path, checkpoint_name, num_epochs=30, lr=1e-5, wd=1e-3,
                loss_fn=None, mode="study", run=DummyRun(), epoch_start=0, optim='Adam',
                num_classes=3, run_name='', freeze=False, multitask=False):
    os.makedirs(save_path, exist_ok=True)

    if optim == 'Adam':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=lr, weight_decay=wd,
                                        # amsgrad=True
                                        )
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=lr, weight_decay=wd, momentum=0.9, nesterov=False)
    lr_scheduler = get_lr_scheduler(optimizer, lr, len(train_loader), num_epochs - epoch_start, pct_start=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr, epochs=num_epochs,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.3
    # )
    show_trainable(model)

    # criterion = nn.BCEWithLogitsLoss()

    ###### modification to criterion to match mlp ######

    all_labels = []
    for batch in train_loader:
        labels = batch[3]
        all_labels.append(labels[0])
    all_labels = torch.cat(all_labels).long()

    pos_freq = all_labels.sum().item()
    neg_freq = len(all_labels) - pos_freq
    pos_weight_val = neg_freq / (pos_freq + 1e-8)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, dtype=torch.float, device=device))

    ####################################################

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = -1
    best_checkpoint_path = None

    sensitivity_stats = {
        'val':  {0.2: [], 0.4: [], 0.6: []},
        'test': {0.2: [], 0.4: [], 0.6: []},
    }

    for epoch in range(epoch_start, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_bal_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            lr_scheduler=lr_scheduler, run=run)

        val_loss, val_bal_acc, val_sens_data = evaluate(
            model, val_loader, criterion, device, epoch, run=run, set_name='val')

        _, test_bal_acc, test_sens_data = evaluate(
            model, test_loader, criterion, device, epoch, run=run, set_name='test')

        for target in (0.2, 0.4, 0.6):
            sens_v, spec_v, thr_v = val_sens_data[target]
            sensitivity_stats["val"][target].append((epoch, sens_v, spec_v, thr_v))
            sens_t, spec_t, thr_t = test_sens_data[target]
            sensitivity_stats["test"][target].append((epoch, sens_t, spec_t, thr_t))

        run.log({
            "loss/train": train_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1,
        }, commit=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

    final_path = os.path.join(save_path, f"{checkpoint_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    # print(f"\nFinal model saved: {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    print("\n=== Sensitivity @ Specificity summary across epochs ===")
    for set_name in ['val', 'test']:
        print(f"\n[{set_name.upper()}] Sensitivity Summary:")
        for target in (0.2, 0.4, 0.6):
            stats = sensitivity_stats[set_name][target]
            epochs, sens, spec, thresh = zip(*stats)
            sens = np.array(sens); spec = np.array(spec); thresh = np.array(thresh)

            avg_sens   = np.mean(sens)
            avg_spec   = np.mean(spec)
            avg_thresh = np.mean(thresh)
            min_idx    = int(np.argmin(sens))
            max_idx    = int(np.argmax(sens))

            print(f"  Specificity Target {int(target * 100)}%:")
            print(f"    Avg Sensitivity: {avg_sens*100:.2f}% "
                f"(Avg Specificity: {avg_spec*100:.2f}%, Avg Threshold: {avg_thresh:.4f})")
            print(f"    Min Sensitivity: {sens[min_idx]*100:.2f}% (Epoch {epochs[min_idx]})")
            print(f"    Max Sensitivity: {sens[max_idx]*100:.2f}% (Epoch {epochs[max_idx]})")

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, epoch, run=DummyRun(), set_name="val"):

    model.eval()
    running_loss, total = 0.0, 0

    preds_list, targets_list, probs_list = [], [], []

    for batch in dataloader:
        videos, tabs, views, labels = batch[:4]

        videos = videos.to(device)
        tabs = tabs.to(device)
        views = views.to(device)
        labels = labels[0].to(device)

        logits = model(videos, tabs, views, training=False)

        loss = criterion(logits.squeeze(), labels.squeeze().float())

        preds = (logits > 0).long()
        probs = torch.sigmoid(logits) #don't need .detach() here because it's inside torch.no_grad - but need to include for training loops

        preds_list.append(preds.cpu())
        targets_list.append(labels.cpu())
        probs_list.append(probs.cpu())

        running_loss += loss.item() * videos.size(0)
        total += videos.size(0)

    loss = running_loss/total
    preds = torch.cat(preds_list).numpy()
    targets = torch.cat(targets_list).numpy()
    probs = torch.cat(probs_list).numpy()

    result = model_eval(
        y_true=targets,
        y_pred=preds,
        y_proba=probs,
        specificities=(0.2, 0.4, 0.6),
        average="weighted",
        verbose=False,
        set_name=set_name)

    metrics = result["metrics"]
    sens_at_spec = result.get("sensitivities", {})

    run_data = {
        f"loss/{set_name}": loss,
        f"acc/{set_name}": 100.0 * metrics["Accuracy"],
        f"acc_balanced/{set_name}": 100.0 * metrics["Balanced Accuracy"],
        f"f1/{set_name}": 100.0 * metrics["F1 Score"],
        f"auc/{set_name}": metrics["ROC AUC"],
    }

    for t, (sens_t, spec_t, thr_t) in sens_at_spec.items():
        pct = int(t*100)
        run_data[f"sens_at_spec_{pct}/{set_name}"] = sens_t
        run_data[f"spec_at_spec_{pct}/{set_name}"] = spec_t
        run_data[f"thr_at_spec_{pct}/{set_name}"]  = thr_t

    run.log(run_data, commit=False)
    return loss, run_data.get(f"acc_balanced/{set_name}"), sens_at_spec

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

        if args.mode == "video+tab":
            study_classifier_init = StudyClassifierV1
        elif args.mode == "late_fusion":
            study_classifier_init = StudyClassifierV1LateFusion
        elif args.mode == "video":
            study_classifier_init = StudyClassifierVideoOnly
        else:
            raise ValueError(f"Unknown mode {args.mode}")

        model = study_classifier_init(
            num_layers=args.num_layers,
            nhead=args.nhead,
            num_classes=NUM_CLASSES['tp'])

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

            # ###### 2-logit softmax to single-logit sigmoid ######
            #
            # weight = tab_state["2.weight"]
            # if weight.dim() == 2 and weight.size(0) == 2 and model.tab_classifier[2].weight.size(0) == 1:
            #     weight_single = (weight[1] - weight[0]).unsqueeze(0)
            #     tab_state["2.weight"] = weight_single
            #
            # #####################################################
            #
            # missing, unexpected = model.tab_classifier.load_state_dict(tab_state, strict=False)
            # print("→ Loaded tab_classifier weights from", args.tab_weight,
            #     "\n   • missing keys:", missing,
            #     "\n   • unexpected keys:", unexpected)
            # if not missing and not unexpected:
            #     print("Proceeding to model weight freezing.")
            #     for p in model.tab_classifier.parameters():
            #         p.requires_grad_(False)

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
                    checkpoint_name=f"embedding_{args.mode}",
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
        
        # ######### visualization #########

        # # 1) Inspect available module names (helps verify hook targets)
        # list_module_names(model)

        # # 2) Names to hook — adjust if your model uses different names
        # VIDEO_FFN = "cross_ffn"         # outputs [B, 1, D] before .squeeze(1)
        # TAB_HEAD  = "tab_classifier"    # tab MLP head
        # TASK_HEAD = "task_classifier"   # video/task head from StudyClassifierV1

        # # 3) Where to save things
        # plots_dir = os.path.join(exp_dir, "plots")
        # npy_dir   = os.path.join(exp_dir, "npy")
        # os.makedirs(plots_dir, exist_ok=True)
        # os.makedirs(npy_dir, exist_ok=True)

        # # 4) Collect features & logits from the *test* split
        # feats = collect_late_fusion_features(
        #     model, loaders['test'], device,
        #     video_ffn_name=VIDEO_FFN,
        #     tab_head_name=TAB_HEAD,
        #     task_head_name=TASK_HEAD
        # )

        # # 5) Save raw arrays so you can reload in a notebook
        # np.save(os.path.join(npy_dir, "cls_repr.npy"),   feats["cls_repr"])
        # np.save(os.path.join(npy_dir, "tab_pen.npy"),    feats["tab_pen"])
        # if feats["task_pen"] is not None:
        #     np.save(os.path.join(npy_dir, "task_pen.npy"), feats["task_pen"])
        # np.save(os.path.join(npy_dir, "y.npy"),          feats["y"])
        # np.save(os.path.join(npy_dir, "tab_logit.npy"),  feats["tab_logit"])
        # np.save(os.path.join(npy_dir, "task_logit.npy"), feats["task_logit"])
        # print(f"[Saved NPYs] → {npy_dir}")

        # # 6) Print learned fusion weights
        # with torch.no_grad():
        #     w = torch.softmax(model.fusion_logits, dim=0).cpu().numpy()
        # print(f"[Fusion weights] w_task={w[0]:.3f}, w_tab={w[1]:.3f}")

        # # 7) UMAPs on fused CLS representation (2D + 3D)
        # run_umap(
        #     feats["cls_repr"], feats["y"],
        #     out_png=os.path.join(plots_dir, "umap_cls_repr_2d.png"),
        #     title="UMAP (2D) — fused CLS representation",
        #     n_components=2, seed=args.seed
        # )
        # run_umap(
        #     feats["cls_repr"], feats["y"],
        #     out_png=os.path.join(plots_dir, "umap_cls_repr_3d.png"),
        #     title="UMAP (3D) — fused CLS representation",
        #     n_components=3, seed=args.seed
        # )

        # # 8) UMAPs on tab head penultimate activations (2D + 3D)
        # run_umap(
        #     feats["tab_pen"], feats["y"],
        #     out_png=os.path.join(plots_dir, "umap_tab_penultimate_2d.png"),
        #     title="UMAP (2D) — tab head penultimate",
        #     n_components=2, seed=args.seed
        # )
        # run_umap(
        #     feats["tab_pen"], feats["y"],
        #     out_png=os.path.join(plots_dir, "umap_tab_penultimate_3d.png"),
        #     title="UMAP (3D) — tab head penultimate",
        #     n_components=3, seed=args.seed
        # )

        # # 9) UMAPs on task head penultimate activations (if present)
        # if feats["task_pen"] is not None:
        #     run_umap(
        #         feats["task_pen"], feats["y"],
        #         out_png=os.path.join(plots_dir, "umap_task_penultimate_2d.png"),
        #         title="UMAP (2D) — task head penultimate",
        #         n_components=2, seed=args.seed
        #     )
        #     run_umap(
        #         feats["task_pen"], feats["y"],
        #         out_png=os.path.join(plots_dir, "umap_task_penultimate_3d.png"),
        #         title="UMAP (3D) — task head penultimate",
        #         n_components=3, seed=args.seed
        #     )

        # # 10) 2D scatter of task vs tab logits (no UMAP)
        # plot_logit_plane(
        #     feats["task_logit"], feats["tab_logit"], feats["y"],
        #     out_png=os.path.join(plots_dir, "logit_plane_task_vs_tab.png"),
        #     title="Logit plane — task vs tab"
        # )

        # print(f"[Saved plots] → {plots_dir}")


