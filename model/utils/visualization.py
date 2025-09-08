import os
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt

def list_module_names(model, limit=200):
    print("\n[Named modules]")
    for i, (n, _) in enumerate(model.named_modules()):
        print(n)
        if i+1 >= limit:
            print("... (truncated)")
            break

def get_module_by_name(model, name: str):
    modules = dict(model.named_modules())
    if name not in modules:
        raise ValueError(f"Module '{name}' not found. Try one of:\n  " + "\n  ".join(list(modules.keys())[:30]))
    return modules[name]

def find_penultimate_in_sequential(seq: torch.nn.Sequential):
    kids = list(seq.children())
    last = kids[-1]
    if isinstance(last, torch.nn.Linear) and getattr(last, "out_features", None) == 1 and len(kids) >= 2:
        return kids[-2]
    return kids[-1]

def resolve_penultimate(model, root_name: str):
    root = get_module_by_name(model, root_name)
    if isinstance(root, torch.nn.Sequential):
        return find_penultimate_in_sequential(root)
    # look for a Sequential child
    for _, child in root.named_children():
        if isinstance(child, torch.nn.Sequential):
            return find_penultimate_in_sequential(child)
    # fallback: last child
    last_child = None
    for _, child in root.named_children():
        last_child = child
    return last_child if last_child is not None else root

@torch.no_grad()
def collect_late_fusion_features(model, dataloader, device,
                                 video_ffn_name="cross_ffn",
                                 tab_head_name="tab_classifier",
                                 task_head_name="task_classifier"):
    """
    Collect:
      - cls_repr after cross_ffn().squeeze(1)
      - tab penultimate activations (192-d ReLU)
      - task penultimate activations (if present)
      - task_logit, tab_logit
      - labels
    """
    # --- Resolve modules we want to hook ---
    video_ffn = get_module_by_name(model, video_ffn_name)
    tab_penult = resolve_penultimate(model, tab_head_name)
    try:
        task_penult = resolve_penultimate(model, task_head_name)
    except Exception:
        task_penult = None

    feats_cls, feats_tab_pen, feats_task_pen = [], [], []
    labels_all = []
    tab_logits_list, task_logits_list = [], []

    # Hooks
    def _hook_video_ffn(_m, _inp, out):
        feats_cls.append(out.squeeze(1).detach().cpu())

    def _hook_tab_pen(_m, _inp, out):
        feats_tab_pen.append(out.detach().cpu())

    def _hook_task_pen(_m, _inp, out):
        feats_task_pen.append(out.detach().cpu())

    def _hook_tab_head(_m, _inp, out):
        tab_logits_list.append(out.detach().cpu())

    def _hook_task_head(_m, _inp, out):
        task_logits_list.append(out.detach().cpu())

    # Register hooks
    h_video = video_ffn.register_forward_hook(_hook_video_ffn)
    h_tab_pen = tab_penult.register_forward_hook(_hook_tab_pen)
    h_task_pen = None
    if task_penult is not None:
        h_task_pen = task_penult.register_forward_hook(_hook_task_pen)

    tab_head = get_module_by_name(model, tab_head_name)
    task_head = get_module_by_name(model, task_head_name)
    h_tab_head = tab_head.register_forward_hook(_hook_tab_head)
    h_task_head = task_head.register_forward_hook(_hook_task_head)

    # Run dataloader
    model.eval()
    for batch in dataloader:
        videos, tabs, views, labels = batch[:4]
        videos, tabs, views = videos.to(device), tabs.to(device), views.to(device)
        labels = labels[0].long()
        _ = model(videos, tabs, views, training=False)
        labels_all.append(labels.cpu())

    # Remove hooks
    h_video.remove()
    h_tab_pen.remove()
    if h_task_pen is not None:
        h_task_pen.remove()
    h_tab_head.remove()
    h_task_head.remove()

    # Stack
    X_cls = torch.cat(feats_cls, dim=0).numpy()
    X_tab_pen = torch.cat(feats_tab_pen, dim=0).numpy()
    X_task_pen = torch.cat(feats_task_pen, dim=0).numpy() if feats_task_pen else None
    y = torch.cat(labels_all, dim=0).numpy()
    tab_logit = torch.cat(tab_logits_list, dim=0).squeeze(1).numpy()
    task_logit = torch.cat(task_logits_list, dim=0).squeeze(1).numpy()

    return {
        "cls_repr": X_cls,
        "tab_pen": X_tab_pen,
        "task_pen": X_task_pen,
        "tab_logit": tab_logit,
        "task_logit": task_logit,
        "y": y,
    }

def run_umap(X, y, out_png, title, n_components=2,
             n_neighbors=5, min_dist=0.5, metric="cosine", seed=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=seed,
    )
    Z = reducer.fit_transform(X)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    if n_components == 2:
        plt.figure(figsize=(7,5))
        sc = plt.scatter(Z[:,0], Z[:,1], c=y, cmap="coolwarm", s=6, alpha=0.8, linewidths=0)
        plt.colorbar(sc, label="Label (0=MANAG, 1=INTERVENTION)")
        plt.title(title); plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[UMAP] Saved → {out_png}")
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=y, cmap="coolwarm", s=4, alpha=0.9)
        ax.set_title(title); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
        fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.05, label="Label")
        plt.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[UMAP] Saved → {out_png}")

def plot_logit_plane(task_logit, tab_logit, y, out_png, title="Logit plane (task vs tab)"):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(6,5))
    sc = plt.scatter(task_logit, tab_logit, c=y, cmap="coolwarm", s=10, alpha=0.8, linewidths=0)
    plt.axvline(0.0, ls="--", lw=1, c="k", alpha=0.4)
    plt.axhline(0.0, ls="--", lw=1, c="k", alpha=0.4)
    plt.colorbar(sc, label="Label (0/1)")
    plt.xlabel("task_logit"); plt.ylabel("tab_logit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Logit plane] Saved → {out_png}")