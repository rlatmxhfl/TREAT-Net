import os
import random

import numpy as np

import torch
from torchvision.transforms import v2


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")


def get_transform_v0(resolution=(224, 224), min_scale=0.8, flip_rate=0.3):
    """This will return a warning on deprecated version."""
    from torchvision.transforms._transforms_video import RandomResizedCropVideo, RandomHorizontalFlipVideo
    transform = v2.Compose(
        [RandomResizedCropVideo(size=resolution, scale=(min_scale, 1)),
         RandomHorizontalFlipVideo(p=flip_rate)]
    )
    return transform


def get_transform(
        resolution=(224, 224),
        min_scale=0.8,
        max_scale=1,
        flip_rate=0.5,
        # max_padding=10
):
    transform = v2.Compose([
        v2.RandomHorizontalFlip(p=flip_rate),
        # v2.Pad([max_padding, max_padding, max_padding, max_padding],
        #        padding_mode='edge'),
        v2.RandomResizedCrop(size=resolution, scale=(min_scale, max_scale)),
    ])
    return transform


def get_transform_echoclip(preprocess):
    import torchvision.transforms as T

    def __call__(video):
        """

        :param video: TxHxWx3
        :return:
        """
        return torch.stack([preprocess(T.ToPILImage()(frame)) for frame in video], dim=0)

    return __call__


class DummyRun:
    def __init__(self):
        ...

    def __call__(self, *args, **kwargs):
        ...

    def log(self, *args, **kwargs):
        ...

    def finish(self, *args, **kwargs):
        ...

    def define_metric(self, *args, **kwargs):
        ...


def show_video(video_array, gif_name='animation'):
    """show animation of a TxHxW (or TxHxWx3) array"""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if video_array.ndim == 4:
        video_array = (video_array - video_array.min()) / (video_array.max() - video_array.min())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    im = ax.imshow(video_array[0], cmap='gray', animated=True)
    ax.axis('off')  # Hide axes for better visualization

    def update(frame):
        im.set_array(video_array[frame])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=range(video_array.shape[0]), interval=100, blit=True
    )

    # Save the animation
    ani.save(f'{gif_name}.gif', writer=animation.PillowWriter(fps=10))
    plt.show()


# def fix_seed(seed, benchmark=False):
#     # Set PYTHONHASHSEED environment variable for consistent hashing
#     os.environ['PYTHONHASHSEED'] = str(seed)

#     # Set seeds for Python's random module, NumPy, and PyTorch
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

#     # Ensure deterministic behavior in cuDNN
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = benchmark


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_trainable_layer_parameters(model):
    for name, module in model.named_modules():
        # Only consider layers with trainable parameters
        params = list(p for p in module.parameters(recurse=False) if p.requires_grad)
        if params:
            num_params = sum(p.numel() for p in params)
            print(f"Layer: {name or '[root]'} | Trainable Parameters: {num_params}")


def show_trainable(model):
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    print_trainable_layer_parameters(model)