import os
import csv
import random
from functools import partial
from typing import List, Dict
from collections import Counter

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy.io
from collections import defaultdict
# import EchoPrime.video_utils as video_utils
from skimage.transform import resize
# import cv2
import pandas as pd

prefix = "/mnt/rcl-server/workspace/diane"
EMB_DIR = r'/raid/home/minht/workspace/datasets/cardiac/embeddings_v1'

VIEW2ID = {'AP2': 0, 'AP4': 1, 'PLAX': 2}

LABEL_MAPPING_DICT = {
    'view': {'AP2': 0, 'AP4': 1, 'PLAX': 2},
    'tp': {"MANAG": 0, "INTERVENTION": 1}
    # 'tp': {"MANAG": 0, "PTCA": 1, "SURG": 1}
}

NUM_CLASSES = {'tp': len(np.unique(list(LABEL_MAPPING_DICT['tp'].values())))}

def get_all_files_and_ground_truths(
        filename,
        split='train',
        mode="study"  # or "cine"
):
    single_sample = defaultdict(lambda: {"files": [], "views": [], "target": None})
    results = []

    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get('Split') != split:
                continue

            mrn = row['mrn_1']
            raw_path = row['processed_file_address'].lstrip("/")
            view = row['view']
            target_GT = row['MANAG']
            full_path = os.path.join(prefix, raw_path)

            if mode == "cine":
                results.append((mrn, full_path, view, target_GT))  # one sample per video
            else:
                single_sample[mrn]["files"].append(full_path)
                single_sample[mrn]["views"].append(view)
                single_sample[mrn]["target"] = target_GT

    if mode == "study":
        for mrn, data in single_sample.items():
            results.append((mrn, data["files"], data["views"], data["target"])) #create a single entry per patient (e.g., (mrn, [list of files], [list of views], target))

    return results


class ACSDataset(Dataset):
    def __init__(
            self,
            files,
            frames_to_take=16,
            frame_stride=2,
            video_size=224,
            mean=None,
            std=None,
            transform=None,
            mode="study",  # or "cine"
            resolution=(224, 224),
            min_scale=0.8,
            flip_rate=0.3,
    ):
        self.files = files
        self.frames_to_take = frames_to_take
        self.frame_stride = frame_stride
        self.video_size = video_size
        self.mean = mean if mean is not None else torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = std if std is not None else torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        self.mode = mode
        self.transform = transform

    # def preprocess_video(self, mat_file_path):
    #     try:
    #         mat_data = scipy.io.loadmat(mat_file_path)
    #         if 'cropped' in mat_data:
    #             pixels = mat_data['cropped']
        #     elif 'cine' in mat_data:
        #         pixels = mat_data['cine']
        #     else:
        #         print(f"Missing expected keys in: {mat_file_path}")
        #         return None

        #     if pixels.ndim == 3:
        #         pixels = np.repeat(pixels[..., None], 3, axis=3)

        #     resized = resize(
        #         pixels,
        #         output_shape=(self.frames_to_take, self.video_size, self.video_size, 3),
        #         preserve_range=True,
        #         anti_aliasing=True,
        #     )

        #     x = torch.tensor(resized, dtype=torch.float32).permute(3, 0, 1, 2)

        #     x.sub_(self.mean).div_(self.std)

        #     assert x.shape == (3, self.frames_to_take, self.video_size, self.video_size), f"Invalid shape: {x.shape}"
        #     return x

        # except Exception as e:
        #     print("Preprocessed video failed:", mat_file_path)
        #     import traceback
        #     traceback.print_exc()
        #     return None

    def __getitem__(self, idx):
        pt_id, mat_paths, views, target_GT = self.files[idx]

        # Group videos by view
        view_to_paths = defaultdict(list)
        for path, view in zip(mat_paths, views):
            view_to_paths[view.upper()].append(path)  # normalize view names (e.g., 'AP2', 'ap2' → 'AP2')

        if self.mode == "cine":
            pt_id, mat_path, view, target_GT = self.files[idx]
            tensor = self.preprocess_video(mat_path)
            if tensor is None:
                return self.__getitem__((idx + 1) % len(self.files))
            return pt_id, tensor, target_GT

        elif self.mode == "study":
            selected_paths, selected_views = [], []
            for view in ['AP2', 'AP4', 'PLAX']:
                paths = view_to_paths.get(view, [])
                if len(paths) > 5:
                    paths = random.sample(paths, 5)
                selected_paths.extend(paths)
                selected_views.extend([view] * len(paths))

            video_tensors = []
            for mat_path in selected_paths:
                tensor = self.preprocess_video(mat_path)
                if tensor is not None:
                    video_tensors.append(tensor)

            if not video_tensors:
                with open("skipped_samples.log", "a") as f:
                    f.write(f"{pt_id}\n")
                return self.__getitem__((idx + 1) % len(self.files))

            video_tensor = torch.stack(video_tensors)  # [V, C, T, H, W]
            return pt_id, video_tensor, selected_views, target_GT

        return None


class PreprocessedACSDataset(ACSDataset):
    def preprocess_video(self, mat_paths):
        """This load the numpy file instead of the .mat file (which was resized to 224x224x16)"""
        file = mat_paths.replace('.mat', '.npy').replace('diane', 'minh/datasets')
        file = file.replace(
            '/mnt/rcl-server/workspace/minh',
            '/raid/home/minht/workspace',  # new storing location with faster loading speed
        )
        x = np.load(file)  # , mmap_mode='r')

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.files)


def custom_collate(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


def subsample_files(files, frac, random_state=42):
    files = pd.DataFrame(files, columns=['pid', 'filename', 'views', 'gt'])
    files = files.groupby('gt', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=random_state))
    files = list(files.itertuples(index=False, name=None))
    return files


def create_dataloader(
        csv_file,
        split='train',
        batch_size=1,
        shuffle=True,
        num_workers=8,
        debug=False,
        collate_fn=custom_collate,
        # balance_classes=False,
        balanced_batch=False,
        label_mapping=None,
        mode="study",  # or "cine"
        transform=None,
        subsample_frac=1.0,
):
    files = get_all_files_and_ground_truths(csv_file, split=split, mode=mode)
    if subsample_frac < 1:
        files = subsample_files(files, subsample_frac)

    if debug:
        files = files[:20]
        print(f"Debug mode: Only using {len(files)} patient samples for split '{split}'")

    # if balance_classes:
    #     print(f"Balancing classes for split '{split}'")
    #     # Group files by class
    #     class_to_files = defaultdict(list)
    #     for sample in files:
    #         label = sample[-1]
    #         class_to_files[label].append(sample)
    #
    #     # Find the minimum class size
    #     min_class_count = min(len(v) for v in class_to_files.values())
    #     print(f"🟰 Using {min_class_count} samples per class")
    #
    #     # Sample min_class_count from each class
    #     balanced_files = []
    #     for cls, samples in class_to_files.items():
    #         selected = random.sample(samples, min_class_count)
    #         balanced_files.extend(selected)
    #
    #     files = balanced_files

    dataset = PreprocessedACSDataset(files, mode=mode, transform=transform)

    if (split == 'train') and balanced_batch:
        train_y = torch.tensor([label_mapping[_[-1]] for _ in dataset.files])
        class_counts = torch.bincount(train_y)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[train_y]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            # shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            # collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # collate_fn=collate_fn,
            pin_memory=True,
        )

    return dataloader


def collate_fn_v0(batch):
    X, y, idx = zip(*batch)

    min_len = min((sum(len(_) for _ in sample.values()) for sample in X))

    for sample in X:
        for view in sample:
            emb_dim = sample[view].shape[-1]
            break

    batch_size = len(X)
    batched_videos = np.zeros((batch_size, min_len, emb_dim), dtype=np.float32)
    batched_views = np.zeros((batch_size, min_len), dtype=np.int64)

    for i, sample in enumerate(X):
        sample_views, sample_videos = [], []
        for view in sample.keys():
            for cine in sample[view]:
                sample_views.append(VIEW2ID[view])
                sample_videos.append(cine)
        chosen_idx = np.random.choice(len(sample_views), min_len, replace=False)
        sample_views = np.array(sample_views)[chosen_idx]
        sample_videos = np.array(sample_videos)[chosen_idx]

        batched_videos[i] = sample_videos
        batched_views[i] = sample_views

    return torch.tensor(batched_videos), torch.tensor(batched_views), torch.tensor(y), torch.tensor(idx)


def collate_fn_main_task():
    def collate(batch):
        X, y, idx = zip(*batch)

        for sample in X:
            for view in sample:
                emb_dim = sample[view][0].shape[-1]
                break

        min_len = min(sum(len(v) for v in sample.values()) for sample in X)
        B = len(X)
        batched_videos = np.zeros((B, min_len, emb_dim), dtype=np.float32)
        batched_views = np.zeros((B, min_len), dtype=np.int64)

        for i, sample in enumerate(X):
            vids, views = [], []
            for view in sample:
                for cine in sample[view]:
                    vids.append(cine)
                    views.append(VIEW2ID[view])

            idx_chosen = np.random.choice(len(vids), min_len, replace=False)
            vids = np.array(vids)[idx_chosen]
            views = np.array(views)[idx_chosen]

            batched_videos[i] = vids
            batched_views[i] = views

        return (
            torch.tensor(batched_videos),  # [B, L, D]
            torch.tensor(batched_views),  # [B, L]
            torch.tensor(y),  # [B]
            torch.tensor(idx),  # [B]
        )

    return collate


def collate_fn_v1(batch, pretrain=False):
    if pretrain:
        X, _, idx = zip(*batch)
    else:
        X, tab, y, idx = zip(*batch)

    # Get embedding dimension from first sample
    for sample in X:
        for view in sample:
            emb_dim = sample[view][0].shape[-1]
            break
    # For pretraining: randomly select one view per study, keep all its videos
    if pretrain:
        batch_videos, batch_views, view_labels = [], [], []
        for sample in X:
            available_views = list(sample.keys())
            chosen_view = random.choice(available_views)
            chosen_videos = sample[chosen_view]
            num_vids = len(chosen_videos)

            batch_videos.append(np.stack(chosen_videos))  # [num_vids, D]
            batch_views.append(np.full((num_vids,), VIEW2ID[chosen_view], dtype=np.int64))
            view_labels.append(VIEW2ID[chosen_view])  # one label per study

        # Pad to max length
        max_len = max(len(v) for v in batch_videos)
        batch_size = len(batch_videos)

        batched_videos = np.zeros((batch_size, max_len, emb_dim), dtype=np.float32)
        batched_views = np.full((batch_size, max_len), fill_value=-1, dtype=np.int64)

        for i, (vids, views) in enumerate(zip(batch_videos, batch_views)):
            batched_videos[i, :len(vids)] = vids
            batched_views[i, :len(views)] = views

        return (
            torch.tensor(batched_videos),  # [B, L, D]
            torch.tensor(batched_views),  # [B, L]
            (torch.tensor(view_labels),),  # [B]; adapted to multiple labels return
            torch.tensor(idx),  # [B]
        )
    else:
        # Original downstream task: pad to min length across all views
        min_len = min((sum(len(_) for _ in sample.values()) for sample in X))
        batch_size = len(X)

        batched_videos = np.zeros((batch_size, min_len, emb_dim), dtype=np.float32)
        batched_views = np.zeros((batch_size, min_len), dtype=np.int64)
        batched_tab = np.zeros((batch_size, tab[0].shape[-1]), dtype=np.float32)

        for i, sample in enumerate(X):
            sample_views, sample_videos = [], []
            for view in sample:
                for cine in sample[view]:
                    sample_views.append(VIEW2ID[view])
                    sample_videos.append(cine)

            chosen_idx = np.random.choice(len(sample_videos), min_len, replace=False)
            sample_videos = np.array(sample_videos)[chosen_idx]
            sample_views = np.array(sample_views)[chosen_idx]

            batched_videos[i] = sample_videos
            batched_views[i] = sample_views
            batched_tab[i] = tab[i]

        return (
            torch.tensor(batched_videos),
            torch.tensor(batched_tab),
            torch.tensor(batched_views),
            (torch.tensor(y),),  # [B]; adapted to multiple labels return
            torch.tensor(idx),
        )


def collate_fn_pretrain_eval(batch, pretrain=True):
    id2view = {v: k for k, v in VIEW2ID.items()}

    if pretrain:
        X, _, idx = zip(*batch)
    else:
        X, y, idx = zip(*batch)

    # --- Separate each view ---
    view_videos_dict = {view: [] for view in VIEW2ID}
    view_lengths = {view: [] for view in VIEW2ID}
    view_labels = {view: [] for view in VIEW2ID}
    view_indices = {view: [] for view in VIEW2ID}

    for i, sample in enumerate(X):  # each sample = a study (dict of views)
        for view, clips in sample.items():
            view_videos_dict[view].append(np.stack(clips))  # [num_clips, D]
            view_lengths[view].append(len(clips))
            view_labels[view].append(VIEW2ID[view])  # for pretraining, label = view id
            view_indices[view].append(idx[i])

    # --- Pad each view separately ---
    view_batch_output = {}
    for view in VIEW2ID:
        vids_list = view_videos_dict[view]
        if not vids_list:
            continue  # skip if this view is not in the batch

        emb_dim = vids_list[0].shape[-1]
        max_len = max(view_lengths[view])
        batch_size = len(vids_list)

        batched_videos = np.zeros((batch_size, max_len, emb_dim), dtype=np.float32)
        view_mask = np.full((batch_size, max_len), fill_value=-1, dtype=np.int64)

        for i, clips in enumerate(vids_list):
            L = len(clips)
            batched_videos[i, :L] = clips
            view_mask[i, :L] = VIEW2ID[view]

        view_batch_output[view] = {
            'videos': torch.tensor(batched_videos),  # [B, L, D]
            'view_masks': torch.tensor(view_mask),  # [B, L]
            'labels': torch.tensor(view_labels[view]),  # [B]
            'indices': torch.tensor(view_indices[view])  # [B]
        }

    if pretrain:
        return view_batch_output
    else:
        # For downstream task, merge views for min-length sampling (as you had before)
        # Or optionally adapt this to also return per-view if desired
        raise NotImplementedError("View-specific collation only supported in pretrain mode.")


class ViewLabel(Dataset):
    def __init__(self, grouped_data, use_view_tokens=False, target=None):
        self.grouped_data = grouped_data
        self.use_view_tokens = use_view_tokens
        self.target = target

    def __getitem__(self, index):
        if self.target is not None:
            if self.target == 'cad':
                k = str(int(float(self.grouped_data[index][self.target])))
            else:
                k = self.grouped_data[index][self.target]
            label = LABEL_MAPPING_DICT[self.target][k]
        else:
            label = self.grouped_data[index]['label']
        embeddings = self.grouped_data[index]['view_token'] if self.use_view_tokens \
            else self.grouped_data[index]['views']
        return embeddings, label, index

    def __len__(self):
        return len(self.grouped_data)


class ViewLabelV1(Dataset):
    def __init__(self, grouped_data, use_view_tokens=False, target=None, subsample_frac=None, subsample_n=None,
                 seed=42):
        assert not (subsample_frac and subsample_n), "Only one of subsample_frac or subsample_n should be set"
        self.use_view_tokens = use_view_tokens
        self.target = target

        # Subsample logic
        if subsample_frac:
            random.seed(seed)
            n_samples = int(len(grouped_data) * subsample_frac)
            self.grouped_data = random.sample(grouped_data, n_samples)
        elif subsample_n:
            random.seed(seed)
            self.grouped_data = random.sample(grouped_data, subsample_n)
        else:
            self.grouped_data = grouped_data

    def __getitem__(self, index):
        sample = self.grouped_data[index]
        if self.target is not None:
            if self.target == 'cad':
                k = str(int(float(sample[self.target])))
            else:
                k = sample[self.target]
            label = LABEL_MAPPING_DICT[self.target][k]
        else:
            label = sample['label']

        embeddings = sample['view_token'] if self.use_view_tokens else sample['views']
        if 'tab_embedding' in sample:
            tab_embedding = sample['tab_embedding'].astype(np.float32)
        else:
            tab_embedding = np.zeros(192, dtype=np.float32)
        return embeddings, tab_embedding, label, index

    def __len__(self):
        return len(self.grouped_data)


def organize_by_patient_v0(video_info_list, target='cad'):
    """

    :param video_info_list:
    :param target: 'cad' or 'tp'
    :return:
    """
    label_mapping = LABEL_MAPPING_DICT[target]

    grouped = defaultdict(lambda: {'views': defaultdict(list), 'label': None})
    for item in video_info_list:
        pid = item['patient_id']
        view = item['view'].upper()
        grouped[pid]['views'][view].append(item['embedding'])
        # grouped[pid]['label'] = label_mapping[str(int(float(item[target])))]  # string-safe
        k = str(int(float(item[target]))) if target == 'cad' else item[target]
        grouped[pid]['label'] = label_mapping[k]

    dataset = []
    for pid, data in grouped.items():
        sample = {'views': {}, 'label': data['label']}
        for view in ['AP2', 'AP4', 'PLAX']:
            if data['views'][view]:
                arr = np.stack(data['views'][view])  # shape [N, 512]
            else:
                arr = np.zeros((1, 512))  # fallback
            sample['views'][view] = arr
        dataset.append(sample)
    return dataset


def organize_by_patient(video_info_list, target='cad', use_view_tokens=False, fallback_zero=True):
    """
    Groups video embeddings by patient and view. Supports either:
      - 'views': {view -> [N, 512]} for clip-level sequences
      - 'view_token': {view -> [512]} for aggregated view-level vectors

    Args:
        video_info_list (list of dicts): Each dict must include:
            - 'patient_id'
            - 'view'
            - either 'embedding' or 'view_token'
            - 'cad' / 'tp' / 'view' as label
        target (str): Label to use ('cad', 'tp', or 'view')
        use_view_tokens (bool): If True, use 'view_token' instead of raw sequences
        fallback_zero (bool): If True, use zeros when a view is missing

    Returns:
        List of samples, each as:
            {
                'views' or 'view_token': {view -> np.array},
                'label': int
            }
    """

    label_mapping = LABEL_MAPPING_DICT[target]

    # Group data by patient
    grouped = defaultdict(lambda: {'views': defaultdict(list), 'view_token': {}, 'label': None})

    for item in video_info_list:
        pid = item['patient_id']
        view = item['view'].upper()

        if use_view_tokens:
            grouped[pid]['view_token'][view] = np.array(item['view_token'][view])
        else:
            grouped[pid]['views'][view].append(np.array(item['embedding']))

        if grouped[pid]['label'] is None:
            raw_label = str(int(float(item[target]))) if target == 'cad' else item[target]
            grouped[pid]['label'] = label_mapping[raw_label]

        grouped[pid]['patient_id'] = pid
        grouped[pid]['cad'] = item['cad']
        grouped[pid]['tp'] = item['tp']

    # Build dataset
    dataset = []
    for pid, data in grouped.items():
        sample = {'label': data['label'], 'cad': data['cad'], 'tp': data['tp'], 'patient_id': data['patient_id']}
        container = {}

        for view in ['AP2', 'AP4', 'PLAX']:
            if use_view_tokens:
                if view in data['view_token']:
                    container[view] = data['view_token'][view]  # shape [512]
                elif fallback_zero:
                    container[view] = np.zeros((512,), dtype=np.float32)
            else:
                if data['views'][view]:
                    container[view] = np.stack(data['views'][view])  # shape [N, 512]
                elif fallback_zero:
                    container[view] = np.zeros((1, 512), dtype=np.float32)

        key = 'view_token' if use_view_tokens else 'views'
        sample[key] = container
        dataset.append(sample)

    return dataset


def filter_by_class(video_info, target, dropped_class):
    if dropped_class is None:
        return video_info
    # 🔽 Filter unwanted classes (e.g., drop cad label 0 after mapping)
    filtered_info = []
    for sample in video_info:
        raw_label = str(int(float(sample['cad']))) if target == 'cad' else sample['tp']
        if raw_label != dropped_class:  # change `0` to the label you want to drop
            filtered_info.append(sample)

    return filtered_info


def collate_fn_concat_views(batch):
    X, y, idx = zip(*batch)

    emb_dim = 512
    d_total = emb_dim * 3
    B = len(X)

    batched_embeddings = np.zeros((B, d_total), dtype=np.float32)

    for i, sample in enumerate(X):
        view_embs = []
        for view in ['AP2', 'AP4', 'PLAX']:
            if view in sample:
                view_embs.append(sample[view])  # shape (512,)
            else:
                view_embs.append(np.zeros((emb_dim,), dtype=np.float32))  # zero fill

        batched_embeddings[i] = np.concatenate(view_embs)  # shape (1536,)

    return (
        torch.tensor(batched_embeddings),  # [B, 1536]
        torch.tensor([-1]),
        torch.tensor(y),
        torch.tensor(idx)
    )


def describe_class_distribution(video_info, target='cad', split_name=''):
    from collections import Counter
    from pprint import pprint

    """
    Print the raw and mapped class distributions in a list of video_info samples.

    Args:
        video_info (list): List of dicts, each with keys including the `target` (e.g., 'cad', 'tp').
        target (str): One of 'cad', 'tp', or 'view'.
        split_name (str): Optional name of the dataset split (e.g., 'train', 'val').
    """
    raw_label_counter = Counter()
    mapped_label_counter = Counter()

    for sample in video_info:
        raw_label = str(int(float(sample[target]))) if target == 'cad' else sample[target]
        mapped_label = LABEL_MAPPING_DICT[target][raw_label]
        raw_label_counter[raw_label] += 1
        mapped_label_counter[mapped_label] += 1

    print(f"\n=== Class Distribution ({split_name}) ===")
    print("Raw Label Distribution:")
    pprint(dict(raw_label_counter))
    print("Mapped Label Distribution:")
    pprint(dict(mapped_label_counter))


def collect_class_distributions_to_df(split2video_info: Dict[str, List[dict]], target: str = 'cad') -> pd.DataFrame:
    """
    Computes raw and mapped label distributions across multiple splits, returns as a pandas DataFrame.

    Args:
        split2video_info: Dict mapping split name -> list of video_info samples
        target: One of 'cad', 'tp', or 'view'

    Returns:
        A pandas DataFrame with split x label counts (both raw and mapped)
    """
    all_rows = []

    for split, video_info in split2video_info.items():
        raw_counter = Counter()
        mapped_counter = Counter()

        for sample in video_info:
            raw_label = str(int(float(sample[target]))) if target == 'cad' else sample[target]
            mapped_label = LABEL_MAPPING_DICT[target][raw_label]
            raw_counter[raw_label] += 1
            mapped_counter[mapped_label] += 1

        for k, v in raw_counter.items():
            all_rows.append({'Split': split, 'Label Type': 'Raw', 'Label': k, 'Count': v})

        for k, v in mapped_counter.items():
            all_rows.append({'Split': split, 'Label Type': 'Mapped', 'Label': k, 'Count': v})

    df = pd.DataFrame(all_rows)
    df = df.sort_values(by=['Label Type', 'Split', 'Label']).reset_index(drop=True)
    return df


class ViewLabelMultiTask(Dataset):
    def __init__(self, grouped_data, use_view_tokens=False, target_cad='cad', target_tp='tp',
                 subsample_frac=None, subsample_n=None, seed=42):
        assert not (subsample_frac and subsample_n), "Only one of subsample_frac or subsample_n should be set"
        self.use_view_tokens = use_view_tokens
        self.target_cad = target_cad
        self.target_tp = target_tp

        # Subsample
        if subsample_frac:
            random.seed(seed)
            n_samples = int(len(grouped_data) * subsample_frac)
            self.grouped_data = random.sample(grouped_data, n_samples)
        elif subsample_n:
            random.seed(seed)
            self.grouped_data = random.sample(grouped_data, subsample_n)
        else:
            self.grouped_data = grouped_data

    def __getitem__(self, index):
        sample = self.grouped_data[index]

        cad_label = LABEL_MAPPING_DICT[self.target_cad][str(int(float(sample[self.target_cad])))]
        tp_label = LABEL_MAPPING_DICT[self.target_tp][sample[self.target_tp]]

        embeddings = sample['view_token'] if self.use_view_tokens else sample['views']
        tab_embedding = sample['tab_embedding'].astype(np.float32)

        return embeddings, tab_embedding, (tp_label, cad_label), index

    def __len__(self):
        return len(self.grouped_data)


def collate_fn_multitask(batch):
    X, tab, labels, idx = zip(*batch)
    y_tp, y_cad = zip(*labels)

    min_len = min((sum(len(_) for _ in sample.values()) for sample in X))
    batch_size = len(X)
    emb_dim = list(X[0].values())[0][0].shape[-1]

    batched_videos = np.zeros((batch_size, min_len, emb_dim), dtype=np.float32)
    batched_views = np.zeros((batch_size, min_len), dtype=np.int64)
    batched_tab = np.zeros((batch_size, tab[0].shape[-1]), dtype=np.float32)

    for i, sample in enumerate(X):
        sample_views, sample_videos = [], []
        for view in sample:
            for cine in sample[view]:
                sample_views.append(VIEW2ID[view])
                sample_videos.append(cine)

        chosen_idx = np.random.choice(len(sample_videos), min_len, replace=False)
        sample_videos = np.array(sample_videos)[chosen_idx]
        sample_views = np.array(sample_views)[chosen_idx]

        batched_videos[i] = sample_videos
        batched_views[i] = sample_views
        batched_tab[i] = tab[i]

    return (
        torch.tensor(batched_videos),
        torch.tensor(batched_tab),
        torch.tensor(batched_views),
        (torch.tensor(y_tp), torch.tensor(y_cad),),
        torch.tensor(idx),
    )


def set_loaders(args, pretrain_view=False, collate_fn=None, use_view_tokens=False, target='cad',
                dropped_class=None, *_args, **_kwargs):
    dset, loaders = dict(), dict()
    # file_suffix = ''
    file_suffix = '_grouped_by_mrn'

    if collate_fn is None:
        if use_view_tokens:
            collate_fn = collate_fn_concat_views
        else:
            collate_fn = partial(collate_fn_v1, pretrain=pretrain_view)

    split2video_info = dict()

    # for split in ['train', 'val', 'test']:
    #     video_info = torch.load(f'{EMB_DIR}/echoprime_{split}{file_suffix}.pt', weights_only=False)
    #     video_info = filter_by_class(video_info, target, dropped_class)
    #
    #     if 'grouped_by_patient' not in file_suffix:
    #         video_info = organize_by_patient(video_info, target=target)
    # video_info_with_tab = torch.load(f'{EMB_DIR}/echoprime_{split}_grouped_by_patient.pt', weights_only=False)
    # for _viwt, _vi in zip(video_info_with_tab, video_info):
    #     assert _viwt['patient_id'] == _vi['patient_id']
    #     _viwt['label'] = _vi['label']
    # torch.save(video_info_with_tab, f'{EMB_DIR}/echoprime_{split}_grouped_by_patient.pt')

    for split in ['train', 'val', 'test']:
        video_info = torch.load(f'{EMB_DIR}/echoprime_{split}{file_suffix}.pt', weights_only=False)
        video_info = filter_by_class(video_info, target, dropped_class)

        if 'grouped_by_mrn' not in file_suffix:
            video_info = organize_by_patient(video_info, target=target)
            # video_info_with_tab = torch.load(f'{EMB_DIR}/echoprime_{split}_grouped_by_patient.pt', weights_only=False)
            # breakpoint()

        split2video_info[split] = video_info

        subsample_frac = None if split != 'train' else args.subsample_frac
        subsample_n = None if split != 'train' else args.subsample_n
        dset[split] = ViewLabelV1(video_info, subsample_frac=subsample_frac, subsample_n=subsample_n, target=target,
                                  use_view_tokens=use_view_tokens)

        sampler, shuffle = None, False
        if split in ['val', 'test']:
            shuffle = False
        elif split == 'train' and not args.non_balanced_batch and not pretrain_view:
            targets = torch.tensor([_[2] for _ in dset[split]])  # assuming (x, y, idx)
            class_counts = torch.bincount(targets)
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            shuffle = True

        loaders[split] = DataLoader(
            dset[split],
            batch_size=args.batch_size if split == 'train' else args.eval_batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    df_stats = collect_class_distributions_to_df(split2video_info, target=target)
    print(df_stats)

    return loaders


def set_loaders_multitask(args, pretrain_view=False, use_view_tokens=False,
                          target_cad='cad', target_tp='tp', dropped_class=None, *_args, **_kwargs):
    dset, loaders = dict(), dict()
    file_suffix = '_grouped_by_mrn'

    for split in ['train', 'val', 'test']:
        video_info = torch.load(f'{EMB_DIR}/echoprime_{split}{file_suffix}.pt', weights_only=False)
        video_info = filter_by_class(video_info, target_cad, dropped_class)

        if 'grouped_by_mrn' not in file_suffix:
            video_info = organize_by_patient(video_info, target=target_cad)

        subsample_frac = args.subsample_frac if split == 'train' else None
        subsample_n = args.subsample_n if split == 'train' else None

        dset[split] = ViewLabelMultiTask(
            video_info,
            use_view_tokens=use_view_tokens,
            target_cad=target_cad,
            target_tp=target_tp,
            subsample_frac=subsample_frac,
            subsample_n=subsample_n
        )

        sampler, shuffle = None, False
        if split == 'train' and not args.non_balanced_batch and not pretrain_view:
            targets = torch.tensor([_[2][0] for _ in dset[split]])  # assuming (x, y, idx)
            class_counts = torch.bincount(targets)
            class_weights = 1.0 / class_counts.float()
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            shuffle = True

        loaders[split] = DataLoader(
            dset[split],
            batch_size=args.batch_size if split == 'train' else args.eval_batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            collate_fn=collate_fn_multitask,
            sampler=sampler,
        )

    return loaders
