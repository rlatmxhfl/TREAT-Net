# Project Title

This repository contains the implementation of TREAT-Net, a framework that integrates echocardiography video embeddings with clinical tabular data to predict treatment strategies for acute coronary syndrome (ACS) patients. The model combines cross-attention fusion and late fusion to capture complementary information, improving robustness and labeled data efficiency.

## 📖 Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 📌 About
Describe the purpose of the project, the problem it solves, and why it’s useful.  
Optionally, add a project logo or demo screenshot.


## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/rlatmxhfl/nature.git
cd nature
pip install -r requirements.txt
```

## 🚀 Usage
### 1) Setup
\`\`\`bash
# Create env (Python ≥3.9)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # (or) pip install torch torchvision torchaudio scikit-learn pandas numpy tqdm wandb matplotlib
\`\`\`
- Weights & Biases is optional; add `-nw/--no_wandb` to disable logging.

---

### 2) Data & Inputs
TREAT-Net trains on **EchoPrime video embeddings** plus **clinical tabular features**.  
You have two ways to feed data:

**A. Precomputed embeddings (recommended)**
- Place the following files under `EMB_DIR` (see constant in `model/src/dataloader.py`):
  - `echoprime_train_grouped_by_mrn.pt`
  - `echoprime_val_grouped_by_mrn.pt`
  - `echoprime_test_grouped_by_mrn.pt`
- Update `EMB_DIR` in `model/src/dataloader.py` to your path.

**B. Raw-study CSV (cine/study mode)**
- CSV must include columns: `mrn_1`, `processed_file_address`, `view`, and `MANAG`  
  (`MANAG ∈ {MANAG, INTERVENTION}` for treatment labels; views use `AP2`, `AP4`, `PLAX`).
- See `model/src/dataloader.py:get_all_files_and_ground_truths` for the exact parsing.

---

### 3) Quick Start
Run from repo root:

\`\`\`bash
# Multimodal (video + tabular) with late fusion, treatment prediction (tp)
python main.py \
  --exp_dir runs/exp_tp_late \
  --target tp \
  --mode late_fusion \
  --epochs 150 \
  --batch_size 8 \
  --eval_batch_size 32 \
  --lr 1e-3 \
  --num_layers 2 \
  --nhead 4 \
  -nw  # disable wandb (optional)
\`\`\`

**Other common modes**
\`\`\`bash
# Video-only
python main.py --target tp --mode video -nw --exp_dir runs/exp_video

# Joint fusion (cross-attn + tabular)
python main.py --target tp --mode video+tab -nw --exp_dir runs/exp_vtab
\`\`\`

---

### 4) Useful Flags (subset)
- `--target {cad,tp}`: prediction task (default `cad`; repo mainly uses `tp`).
- `--mode {video,video+tab,late_fusion}`: fusion strategy.
- `--epochs`, `--batch_size`, `--eval_batch_size`, `--lr`, `--weight_decay`.
- `--num_layers`, `--nhead`: Transformer backbone size.
- `--freeze`: freeze backbone; `--unfreeze_encoder` to undo.
- `--save_embeddings`: dump learned embeddings.
- `--tab_weight /path/to/tab_model.pt`: load pretrained tabular weights.
- `--seed`, `--no_wandb` (a.k.a. `-nw`), `--debug`.

---

### 5) Outputs
- Checkpoints, metrics, and CSVs (predictions/probabilities) are saved under `--exp_dir`.
- Per-split predictions are written to `{exp_dir}/{split}_preds.csv` and `{split}_probs.csv`.

---


## 🔧 Configuration
Explain environment variables, configuration files, or arguments:

- `--data_dir`: Path to dataset  
- `--epochs`: Training epochs  
- `--lr`: Learning rate  

## 📊 Examples
Provide code snippets, images, or links to notebooks showing how the project works.

## 🤝 Contributing
1. Fork the repo  
2. Create your feature branch: `git checkout -b feature/my-feature`  
3. Commit changes: `git commit -m "Add new feature"`  
4. Push branch: `git push origin feature/my-feature`  
5. Open a Pull Request  

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments
We thank our supervisors, collaborators, and clinical partners for their invaluable support and feedback. This work was developed as part of our MICCAI 2025 workshop paper, "TREAT-Net: Tabular-Referenced Echo Analysis for Treatment prediction in ACS patients".
