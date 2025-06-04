# DL_vs_HateSpeech

This folder contains the main codebase for the Deep Learning Against Hate Speech project. Below is an overview of the folder structure and the main functions/classes in each file.

---

## Folder Structure

- **CLIP/**
  - `attention_clip.py`: Defines `AttentionCLIP`, a wrapper around HuggingFace CLIP with attention extraction and projection.

- **models/**
  - `base_model.py`: Abstract base class for all models, providing a common interface and parameter freezing checks.
  - `model_v0.py`: Defines `ModelV0`, a baseline multimodal model using CLIP and a transformer classifier.
  - `model_v1.py`: Defines `ModelV1`, a multimodal model using a fine-tuned CLIP and an attention classifier.
  - `model_v2.py`: Defines `ModelV2`, a multimodal model using `AttentionCLIP` and an attention classifier.
  - `utils.py`: Utility functions for loading models from checkpoints.

- **plots/**
  - `plot_loss.py`: Functions for plotting training/validation loss and metrics curves, and loading them from disk.

- **attention_rollout/**
  - Contains scripts and utilities for visualizing attention maps and rollouts.

- **loading_data/**
  - Contains data loading utilities and dataset classes.

- **training/**
  - Contains training loops, optimizer/criterion setup, and collate functions.

- **evaluation/**
  - Contains evaluation functions for computing metrics.

---

## Main Functions and Classes

### CLIP/attention_clip.py
- `AttentionCLIP`: Wrapper for HuggingFace CLIP with attention extraction and projection.

### models/base_model.py
- `BaseModel`: Abstract base class for all models, enforcing a common interface.

### models/model_v0.py
- `ModelV0`: Baseline model using CLIP and a transformer classifier.

### models/model_v1.py
- `ModelV1`: Model using fine-tuned CLIP and an attention classifier.

### models/model_v2.py
- `ModelV2`: Model using `AttentionCLIP` and an attention classifier.

### models/utils.py
- `load_model_from_path`: Loads a model from a checkpoint.

### plots/plot_loss.py
- `plot_losses_from_path`: Loads and plots loss curves from disk.
- `plot_metrics_from_path`: Loads and plots metrics curves from disk.
- `plot_losses`: Plots training and validation loss curves.
- `plot_metrics`: Plots accuracy and F1 score curves.

### utils.py
- Label conversion functions: `get_label_num`, `get_label_num_list`, `get_label_str`, `get_label_str_list`
- Data loading: `find_text_and_label_jsonl`, `load_json`
- Model parameter checking: `check_frozen_params`
- Config loading: `read_yaml_file`

---

For more details, see the docstrings in each file.