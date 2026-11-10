"""HEASNet component: Runtime imports, configuration, reproducibility and logging.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# =====================================================================================
# HEASNet FINAL ALL-IN-ONE EXPERIMENT  (HyperKvasir + Kvasir-V2)
# =====================================================================================
import os, gc, sys, io, json, time, random, warnings, shutil, re, copy, zipfile, platform, datetime, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score, cohen_kappa_score, matthews_corrcoef, log_loss,
    f1_score, precision_recall_curve, average_precision_score)
from scipy.stats import binomtest
warnings.filterwarnings("ignore")
T0 = time.time()

# ================================ SWITCHES (edit only here) ================================
QUICK = os.environ.get("HEAS_QUICK") == "1"
SEED = 42
EPOCHS, HEAD_WARMUP_EPOCHS = 40, 5
IMG_SIZE, BATCH_SIZE, NUM_WORKERS = 224, 16, 2
LR_HEAD, LR_BACKBONE, WEIGHT_DECAY = 1e-4, 1e-5, 1e-5
LABEL_SMOOTHING, DROPOUT = 0.05, 0.30
EMBED_DIM, NUM_HEADS, WINDOW_SIZE = 512, 8, 4
VAL_FRAC = TEST_FRAC = 0.15
BOOTSTRAP_REPS = 1000
USE_PRETRAINED = True
DPI = 500
REUSE_CHECKPOINT = False
GRADCAM_PER_CLASS = 2
FAITH_PER_CLASS, FAITH_STEPS = 5, 20
CORR_MAX_PER_CLASS = 150
DUP_HAMMING = 4
RUN_EXTRAS = True
EXTRA_DATASETS = ["HyperKvasir"]
RUN_ABLATION = True
RUN_BASELINES = True
REPEAT_SEEDS = []
RUN_SWEEPS = False
SWEEP_EPOCHS = 15
LR_SWEEP = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
OPT_SWEEP = ["adam", "sgd", "rmsprop"]
TIME_BUDGET_HOURS = 10.5
INCLUDE_ALL_CHECKPOINTS_IN_ZIP = False

DATASETS = {
    "HyperKvasir": {"root": "/kaggle/input/datasets/visheshtanwar26/processed-datatset",
                    "layout": "predefined_train_test", "expected_classes": int(os.environ.get("HEAS_HK_N", "12"))},
    "Kvasir-V2": {"root": "/kaggle/input/datasets/subhajeetdas/kvasir-v2-gastrointestinal-image-dataset/KVASIR V2",
                  "layout": "class_folders", "expected_classes": int(os.environ.get("HEAS_KV_N", "8"))},
}
OUTPUT_ROOT = "/kaggle/working/HEASNet_FINAL_ALL_RESULTS"
ZIP_BASE = "/kaggle/working/HEASNet_FINAL_ALL_RESULTS"
if QUICK:
    EPOCHS, HEAD_WARMUP_EPOCHS, BOOTSTRAP_REPS, NUM_WORKERS = 2, 1, 30, 0
    USE_PRETRAINED, SWEEP_EPOCHS, REPEAT_SEEDS = False, 1, [43]
    FAITH_PER_CLASS, CORR_MAX_PER_CLASS, RUN_SWEEPS, LR_SWEEP, OPT_SWEEP, DPI = 2, 10, True, [1e-3, 1e-4], ["sgd"], 80
    DATASETS["HyperKvasir"]["root"] = os.environ["HEAS_HK_ROOT"]; DATASETS["Kvasir-V2"]["root"] = os.environ["HEAS_KV_ROOT"]
    OUTPUT_ROOT = os.environ["HEAS_OUT"]; ZIP_BASE = OUTPUT_ROOT + "_zip"; EXTRA_DATASETS = ["HyperKvasir", "Kvasir-V2"]
os.makedirs(OUTPUT_ROOT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()

# ---------------------------------- tee: everything printed is also saved ----------------------------------
class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for f in self.s:
            try: f.write(x)
            except Exception: pass
    def flush(self):
        for f in self.s:
            try: f.flush()
            except Exception: pass
_LOG = open(os.path.join(OUTPUT_ROOT, "FULL_NOTEBOOK_LOG.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _LOG)

def banner(t, ch="="):
    print("\n" + ch * 100 + f"\n{t}\n" + ch * 100)
def elapsed_h(): return (time.time() - T0) / 3600
def time_left_ok(need_hours=0.9): return (TIME_BUDGET_HOURS - elapsed_h()) > need_hours

banner("HEASNet FINAL ALL-IN-ONE EXPERIMENT")
print("Start:", datetime.datetime.now().isoformat(timespec="seconds"), "| Device:", DEVICE,
      "| GPU:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "none")
print(f"torch {torch.__version__} | python {platform.python_version()} | epochs {EPOCHS} (head {HEAD_WARMUP_EPOCHS}) | batch {BATCH_SIZE} | seed {SEED} | PNG dpi {DPI}")
print("""
The run creates dataset audits, training histories, train/validation/test metrics,
class-wise reports, confidence intervals, confusion matrices, ROC/PR plots,
Grad-CAM and attention explanations, duplicate audits, complexity profiles,
inference benchmarks, optional ablations and baselines, cross-dataset tables,
and a ZIP of generated experiment outputs.
""")

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def savefig(fig, path, show=True):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100})
C_TRAIN, C_VAL, C_TEST = "#1f6f8b", "#d1603d", "#2a9d58"
