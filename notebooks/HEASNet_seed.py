# =====================================================================================================
# Dataset: HyperKvasir only (12 classes, processed version with fixed train/test folders).
#   evaluated on the same 2,609 test images as the main run.
#
# Seeds: the model was trained three times with the same split and protocol, seeds 42, 43 and 44.
#
# Settings (CONFIG block below): DATASET = "hyperkvasir", PROCESSED = True, USE_AUG_IN_TRAIN = False,
#   RUN = ["main", "overlap"], SEEDS = [42, 43, 44], BACKBONE = "s", WINDOW = 4,
#   40 epochs (5 frozen-backbone + 35 fine-tuning).
# Differences from the main protocol (Section 3.4): Adam optimizer (not AdamW) and label smoothing 0.1 (not 0.05).
#
# Output folder: /kaggle/working/out_hyperkvasir_proc_noaug
#   splits.csv                      split configuration (every image, its label and subset)
#   main_per_seed.csv               seed-specific results
#   preds_HEASNet_s42/43/44.npz     test predictions (class probabilities, labels) per seed
#   test_metrics.json, overlap_check.json, dedup_accuracy.csv
# The console output of the run used for Table 10 is in HEASNet_originals_only_Table10_run_log.txt.
# =====================================================================================================

# %% [markdown]
# # HEASNet originals-only experiment (Kaggle notebook)
# Paste into a Kaggle notebook (GPU on, Internet on).
# Everything is resumable: finished runs are stored in OUT_DIR and skipped on re-run.
#
# MODES (RUN):
#   "main"      HEASNet, SEEDS, test metrics + efficiency       -> test_metrics.json, main_per_seed.csv, test_classification_report.csv
#   "overlap"   perceptual-hash near-duplicate checks            -> overlap_check.json, dedup_accuracy.csv
#
# Protocol: checkpoint chosen on validation accuracy, test set evaluated once per run.
# Data: the processed HyperKvasir dataset (train/ with offline *_aug_* copies, test/ with originals).
#       Split BY ORIGINAL IMAGE: validation and test contain originals only. USE_AUG_IN_TRAIN = False gives
#       the originals-only run (no augmented copies in training).

# %%
import os, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix)

# %% CONFIG ----------------------------------------------------------------
DATASET        = "hyperkvasir"
QUICK            = False     # True = 1-epoch pipeline test (about a few minutes). NOT for the paper. Set False for the real run.
PROCESSED        = True      # your processed dataset (train/test folders with *_aug_* files)
USE_AUG_IN_TRAIN = False     # True: originals + their augmented copies in training; False: originals only
PROC_VAL_FRAC    = 0.15      # validation share of the ORIGINAL training images
DATA_ROOT        = "/kaggle/input/datasets/visheshtanwar26/processed-datatset"
OUT_DIR        = f"/kaggle/working/out_{DATASET}_{('proc_aug' if USE_AUG_IN_TRAIN else 'proc_noaug') if PROCESSED else 'plain'}{'_QUICK' if QUICK else ''}"
RUN            = ["main", "overlap"]

BACKBONE       = "s"      # EfficientNetV2-S
SPLIT_SEED     = 42
SEEDS          = [42, 43, 44]
IMG_SIZE, BATCH_SIZE = 224, 16
LR, WEIGHT_DECAY = 1e-4, 1e-5
HEAD_EPOCHS, FT_EPOCHS = 5, 35                      # total 40
LABEL_SMOOTHING = 0.1
EMBED_DIM, NUM_HEADS = 512, 8
WINDOW         = 4        # 224 px -> 7x7 map, zero-padded to 8x8 -> four 4x4 windows. WINDOW = 7 would be global attention
DROPOUT        = 0.3
NUM_WORKERS    = 0        # images are cached in RAM, workers are not needed
HASH_NEAR      = 4        # Hamming distance (of 64) counted as a near-duplicate
if QUICK:
    HEAD_EPOCHS, FT_EPOCHS, SEEDS = 1, 0, [42]
    print("QUICK MODE: 1 epoch, 1 seed. These numbers are only a pipeline test and must not be used in the paper.")

EXT = {".jpg", ".jpeg", ".png", ".bmp"}
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

# %% DATA ------------------------------------------------------------------
def make_split_processed(root):
    """train/ may contain augmented copies named <id>_aug_<hash>.jpg; test/ holds originals.
    Validation is cut from the ORIGINAL training images (by id); augmented copies follow their original into train only."""
    rows = []
    for part in ("train", "test"):
        base = os.path.join(root, part)
        for dp, _, files in os.walk(base):
            if dp == base: continue
            lab = Path(dp).name
            for f in files:
                if Path(f).suffix.lower() in EXT:
                    stem = Path(f).stem
                    rows.append((os.path.join(dp, f), lab, part, "_aug_" in stem, stem.split("_aug_")[0]))
    df = pd.DataFrame(rows, columns=["path", "label", "part", "is_aug", "orig_id"])
    classes = sorted(df.label.unique()); df["y"] = df.label.map({c: i for i, c in enumerate(classes)})
    trn, te = df[df.part == "train"], df[df.part == "test"]
    orig = trn[~trn.is_aug].drop_duplicates("orig_id")
    shared = set(te.orig_id) & set(orig.orig_id)
    print(f"train: {len(orig)} originals + {int(trn.is_aug.sum())} augmented copies | test: {len(te)} | ids shared by train and test: {len(shared)}")
    if "is_aug" in te and te.is_aug.any(): print("WARNING: test folder contains augmented files:", int(te.is_aug.sum()))
    te = te[~te.orig_id.isin(orig.orig_id)]                       # drop any test image that also exists in train
    o_tr, o_va = train_test_split(orig, test_size=PROC_VAL_FRAC, stratify=orig.y, random_state=SPLIT_SEED)
    tr = trn[trn.orig_id.isin(o_tr.orig_id)]
    if not USE_AUG_IN_TRAIN: tr = tr[~tr.is_aug]
    va = trn[trn.orig_id.isin(o_va.orig_id) & ~trn.is_aug]
    print(f"originals per class in train/:\n{orig.label.value_counts().sort_index().to_string()}")
    print(f"augmented copies in training set: {int(tr.is_aug.sum())} | originals in training set: {int((~tr.is_aug).sum())}")
    return classes, df, tr, va, te

# Images are decoded and resized ONCE and kept in RAM as uint8 (about 150 KB each). Same preprocessing as before
# (resize to 224 x 224, ImageNet normalisation, no augmentation), but an epoch no longer waits for JPEG decoding.
from concurrent.futures import ThreadPoolExecutor
_CACHE = {}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _load(p):
    im = Image.open(p)
    im.draft("RGB", (IMG_SIZE, IMG_SIZE))          # fast JPEG decode at reduced size (no-op for PNG)
    return np.array(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)

def cache_images(paths):
    todo = [p for p in dict.fromkeys(paths) if p not in _CACHE]
    if not todo: return
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for p, a in zip(todo, ex.map(_load, todo)): _CACHE[p] = a
    print(f"cached {len(todo)} images in {time.time() - t0:.0f}s")

class DS(Dataset):
    def __init__(self, d):
        self.p, self.y = d.path.tolist(), d.y.tolist(); cache_images(self.p)
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        x = torch.from_numpy(_CACHE[self.p[i]]).permute(2, 0, 1).float().div_(255)
        return (x - _MEAN) / _STD, self.y[i]

def loader(d, shuffle, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(DS(d), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=True, generator=g if shuffle else None)

# %% MODELS ----------------------------------------------------------------
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__(); self.fc1, self.fc2 = nn.Linear(c, c // r), nn.Linear(c // r, c)
    def forward(self, x):
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(x.mean((2, 3))))))
        return x * s[:, :, None, None]

class WindowAttention(nn.Module):
    """Non-overlapping window self-attention; the map is zero-padded to a multiple of the window size."""
    def __init__(self, dim, heads, win):
        super().__init__(); self.win = win
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True); self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, H, W = x.shape; w = self.win
        ph, pw = (-H) % w, (-W) % w
        if ph or pw: x = nn.functional.pad(x, (0, pw, 0, ph))
        Hp, Wp = H + ph, W + pw
        t = x.permute(0, 2, 3, 1).reshape(B, Hp // w, w, Wp // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)
        t = self.norm(t + self.attn(t, t, t, need_weights=False)[0])
        t = t.reshape(B, Hp // w, Wp // w, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        return t.permute(0, 3, 1, 2)[:, :, :H, :W]

class AxialAttentionH(nn.Module):
    """Attention along the height axis only."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True); self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        t = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        t = self.norm(t + self.attn(t, t, t, need_weights=False)[0])
        return t.reshape(B, W, H, C).permute(0, 3, 2, 1)

def effnet_features():
    if BACKBONE == "m":
        return models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT).features
    return models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT).features

class HEASNet(nn.Module):
    def __init__(self, n_classes, se=True, win=True, ax=True):
        super().__init__()
        self.backbone = effnet_features()
        self.proj = nn.Conv2d(1280, EMBED_DIM, 1)
        self.se = SE(EMBED_DIM) if se else nn.Identity()
        self.win = WindowAttention(EMBED_DIM, NUM_HEADS, WINDOW) if win else nn.Identity()
        self.ax = AxialAttentionH(EMBED_DIM, NUM_HEADS) if ax else nn.Identity()
        self.drop, self.fc = nn.Dropout(DROPOUT), nn.Linear(EMBED_DIM, n_classes)
    def attn_modules(self): return [self.proj, self.se, self.win, self.ax]
    def forward(self, x):
        x = self.ax(self.win(self.se(self.proj(self.backbone(x)))))
        return self.fc(self.drop(x.mean((2, 3))))

# %% METRICS (pure numpy: no torch needed) ---------------------------------
def ece_score(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)

def boot_ci(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [accuracy_score(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def full_metrics(y, probs):
    K = probs.shape[1]; pred = probs.argmax(1)
    p, r, f1, sup = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    wf = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)[2]
    oh = np.eye(K)[y]
    return dict(acc=accuracy_score(y, pred), acc_ci=boot_ci(y, pred), macro_precision=mp, macro_recall=mr, macro_f1=mf,
                weighted_f1=wf, kappa=cohen_kappa_score(y, pred), mcc=matthews_corrcoef(y, pred),
                roc_auc_macro=roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(K))),
                ece=ece_score(probs, y)), (p, r, f1, sup, [roc_auc_score(oh[:, k], probs[:, k]) for k in range(K)])

# %% TRAINING --------------------------------------------------------------
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def run_epoch(model, dl, crit, opt=None, scaler=None):
    train = opt is not None; model.train(train); tl = co = n = 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(x); loss = crit(out, y)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); co += (out.argmax(1) == y).sum().item(); n += len(y)
    return tl / n, co / n

def make_opt(name, groups, lr):
    ps = [dict(params=g["params"], lr=g["lr"]) for g in groups]
    if name == "adam": return torch.optim.Adam(ps, weight_decay=WEIGHT_DECAY)
    raise ValueError(name)

def train_model(model, dl_tr, dl_va, lr=LR, opt_name="adam", epochs=(HEAD_EPOCHS, FT_EPOCHS)):
    """Two-stage training: 5 epochs with the backbone frozen, then end-to-end fine-tuning."""
    model.to(device); crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best, state, hist = -1.0, None, []
    def stage(groups, n_ep, tag):
        nonlocal best, state
        opt = make_opt(opt_name, groups, lr)
        for ep in range(n_ep):
            t_ep = time.time()
            trl, tra = run_epoch(model, dl_tr, crit, opt, scaler); vl, va = run_epoch(model, dl_va, crit)
            print(f"  stage {tag} epoch {ep + 1}/{n_ep}: train {tra:.4f} val {va:.4f} ({time.time() - t_ep:.0f}s)")
            hist.append(dict(stage=tag, epoch=len(hist) + 1, train_loss=trl, train_acc=tra, val_loss=vl, val_acc=va))
            if va > best: best, state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for p in model.parameters(): p.requires_grad = False
    head = [p for m in model.attn_modules() for p in m.parameters()] + list(model.fc.parameters())
    for p in head: p.requires_grad = True
    stage([dict(params=head, lr=lr)], epochs[0], 1)                       # stage 1: new layers only
    for p in model.parameters(): p.requires_grad = True
    bb = list(model.backbone.parameters()); ids = {id(p) for p in bb}
    stage([dict(params=bb, lr=lr * 0.1), dict(params=[p for p in model.parameters() if id(p) not in ids], lr=lr)], epochs[1], 2)
    model.load_state_dict(state)
    return best, pd.DataFrame(hist)

@torch.no_grad()
def predict(model, dl):
    model.eval(); P, Y = [], []
    for x, y in dl:
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            o = model(x.to(device))
        P.append(torch.softmax(o.float(), 1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)

def efficiency(model):
    out = dict(params_M=sum(p.numel() for p in model.parameters()) / 1e6, gmacs=None, latency_ms=None, peak_mem_MB=None)
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device); model.eval()
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad(): model(x)
        out["gmacs"] = fc.get_total_flops() / 2 / 1e9
    except Exception as e: print("GMACs not computed:", e)
    if device == "cuda":
        with torch.no_grad():
            for _ in range(10): model(x)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
            for _ in range(100): model(x)
            torch.cuda.synchronize()
        out["latency_ms"] = (time.time() - t0) / 100 * 1000; out["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    return out

def one_run(name, builder, seed, dls, **kw):
    """Train, pick best-val checkpoint, evaluate val and test, save predictions."""
    f = f"{OUT_DIR}/preds_{name.replace('/', '_')}_s{seed}.npz"
    if os.path.exists(f):
        z = np.load(f); return dict(val_acc=float(z["val_acc"]), probs=z["probs"], y=z["y"], cached=True)
    seed_all(seed); model = builder(); dl_tr, dl_va, dl_te = dls
    best, hist = train_model(model, dl_tr, dl_va, **kw)
    probs, y = predict(model, dl_te)
    np.savez(f, probs=probs, y=y, val_acc=best); hist.to_csv(f.replace("preds_", "history_").replace(".npz", ".csv"), index=False)
    if name == "HEASNet":      # best-validation weights, accuracies in the file name
        torch.save(model.state_dict(), f.replace("preds_", "model_").replace(".npz", f"_val{100 * best:.2f}_test{100 * accuracy_score(y, probs.argmax(1)):.2f}.pt"))
    res = dict(val_acc=best, probs=probs, y=y, cached=False, model=model)
    print(f"[{name} seed {seed}] val {best:.4f} test {accuracy_score(y, probs.argmax(1)):.4f}")
    return res

# %% SETUP ----------------------------------------------------------------
classes, df, tr, va, te = make_split_processed(DATA_ROOT)
K = len(classes)
print("device:", device)
print(f"{DATASET}: {len(df)} images, {K} classes | train {len(tr)} val {len(va)} test {len(te)} | backbone EfficientNetV2-{BACKBONE.upper()}")
pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]).to_csv(f"{OUT_DIR}/splits.csv", index=False)
pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]).pivot_table(
    index="label", columns="split", values="y", aggfunc="count").to_csv(f"{OUT_DIR}/class_split_counts.csv")
dls = (loader(tr, True), loader(va, False), loader(te, False))

# %% MAIN -------------------------------------------------------------------
if "main" in RUN:
    accs, rows = [], []
    for s in SEEDS:
        r = one_run("HEASNet", lambda: HEASNet(K), s, dls); m, pc = full_metrics(r["y"], r["probs"]); accs.append(m["acc"])
        rows.append(dict(seed=s, val_acc=r["val_acc"], **{k: v for k, v in m.items() if k != "acc_ci"}, acc_ci_lo=m["acc_ci"][0], acc_ci_hi=m["acc_ci"][1]))
        if s == SEEDS[0]:
            pd.DataFrame(dict(**{"class": classes}, precision=pc[0], recall=pc[1], f1=pc[2], support=pc[3], roc_auc=pc[4])).to_csv(
                f"{OUT_DIR}/test_classification_report.csv", index=False)
            pd.DataFrame(confusion_matrix(r["y"], r["probs"].argmax(1)), index=classes, columns=classes).to_csv(f"{OUT_DIR}/confusion_matrix.csv")
            eff = efficiency(r["model"] if "model" in r else HEASNet(K).to(device))
    per_seed = pd.DataFrame(rows); per_seed.to_csv(f"{OUT_DIR}/main_per_seed.csv", index=False)
    summary = dict(dataset=DATASET, n_classes=K, n_train=len(tr), n_val=len(va), n_test=len(te), backbone=f"EfficientNetV2-{BACKBONE.upper()}",
                   seeds=SEEDS, test_accuracy_mean=float(np.mean(accs)), test_accuracy_sd=float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                   first_seed=rows[0], efficiency=eff, augmentation=("offline copies in training only" if (PROCESSED and USE_AUG_IN_TRAIN) else "none"), epochs_total=HEAD_EPOCHS + FT_EPOCHS)
    json.dump(summary, open(f"{OUT_DIR}/test_metrics.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))

# %% OVERLAP / NEAR-DUPLICATE CHECKS --------------------------------------------
_POP = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)
def popcount64(x): return _POP[x.view(np.uint16).reshape(-1, 4)].sum(1)

def dhash(path, size=8):
    im = Image.open(path); im.draft("L", (4 * size, 4 * size))
    im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16); bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)

def min_dist(queries, ref):
    q, r = np.array(queries, dtype=np.uint64), np.array(ref, dtype=np.uint64)
    return np.array([popcount64(r ^ h).min() for h in q])

if "overlap" in RUN:
    h = {s: [dhash(p) for p in part.path] for s, part in (("train", tr), ("val", va), ("test", te))}
    out = {"dataset": DATASET, "hash": "dHash 64-bit", "near_threshold_bits": HASH_NEAR}
    for s in ("val", "test"):
        d_ = min_dist(h[s], h["train"])
        out[f"{s}_exact_duplicates_in_train"] = int((d_ == 0).sum()); out[f"{s}_near_duplicates_in_train"] = int((d_ <= HASH_NEAR).sum()); out[f"{s}_n"] = len(h[s])
    json.dump(out, open(f"{OUT_DIR}/overlap_check.json", "w"), indent=2); print(json.dumps(out, indent=2))
    # accuracy of the main model on test images WITHOUT a near-duplicate in the training set
    dt = min_dist(h["test"], h["train"]); keep = dt > HASH_NEAR; rows_dd = []
    for sd in SEEDS:
        f_ = f"{OUT_DIR}/preds_HEASNet_s{sd}.npz"
        if os.path.exists(f_):
            z = np.load(f_); ok = z["probs"].argmax(1) == z["y"]
            rows_dd.append(dict(seed=sd, n_all=len(ok), acc_all=100 * ok.mean(), n_no_dup=int(keep.sum()), acc_no_dup=100 * ok[keep].mean(),
                                n_dup=int((~keep).sum()), acc_dup=100 * ok[~keep].mean()))
    if rows_dd: pd.DataFrame(rows_dd).to_csv(f"{OUT_DIR}/dedup_accuracy.csv", index=False)

# %% FIGURES (seed SEEDS[0]) -> PNG files in OUT_DIR ---------------------------------
def make_figures():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    s0 = SEEDS[0]; hf = f"{OUT_DIR}/history_HEASNet_s{s0}.csv"; pf = f"{OUT_DIR}/preds_HEASNet_s{s0}.npz"
    if os.path.exists(hf):
        hh = pd.read_csv(hf); x = np.arange(1, len(hh) + 1)
        for key, lab, fn in (("loss", "Loss", "fig2_loss.png"), ("acc", "Accuracy", "fig3_accuracy.png")):
            fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
            ax.plot(x, hh[f"train_{key}"], label="Training"); ax.plot(x, hh[f"val_{key}"], label="Validation")
            ax.axvline(HEAD_EPOCHS + 0.5, color="gray", ls=":"); ax.set_xlabel("Epoch"); ax.set_ylabel(lab); ax.grid(alpha=.25); ax.legend()
            fig.tight_layout(); fig.savefig(f"{OUT_DIR}/{fn}"); plt.close(fig)
    if os.path.exists(pf):
        z = np.load(pf); P, Y = z["probs"], z["y"]
        cm = confusion_matrix(Y, P.argmax(1)); fig, ax = plt.subplots(figsize=(7.6, 6), dpi=200); im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(classes, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(classes, fontsize=7)
        for i in range(K):
            for j in range(K):
                if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig4_confusion.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.4, 5), dpi=200)
        for c in range(K):
            fpr, tpr, _ = roc_curve((Y == c).astype(int), P[:, c]); ax.plot(fpr, tpr, lw=1.2, label=f"{classes[c]} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=6, loc="lower right")
        fig.tight_layout(); fig.savefig(f"{OUT_DIR}/fig5_roc.png"); plt.close(fig)
        print("figures written to", OUT_DIR)
if "main" in RUN:
    try: make_figures()
    except Exception as e: print("figures skipped:", e)

# %% ONE BLOCK TO COPY BACK ---------------------------------------------------
def print_results_block():
    """Prints every result file in one block. Copy everything between the two ##### lines and send it back."""
    print("#" * 20, "COPY FROM HERE", "#" * 20)
    if QUICK: print("!!! QUICK MODE (1 epoch): do NOT use these numbers in the paper !!!")
    print(f"dataset={DATASET} processed={PROCESSED} aug_in_train={USE_AUG_IN_TRAIN} backbone=EfficientNetV2-{BACKBONE.upper()} window={WINDOW} seeds={SEEDS}")
    print(f"n_classes={K} n_train={len(tr)} n_val={len(va)} n_test={len(te)}")
    print("classes:", classes)
    for name in ["class_split_counts.csv", "test_metrics.json", "main_per_seed.csv", "test_classification_report.csv", "confusion_matrix.csv",
                 "overlap_check.json", "dedup_accuracy.csv"]:
        f = f"{OUT_DIR}/{name}"
        if os.path.exists(f):
            print(f"\n=== {name} ==="); print(open(f).read().strip())
    h = f"{OUT_DIR}/history_HEASNet_s{SEEDS[0]}.csv"
    if os.path.exists(h):
        hh = pd.read_csv(h); b = hh.loc[hh.val_acc.idxmax()]
        print(f"\n=== training summary (seed {SEEDS[0]}) ===")
        print(f"best val acc {b.val_acc:.4f} at epoch {int(b.epoch)}; final epoch train acc {hh.train_acc.iloc[-1]:.4f}, val acc {hh.val_acc.iloc[-1]:.4f}, train loss {hh.train_loss.iloc[-1]:.4f}, val loss {hh.val_loss.iloc[-1]:.4f}")
        print("history.csv (every 5th epoch):"); print(hh.iloc[4::5].round(4).to_csv(index=False).strip())
    print("#" * 20, "COPY TO HERE", "#" * 20)

print("Done. Results in", OUT_DIR)
print_results_block()


# =====================================================================================================
# KAGGLE OUTPUT of this script (Tesla T4; RUN = ["main", "overlap"]; seeds 42, 43, 44)
# Output folder: /kaggle/working/out_hyperkvasir_proc_noaug
# =====================================================================================================
# train: 6647 originals + 553 augmented copies | test: 2609 | ids shared by train and test: 0
# originals per class in train/:
# label
# bbps-0-1                  484
# bbps-2-3                  600
# cecum                     600
# dyed-lifted-polyps        600
# dyed-resection-margins    600
# esophagitis               497
# polyps                    600
# pylorus                   600
# retroflex-rectum          293
# retroflex-stomach         573
# ulcerative-colitis        600
# z-line                    600
# augmented copies in training set: 0 | originals in training set: 5649
# device: cuda
# hyperkvasir: 9809 images, 12 classes | train 5649 val 998 test 2609 | backbone EfficientNetV2-S
# cached 5649 images in 16s
# cached 998 images in 3s
# cached 2609 images in 8s
#   stage 1 epoch 1/5: train 0.6985 val 0.8427 (29s)
#   stage 1 epoch 2/5: train 0.8136 val 0.8677 (19s)
#   stage 1 epoch 3/5: train 0.8368 val 0.8707 (19s)
#   stage 1 epoch 4/5: train 0.8596 val 0.8778 (19s)
#   stage 1 epoch 5/5: train 0.8621 val 0.8918 (19s)
#   stage 2 epoch 1/35: train 0.8890 val 0.8978 (49s)
#   stage 2 epoch 2/35: train 0.9246 val 0.9178 (48s)
#   stage 2 epoch 3/35: train 0.9419 val 0.9188 (48s)
#   stage 2 epoch 4/35: train 0.9536 val 0.9259 (48s)
#   stage 2 epoch 5/35: train 0.9697 val 0.9309 (48s)
#   stage 2 epoch 6/35: train 0.9780 val 0.9299 (48s)
#   stage 2 epoch 7/35: train 0.9823 val 0.9319 (48s)
#   stage 2 epoch 8/35: train 0.9883 val 0.9279 (48s)
#   stage 2 epoch 9/35: train 0.9903 val 0.9329 (48s)
#   stage 2 epoch 10/35: train 0.9911 val 0.9409 (48s)
#   stage 2 epoch 11/35: train 0.9942 val 0.9329 (48s)
#   stage 2 epoch 12/35: train 0.9927 val 0.9339 (48s)
#   stage 2 epoch 13/35: train 0.9952 val 0.9389 (48s)
#   stage 2 epoch 14/35: train 0.9958 val 0.9429 (48s)
#   stage 2 epoch 15/35: train 0.9947 val 0.9439 (48s)
#   stage 2 epoch 16/35: train 0.9963 val 0.9459 (48s)
#   stage 2 epoch 17/35: train 0.9982 val 0.9429 (48s)
#   stage 2 epoch 18/35: train 0.9966 val 0.9499 (48s)
#   stage 2 epoch 19/35: train 0.9970 val 0.9409 (48s)
#   stage 2 epoch 20/35: train 0.9965 val 0.9419 (48s)
#   stage 2 epoch 21/35: train 0.9968 val 0.9399 (48s)
#   stage 2 epoch 22/35: train 0.9966 val 0.9439 (48s)
#   stage 2 epoch 23/35: train 0.9982 val 0.9429 (48s)
#   stage 2 epoch 24/35: train 0.9977 val 0.9529 (48s)
#   stage 2 epoch 25/35: train 0.9991 val 0.9439 (48s)
#   stage 2 epoch 26/35: train 0.9986 val 0.9429 (48s)
#   stage 2 epoch 27/35: train 0.9984 val 0.9439 (48s)
#   stage 2 epoch 28/35: train 0.9984 val 0.9479 (48s)
#   stage 2 epoch 29/35: train 0.9981 val 0.9509 (48s)
#   stage 2 epoch 30/35: train 0.9986 val 0.9519 (48s)
#   stage 2 epoch 31/35: train 0.9975 val 0.9489 (48s)
#   stage 2 epoch 32/35: train 0.9984 val 0.9439 (48s)
#   stage 2 epoch 33/35: train 0.9981 val 0.9459 (48s)
#   stage 2 epoch 34/35: train 0.9986 val 0.9469 (48s)
#   stage 2 epoch 35/35: train 0.9979 val 0.9499 (48s)
# [HEASNet seed 42] val 0.9529 test 0.9475
#   stage 1 epoch 1/5: train 0.6948 val 0.8186 (19s)
#   stage 1 epoch 2/5: train 0.8242 val 0.8357 (19s)
#   stage 1 epoch 3/5: train 0.8458 val 0.8587 (19s)
#   stage 1 epoch 4/5: train 0.8619 val 0.8808 (19s)
#   stage 1 epoch 5/5: train 0.8671 val 0.8707 (19s)
#   stage 2 epoch 1/35: train 0.8846 val 0.9058 (48s)
#   stage 2 epoch 2/35: train 0.9177 val 0.9178 (48s)
#   stage 2 epoch 3/35: train 0.9448 val 0.9228 (48s)
#   stage 2 epoch 4/35: train 0.9616 val 0.9319 (48s)
#   stage 2 epoch 5/35: train 0.9646 val 0.9309 (48s)
#   stage 2 epoch 6/35: train 0.9754 val 0.9299 (49s)
#   stage 2 epoch 7/35: train 0.9819 val 0.9379 (49s)
#   stage 2 epoch 8/35: train 0.9855 val 0.9289 (49s)
#   stage 2 epoch 9/35: train 0.9890 val 0.9299 (49s)
#   stage 2 epoch 10/35: train 0.9913 val 0.9389 (49s)
#   stage 2 epoch 11/35: train 0.9906 val 0.9449 (49s)
#   stage 2 epoch 12/35: train 0.9945 val 0.9439 (49s)
#   stage 2 epoch 13/35: train 0.9958 val 0.9379 (49s)
#   stage 2 epoch 14/35: train 0.9945 val 0.9439 (49s)
#   stage 2 epoch 15/35: train 0.9954 val 0.9349 (49s)
#   stage 2 epoch 16/35: train 0.9938 val 0.9389 (49s)
#   stage 2 epoch 17/35: train 0.9973 val 0.9429 (50s)
#   stage 2 epoch 18/35: train 0.9977 val 0.9409 (50s)
#   stage 2 epoch 19/35: train 0.9984 val 0.9389 (49s)
#   stage 2 epoch 20/35: train 0.9973 val 0.9269 (49s)
#   stage 2 epoch 21/35: train 0.9972 val 0.9389 (49s)
#   stage 2 epoch 22/35: train 0.9956 val 0.9449 (49s)
#   stage 2 epoch 23/35: train 0.9963 val 0.9389 (49s)
#   stage 2 epoch 24/35: train 0.9968 val 0.9489 (49s)
#   stage 2 epoch 25/35: train 0.9968 val 0.9459 (49s)
#   stage 2 epoch 26/35: train 0.9973 val 0.9459 (49s)
#   stage 2 epoch 27/35: train 0.9975 val 0.9459 (49s)
#   stage 2 epoch 28/35: train 0.9981 val 0.9529 (49s)
#   stage 2 epoch 29/35: train 0.9988 val 0.9489 (49s)
#   stage 2 epoch 30/35: train 0.9989 val 0.9439 (49s)
#   stage 2 epoch 31/35: train 0.9984 val 0.9429 (49s)
#   stage 2 epoch 32/35: train 0.9993 val 0.9449 (49s)
#   stage 2 epoch 33/35: train 0.9977 val 0.9519 (49s)
#   stage 2 epoch 34/35: train 0.9981 val 0.9449 (49s)
#   stage 2 epoch 35/35: train 0.9982 val 0.9439 (49s)
# [HEASNet seed 43] val 0.9529 test 0.9498
#   stage 1 epoch 1/5: train 0.6989 val 0.8477 (20s)
#   stage 1 epoch 2/5: train 0.8237 val 0.8667 (20s)
#   stage 1 epoch 3/5: train 0.8449 val 0.8778 (20s)
#   stage 1 epoch 4/5: train 0.8540 val 0.8808 (20s)
#   stage 1 epoch 5/5: train 0.8662 val 0.8818 (20s)
#   stage 2 epoch 1/35: train 0.8858 val 0.9048 (50s)
#   stage 2 epoch 2/35: train 0.9228 val 0.9238 (49s)
#   stage 2 epoch 3/35: train 0.9495 val 0.9108 (49s)
#   stage 2 epoch 4/35: train 0.9566 val 0.9279 (48s)
#   stage 2 epoch 5/35: train 0.9678 val 0.9419 (48s)
#   stage 2 epoch 6/35: train 0.9747 val 0.9459 (48s)
#   stage 2 epoch 7/35: train 0.9811 val 0.9449 (48s)
#   stage 2 epoch 8/35: train 0.9876 val 0.9339 (50s)
#   stage 2 epoch 9/35: train 0.9892 val 0.9359 (50s)
#   stage 2 epoch 10/35: train 0.9899 val 0.9459 (48s)
#   stage 2 epoch 11/35: train 0.9929 val 0.9399 (49s)
#   stage 2 epoch 12/35: train 0.9938 val 0.9449 (49s)
#   stage 2 epoch 13/35: train 0.9945 val 0.9449 (48s)
#   stage 2 epoch 14/35: train 0.9959 val 0.9439 (48s)
#   stage 2 epoch 15/35: train 0.9947 val 0.9509 (48s)
#   stage 2 epoch 16/35: train 0.9961 val 0.9479 (48s)
#   stage 2 epoch 17/35: train 0.9958 val 0.9479 (48s)
#   stage 2 epoch 18/35: train 0.9963 val 0.9469 (48s)
#   stage 2 epoch 19/35: train 0.9970 val 0.9459 (48s)
#   stage 2 epoch 20/35: train 0.9979 val 0.9439 (48s)
#   stage 2 epoch 21/35: train 0.9975 val 0.9469 (48s)
#   stage 2 epoch 22/35: train 0.9963 val 0.9439 (48s)
#   stage 2 epoch 23/35: train 0.9952 val 0.9379 (48s)
#   stage 2 epoch 24/35: train 0.9972 val 0.9489 (48s)
#   stage 2 epoch 25/35: train 0.9981 val 0.9609 (48s)
#   stage 2 epoch 26/35: train 0.9986 val 0.9489 (48s)
#   stage 2 epoch 27/35: train 0.9982 val 0.9499 (48s)
#   stage 2 epoch 28/35: train 0.9972 val 0.9509 (48s)
#   stage 2 epoch 29/35: train 0.9981 val 0.9579 (48s)
#   stage 2 epoch 30/35: train 0.9986 val 0.9599 (48s)
#   stage 2 epoch 31/35: train 0.9970 val 0.9569 (48s)
#   stage 2 epoch 32/35: train 0.9981 val 0.9509 (48s)
#   stage 2 epoch 33/35: train 0.9972 val 0.9459 (49s)
#   stage 2 epoch 34/35: train 0.9977 val 0.9539 (51s)
#   stage 2 epoch 35/35: train 0.9993 val 0.9569 (51s)
# [HEASNet seed 44] val 0.9609 test 0.9506
# {
#   "dataset": "hyperkvasir",
#   "n_classes": 12,
#   "n_train": 5649,
#   "n_val": 998,
#   "n_test": 2609,
#   "backbone": "EfficientNetV2-S",
#   "seeds": [
#     42,
#     43,
#     44
#   ],
#   "test_accuracy_mean": 0.9492781397725821,
#   "test_accuracy_sd": 0.0015957577611852352,
#   "first_seed": {
#     "seed": 42,
#     "val_acc": 0.9529058116232465,
#     "acc": 0.947489459563051,
#     "macro_precision": 0.9431315350133446,
#     "macro_recall": 0.9437984306946606,
#     "macro_f1": 0.9432817537250822,
#     "weighted_f1": 0.9474361468846599,
#     "kappa": 0.9424248652650387,
#     "mcc": 0.9424634254252872,
#     "roc_auc_macro": 0.988951968577016,
#     "ece": 0.06268434909030643,
#     "acc_ci_lo": 0.9386738213875048,
#     "acc_ci_hi": 0.955538520505941
#   },
#   "efficiency": {
#     "params_M": 22.976124,
#     "gmacs": 2.880767424,
#     "latency_ms": 18.63504409790039,
#     "peak_mem_MB": 204.9560546875
#   },
#   "augmentation": "none",
#   "epochs_total": 40
# }
# {
#   "dataset": "hyperkvasir",
#   "hash": "dHash 64-bit",
#   "near_threshold_bits": 4,
#   "val_exact_duplicates_in_train": 21,
#   "val_near_duplicates_in_train": 144,
#   "val_n": 998,
#   "test_exact_duplicates_in_train": 74,
#   "test_near_duplicates_in_train": 365,
#   "test_n": 2609
# }
# figures written to /kaggle/working/out_hyperkvasir_proc_noaug
# Done. Results in /kaggle/working/out_hyperkvasir_proc_noaug
# #################### COPY FROM HERE ####################
# dataset=hyperkvasir processed=True aug_in_train=False backbone=EfficientNetV2-S window=4 seeds=[42, 43, 44]
# n_classes=12 n_train=5649 n_val=998 n_test=2609
# classes: ['bbps-0-1', 'bbps-2-3', 'cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'polyps', 'pylorus', 'retroflex-rectum', 'retroflex-stomach', 'ulcerative-colitis', 'z-line']
#
# === class_split_counts.csv ===
# label,test,train,val
# bbps-0-1,162,411,73
# bbps-2-3,287,510,90
# cecum,253,510,90
# dyed-lifted-polyps,251,510,90
# dyed-resection-margins,248,510,90
# esophagitis,166,422,75
# polyps,257,510,90
# pylorus,250,510,90
# retroflex-rectum,98,249,44
# retroflex-stomach,191,487,86
# ulcerative-colitis,213,510,90
# z-line,233,510,90
#
# === test_metrics.json ===
# {
#   "dataset": "hyperkvasir",
#   "n_classes": 12,
#   "n_train": 5649,
#   "n_val": 998,
#   "n_test": 2609,
#   "backbone": "EfficientNetV2-S",
#   "seeds": [
#     42,
#     43,
#     44
#   ],
#   "test_accuracy_mean": 0.9492781397725821,
#   "test_accuracy_sd": 0.0015957577611852352,
#   "first_seed": {
#     "seed": 42,
#     "val_acc": 0.9529058116232465,
#     "acc": 0.947489459563051,
#     "macro_precision": 0.9431315350133446,
#     "macro_recall": 0.9437984306946606,
#     "macro_f1": 0.9432817537250822,
#     "weighted_f1": 0.9474361468846599,
#     "kappa": 0.9424248652650387,
#     "mcc": 0.9424634254252872,
#     "roc_auc_macro": 0.988951968577016,
#     "ece": 0.06268434909030643,
#     "acc_ci_lo": 0.9386738213875048,
#     "acc_ci_hi": 0.955538520505941
#   },
#   "efficiency": {
#     "params_M": 22.976124,
#     "gmacs": 2.880767424,
#     "latency_ms": 18.63504409790039,
#     "peak_mem_MB": 204.9560546875
#   },
#   "augmentation": "none",
#   "epochs_total": 40
# }
#
# === main_per_seed.csv ===
# seed,val_acc,acc,macro_precision,macro_recall,macro_f1,weighted_f1,kappa,mcc,roc_auc_macro,ece,acc_ci_lo,acc_ci_hi
# 42,0.9529058116232465,0.947489459563051,0.9431315350133446,0.9437984306946606,0.9432817537250822,0.9474361468846599,0.9424248652650387,0.9424634254252872,0.988951968577016,0.06268434909030643,0.9386738213875048,0.955538520505941
# 43,0.9529058116232465,0.9497891912610196,0.9496140020836542,0.9431394577549749,0.9446456203304269,0.9489941301311976,0.944921083081128,0.9451749212798397,0.986071921224121,0.06007053319546123,0.9405902644691453,0.9574549635875814
# 44,0.9609218436873748,0.9505557684936757,0.9472461642599527,0.9460235685976092,0.945858563451297,0.9502004994313119,0.945776549329483,0.9458896609927503,0.9892520799832473,0.06191503673780458,0.9425067075507857,0.9582215408202377
#
# === test_classification_report.csv ===
# class,precision,recall,f1,support,roc_auc
# bbps-0-1,0.9874213836477987,0.9691358024691358,0.9781931464174455,162,0.9796677211198395
# bbps-2-3,0.9930313588850174,0.9930313588850174,0.9930313588850174,287,0.9960550048468371
# cecum,0.9841897233201581,0.9841897233201581,0.9841897233201581,253,0.9996040720186288
# dyed-lifted-polyps,0.9620253164556962,0.9083665338645418,0.9344262295081968,251,0.9884093819801373
# dyed-resection-margins,0.9125475285171103,0.967741935483871,0.9393346379647749,248,0.9972862100531487
# esophagitis,0.7668711656441718,0.7530120481927711,0.7598784194528876,166,0.940824287736291
# polyps,1.0,0.9766536964980544,0.9881889763779528,257,0.9977583445829693
# pylorus,0.9880952380952381,0.996,0.9920318725099602,250,0.9998117846545147
# retroflex-rectum,0.9405940594059405,0.9693877551020408,0.9547738693467337,98,0.9992075683319923
# retroflex-stomach,0.9895287958115183,0.9895287958115183,0.9895287958115183,191,0.9954442899891305
# ulcerative-colitis,0.967741935483871,0.9859154929577465,0.9767441860465116,213,0.9989850063094202
# z-line,0.825531914893617,0.8326180257510729,0.8290598290598291,233,0.9743699513012818
#
# === confusion_matrix.csv ===
# ,bbps-0-1,bbps-2-3,cecum,dyed-lifted-polyps,dyed-resection-margins,esophagitis,polyps,pylorus,retroflex-rectum,retroflex-stomach,ulcerative-colitis,z-line
# bbps-0-1,157,2,1,0,0,0,0,0,0,0,2,0
# bbps-2-3,2,285,0,0,0,0,0,0,0,0,0,0
# cecum,0,0,249,0,0,0,0,0,0,0,4,0
# dyed-lifted-polyps,0,0,0,228,23,0,0,0,0,0,0,0
# dyed-resection-margins,0,0,0,8,240,0,0,0,0,0,0,0
# esophagitis,0,0,0,0,0,125,0,1,0,0,0,40
# polyps,0,0,1,0,0,0,251,1,3,0,0,1
# pylorus,0,0,0,0,0,1,0,249,0,0,0,0
# retroflex-rectum,0,0,0,1,0,0,0,0,95,1,1,0
# retroflex-stomach,0,0,0,0,0,0,0,0,2,189,0,0
# ulcerative-colitis,0,0,2,0,0,0,0,0,1,0,210,0
# z-line,0,0,0,0,0,37,0,1,0,1,0,194
#
# === overlap_check.json ===
# {
#   "dataset": "hyperkvasir",
#   "hash": "dHash 64-bit",
#   "near_threshold_bits": 4,
#   "val_exact_duplicates_in_train": 21,
#   "val_near_duplicates_in_train": 144,
#   "val_n": 998,
#   "test_exact_duplicates_in_train": 74,
#   "test_near_duplicates_in_train": 365,
#   "test_n": 2609
# }
#
# === dedup_accuracy.csv ===
# seed,n_all,acc_all,n_no_dup,acc_no_dup,n_dup,acc_dup
# 42,2609,94.74894595630509,2244,94.07308377896612,365,98.9041095890411
# 43,2609,94.97891912610196,2244,94.34046345811052,365,98.9041095890411
# 44,2609,95.05557684936757,2244,94.42959001782532,365,98.9041095890411
#
# === training summary (seed 42) ===
# best val acc 0.9529 at epoch 29; final epoch train acc 0.9979, val acc 0.9499, train loss 0.5381, val loss 0.6648
# history.csv (every 5th epoch):
# stage,epoch,train_loss,train_acc,val_loss,val_acc
# 1,5,0.9095,0.8621,0.8554,0.8918
# 2,10,0.6594,0.9697,0.719,0.9309
# 2,15,0.5903,0.9911,0.7046,0.9409
# 2,20,0.5648,0.9947,0.6877,0.9439
# 2,25,0.5524,0.9965,0.6922,0.9419
# 2,30,0.5409,0.9991,0.6679,0.9439
# 2,35,0.5384,0.9986,0.6643,0.9519
# 2,40,0.5381,0.9979,0.6648,0.9499
# #################### COPY TO HERE ####################
# =====================================================================================================
