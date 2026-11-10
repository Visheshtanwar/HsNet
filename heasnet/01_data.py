"""HEASNet component: Dataset discovery, integrity checks, splits and data loaders.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== DATA ========================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
def normalize_class_name(n): return n.lower().strip().replace(" ", "-").replace("_", "-")
def is_aug_path(p): return "_aug_" in Path(p).stem

def collect_class_folder_images(split_root, split_name):
    recs = []
    for p in Path(split_root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(split_root)
            if len(rel.parts) < 2: continue
            recs.append((str(p), normalize_class_name(rel.parts[0]), split_name))
    return recs

def discover_hyperkvasir(root):
    root = Path(root); tr, te = root / "train", root / "test"
    if not tr.exists() or not te.exists(): raise FileNotFoundError(f"Expected {tr} and {te}")
    df = pd.DataFrame(collect_class_folder_images(tr, "train") + collect_class_folder_images(te, "test"),
                      columns=["path", "label", "source_split"]).drop_duplicates("path")
    if df.empty: raise RuntimeError("No HyperKvasir images found.")
    print(f"HyperKvasir: {len(df):,} images | {df.label.nunique()} classes")
    print("Class counts by provided split:"); print(pd.crosstab(df.label, df.source_split).to_string())
    return df.reset_index(drop=True)

def discover_kvasir_v2(root):
    recs = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(root)
            if len(rel.parts) < 2: continue
            recs.append((str(p), normalize_class_name(rel.parts[0])))
    df = pd.DataFrame(recs, columns=["path", "label"]).drop_duplicates("path")
    if df.empty: raise RuntimeError("No Kvasir-V2 images found.")
    print(f"Kvasir-V2: {len(df):,} images | {df.label.nunique()} classes")
    print("Per-class image counts:"); print(df.label.value_counts().sort_index().to_string())
    return df.reset_index(drop=True)

def create_kvasir_splits(df):
    df = df.copy(); classes = sorted(df.label.unique()); m = {c: i for i, c in enumerate(classes)}
    df["target"] = df.label.map(m).astype(int)
    tv, te = train_test_split(df, test_size=TEST_FRAC, stratify=df.target, random_state=SEED)
    tr, va = train_test_split(tv, test_size=VAL_FRAC / (1 - TEST_FRAC), stratify=tv.target, random_state=SEED)
    return classes, tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

def create_hyperkvasir_splits(df):
    """Supplied test set preserved; train folder split into train/val with an original image and its *_aug_* copies kept together."""
    df = df.copy(); classes = sorted(df.label.unique()); m = {c: i for i, c in enumerate(classes)}
    df["target"] = df.label.map(m).astype(int)
    ot = df[df.source_split == "train"].copy(); test_df = df[df.source_split == "test"].copy()
    ot["group_id"] = ot["path"].map(lambda p: re.sub(r"_aug_.*$", "", Path(p).stem))
    best, best_score = None, float("inf")
    for sd in range(SEED, SEED + 100):
        tri, vai = next(GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=sd).split(ot, ot.target, groups=ot.group_id))
        ctr, cva = ot.iloc[tri].copy(), ot.iloc[vai].copy()
        if set(ctr.label) != set(classes) or set(cva.label) != set(classes): continue
        sc = abs(len(cva) / len(ot) - VAL_FRAC)
        if sc < best_score: best_score, best = sc, (ctr, cva)
    if best is None: raise RuntimeError("Could not create a group-aware validation split containing every class.")
    tr, va = best
    if set(tr.group_id) & set(va.group_id): raise RuntimeError("Augmentation-group leakage detected.")
    return (classes, tr.drop(columns=["group_id"]).reset_index(drop=True),
            va.drop(columns=["group_id"]).reset_index(drop=True), test_df.reset_index(drop=True))

def verify_images(df, name, out_dir):
    bad = []
    for r in df.itertuples(index=False):
        try:
            with Image.open(r.path) as im: im.verify()
        except Exception as e: bad.append({"path": r.path, "label": r.label, "error": repr(e)})
    pd.DataFrame(bad, columns=["path", "label", "error"]).to_csv(os.path.join(out_dir, "corrupt_or_unreadable_images.csv"), index=False)
    if bad:
        df = df[~df.path.isin({b["path"] for b in bad})].copy(); print(f"Removed {len(bad)} corrupt/unreadable files from {name}.")
    else: print("Image integrity check: all discovered files are readable.")
    return df.reset_index(drop=True), len(bad)

def build_split_audit(classes, tr, va, te, out_dir, ds_name):
    fr = {"train": tr, "validation": va, "test": te}
    ct = pd.DataFrame(index=classes)
    for k, f in fr.items():
        c = f.label.value_counts(); ct[k] = [int(c.get(x, 0)) for x in classes]
    ct.index.name = "class"; ct["total"] = ct.sum(axis=1)
    ct.to_csv(os.path.join(out_dir, "CLASS_COUNTS_EVERY_SPLIT.csv"))
    pct = ct[["train", "validation", "test"]].div(ct[["train", "validation", "test"]].sum(0), axis=1) * 100
    pct.to_csv(os.path.join(out_dir, "CLASS_PERCENTAGES_WITHIN_SPLIT.csv"))
    summ = pd.DataFrame([{"split": k, "images": len(f), "percentage_of_dataset": 100 * len(f) / (len(tr) + len(va) + len(te)),
        "number_of_classes_present": f.label.nunique(), "missing_classes": "; ".join([c for c in classes if c not in set(f.label)])} for k, f in fr.items()])
    summ.to_csv(os.path.join(out_dir, "SPLIT_SIZE_AND_CLASS_AUDIT.csv"), index=False)
    ps = {k: set(v.path) for k, v in fr.items()}
    ov = {"train_validation": len(ps["train"] & ps["validation"]), "train_test": len(ps["train"] & ps["test"]), "validation_test": len(ps["validation"] & ps["test"])}
    aud = {"class_names": classes, "number_of_classes": len(classes), "split_sizes": {k: len(v) for k, v in fr.items()}, "overlap_counts": ov,
           "all_classes_in_train": all(c in set(tr.label) for c in classes), "all_classes_in_validation": all(c in set(va.label) for c in classes),
           "all_classes_in_test": all(c in set(te.label) for c in classes)}
    json.dump(aud, open(os.path.join(out_dir, "SPLIT_INTEGRITY_AUDIT.json"), "w"), indent=2)
    print("\nCOMPLETE CLASS COUNTS"); print(ct.to_string()); print("\nSPLIT SIZE / CLASS COVERAGE"); print(summ.to_string(index=False))
    print("\nSPLIT OVERLAP AUDIT:", ov)
    if any(ov.values()): raise RuntimeError("Split leakage detected.")
    # class imbalance analysis
    imb = []
    for k in ["train", "validation", "test", "total"]:
        v = ct[k].values.astype(float)
        imb.append({"subset": k, "min_class": classes[int(v.argmin())], "min_count": int(v.min()), "max_class": classes[int(v.argmax())],
                    "max_count": int(v.max()), "imbalance_ratio_max_over_min": float(v.max() / max(v.min(), 1)),
                    "coefficient_of_variation": float(v.std() / v.mean())})
    imb = pd.DataFrame(imb); imb.to_csv(os.path.join(out_dir, "CLASS_IMBALANCE_ANALYSIS.csv"), index=False)
    print("\nCLASS IMBALANCE ANALYSIS"); print(imb.round(3).to_string(index=False))
    # figure: class distribution
    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 0.8), 5.2)); bot = np.zeros(len(classes))
    for k, col in zip(["train", "validation", "test"], [C_TRAIN, C_VAL, C_TEST]):
        ax.bar(range(len(classes)), ct[k].values, bottom=bot, label=k, color=col); bot += ct[k].values
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=55, ha="right"); ax.set_ylabel("Images")
    ax.set_title(f"{ds_name}: class distribution per split"); ax.legend(); fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "Fig_01_class_distribution.png"))
    return ct, summ, aud, imb

def sample_grid(tr, classes, out_dir, ds_name):
    n = len(classes); cols = 4; rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0), squeeze=False)
    for a in axes.ravel(): a.axis("off")
    for i, c in enumerate(classes):
        sub = tr[(tr.label == c)]; sub = sub[~sub.path.map(is_aug_path)] if (~sub.path.map(is_aug_path)).any() else sub
        p = sub.path.iloc[0]; a = axes[i // cols, i % cols]
        a.imshow(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))); a.set_title(c, fontsize=9)
    fig.suptitle(f"{ds_name}: one example per class (training set)"); fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "Fig_02_sample_images.png"))

train_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
eval_transform = train_transform
class ImageDataset(Dataset):
    def __init__(self, frame, transform): self.paths = frame.path.tolist(); self.targets = frame.target.astype(int).tolist(); self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im: x = im.convert("RGB")
        return self.transform(x), self.targets[i]

def make_loaders(tr, va, te, seed=SEED):
    ds = [ImageDataset(f, train_transform if k == 0 else eval_transform) for k, f in enumerate([tr, va, te])]
    g = torch.Generator().manual_seed(seed)
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS > 0))
    return (DataLoader(ds[0], shuffle=True, generator=g, **common), DataLoader(ds[1], shuffle=False, **common),
            DataLoader(ds[2], shuffle=False, **common), ds)
