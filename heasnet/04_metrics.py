"""HEASNet component: Prediction and metric calculations.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== PREDICTION AND METRICS ========================================
@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Y = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")): lo = model(x)
        P.append(torch.softmax(lo.float(), 1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)

def ece_mce(probs, y, bins=15):
    conf = probs.max(1); ok = (probs.argmax(1) == y).astype(float); e = np.linspace(0, 1, bins + 1); ece = mce = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(ok[m].mean() - conf[m].mean()); ece += m.mean() * gap; mce = max(mce, gap)
    return float(ece), float(mce)

def wilson_interval(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)

def compute_metrics(y, probs, classes):
    K = len(classes); pred = probs.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=np.arange(K), zero_division=0)
    cm = confusion_matrix(y, pred, labels=np.arange(K)); spec = []
    for k in range(K):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp; tn = cm.sum() - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) else np.nan)
    yb = np.eye(K)[y]; aucs, aps = [], []
    for k in range(K):
        try: aucs.append(roc_auc_score(yb[:, k], probs[:, k]))
        except ValueError: aucs.append(np.nan)
        try: aps.append(average_precision_score(yb[:, k], probs[:, k]))
        except ValueError: aps.append(np.nan)
    macro_auc = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else np.nan
    try: micro_auc = float(roc_auc_score(yb.ravel(), probs.ravel()))
    except ValueError: micro_auc = np.nan
    ece, mce = ece_mce(probs, y)
    m = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)), "macro_precision": float(np.mean(pr)), "macro_recall": float(np.mean(rc)),
         "macro_f1": float(np.mean(f1)), "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
         "cohen_kappa": float(cohen_kappa_score(y, pred)), "mcc": float(matthews_corrcoef(y, pred)), "macro_roc_auc_ovr": macro_auc,
         "micro_roc_auc": micro_auc, "log_loss": float(log_loss(y, probs, labels=np.arange(K))),
         "brier_score_probability_MSE": float(np.mean(np.sum((probs - yb) ** 2, 1))), "ece_15_bins": ece, "mce_15_bins": mce,
         "macro_average_precision": float(np.nanmean(aps))}
    lo_hi = [wilson_interval(int(round(r * s)), int(s)) for r, s in zip(rc, sup)]
    pcl = pd.DataFrame({"class": classes, "precision": pr, "recall_sensitivity": rc, "recall_CI95_low": [a for a, _ in lo_hi],
        "recall_CI95_high": [b for _, b in lo_hi], "specificity": spec, "f1_score": f1, "roc_auc_ovr": aucs, "average_precision": aps, "support": sup})
    return m, pcl, cm

def bootstrap_all(y, probs, reps=BOOTSTRAP_REPS, seed=SEED):
    rng = np.random.default_rng(seed); n = len(y); K = probs.shape[1]; store = {k: [] for k in ["accuracy", "macro_f1", "macro_roc_auc_ovr", "cohen_kappa", "mcc", "ece_15_bins"]}
    for _ in range(reps):
        i = rng.integers(0, n, n); yy, pp = y[i], probs[i]; pr = pp.argmax(1)
        try:
            store["accuracy"].append(accuracy_score(yy, pr)); store["macro_f1"].append(f1_score(yy, pr, average="macro", zero_division=0))
            store["cohen_kappa"].append(cohen_kappa_score(yy, pr)); store["mcc"].append(matthews_corrcoef(yy, pr)); store["ece_15_bins"].append(ece_mce(pp, yy)[0])
            store["macro_roc_auc_ovr"].append(roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=np.arange(K)))
        except Exception: pass
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [np.nan, np.nan] for k, v in store.items()}

def mcnemar_exact(correct_a, correct_b):
    a = int(np.sum(correct_a & ~correct_b)); b = int(np.sum(~correct_a & correct_b)); d = a + b
    return {"only_A_correct": a, "only_B_correct": b, "discordant": d, "exact_p": float(binomtest(a, d, 0.5).pvalue) if d else 1.0}
