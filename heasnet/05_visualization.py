"""HEASNet component: Training curves, confusion matrices, ROC/PR and calibration plots.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== PLOTS ========================================
def plot_training(h, out_dir, ds, best_ep, warmup=HEAD_WARMUP_EPOCHS):
    ep = h.epoch
    for key, ylab, fn, ttl in [("loss", "Cross-entropy loss", "Fig_03_loss_curve.png", "Training and validation loss"),
                               ("accuracy", "Accuracy", "Fig_04_accuracy_curve.png", "Training and validation accuracy")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ep, h[f"train_{key}"], color=C_TRAIN, lw=2, label="Training"); ax.plot(ep, h[f"val_{key}"], color=C_VAL, lw=2, label="Validation")
        ax.axvline(warmup + .5, color="gray", ls=":"); ax.text(warmup + .8, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * .03, "fine-tuning starts", fontsize=8, color="gray")
        if key == "accuracy": ax.scatter([best_ep], [h.val_accuracy[best_ep - 1]], color=C_VAL, zorder=5); ax.annotate(f"selected epoch {best_ep}", (best_ep, h.val_accuracy[best_ep - 1]), xytext=(-80, -25), textcoords="offset points", fontsize=8, color=C_VAL, arrowprops=dict(arrowstyle="-", color=C_VAL))
        ax.set(xlabel="Epoch", ylabel=ylab, title=f"{ds}: {ttl}"); ax.legend(); fig.tight_layout(); savefig(fig, os.path.join(out_dir, fn))
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    axs[0, 0].plot(ep, h.train_loss, color=C_TRAIN, label="train"); axs[0, 0].plot(ep, h.val_loss, color=C_VAL, label="val"); axs[0, 0].set_title("Loss"); axs[0, 0].legend()
    axs[0, 1].plot(ep, h.train_accuracy, color=C_TRAIN, label="train"); axs[0, 1].plot(ep, h.val_accuracy, color=C_VAL, label="val"); axs[0, 1].set_title("Accuracy"); axs[0, 1].legend()
    axs[1, 0].plot(ep, h.train_accuracy - h.val_accuracy, color="#7b2d8e"); axs[1, 0].axhline(0, color="k", lw=.5); axs[1, 0].set_title("Generalisation gap (train acc - val acc)")
    axs[1, 1].bar(ep, h.epoch_seconds, color="#888"); axs[1, 1].set_title("Seconds per epoch")
    for a in axs.ravel(): a.set_xlabel("Epoch")
    fig.suptitle(f"{ds}: learning dynamics"); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_05_learning_dynamics.png"))

def plot_confusion(cm, classes, out_dir, normalized, split="test", num="06"):
    M = cm.astype(float)
    if normalized: rs = M.sum(1, keepdims=True); M = np.divide(M, rs, out=np.zeros_like(M), where=rs != 0)
    n = len(classes); fig, ax = plt.subplots(figsize=(max(8, n * .8), max(7, n * .72)))
    im = ax.imshow(M, cmap="Blues"); ax.grid(False)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(classes, rotation=55, ha="right", fontsize=8); ax.set_yticklabels(classes, fontsize=8)
    th = np.nanmax(M) / 2
    for i in range(n):
        for j in range(n):
            if normalized and M[i, j] == 0: continue
            ax.text(j, i, f"{M[i,j]:.2f}" if normalized else f"{int(cm[i,j])}", ha="center", va="center", color="white" if M[i, j] > th else "black", fontsize=7)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(f"{split.title()} confusion matrix ({'row-normalised' if normalized else 'counts'})")
    fig.colorbar(im, fraction=.046); fig.tight_layout()
    savefig(fig, os.path.join(out_dir, f"Fig_{num}_confusion_{'normalized' if normalized else 'counts'}_{split}.png"))

def plot_roc_pr(y, probs, classes, out_dir):
    K = len(classes); yb = np.eye(K)[y]; rows = []
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5)); cmap = plt.get_cmap("tab20")
    for k, nm in enumerate(classes):
        if len(np.unique(yb[:, k])) < 2: continue
        fpr, tpr, _ = roc_curve(yb[:, k], probs[:, k]); a = auc(fpr, tpr); rows.append({"class": nm, "auc_ovr": a})
        for ax in axs: ax.plot(fpr, tpr, lw=1.5, color=cmap(k), label=f"{nm} ({a:.3f})")
    fpr, tpr, _ = roc_curve(yb.ravel(), probs.ravel()); ma = auc(fpr, tpr)
    for ax in axs: ax.plot(fpr, tpr, "k--", lw=2, label=f"micro-average ({ma:.3f})"); ax.plot([0, 1], [0, 1], "k:", lw=1)
    axs[0].set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC curves (test)")
    axs[1].set(xlim=(0, .2), ylim=(.8, 1.005), xlabel="False positive rate", ylabel="True positive rate", title="Zoom: top-left corner"); axs[0].legend(loc="lower right", fontsize=7)
    fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_09_ROC_curves_test.png")); pd.DataFrame(rows).to_csv(os.path.join(out_dir, "test_classwise_AUC.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for k, nm in enumerate(classes):
        if len(np.unique(yb[:, k])) < 2: continue
        p, r, _ = precision_recall_curve(yb[:, k], probs[:, k]); ax.plot(r, p, lw=1.5, color=cmap(k), label=f"{nm} (AP {average_precision_score(yb[:, k], probs[:, k]):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="One-vs-rest precision-recall curves (test)"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_10_precision_recall_curves_test.png"))

def plot_classwise(pcl, out_dir):
    cols = ["precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc_ovr", "average_precision"]; d = pcl.set_index("class")[cols]
    fig, ax = plt.subplots(figsize=(8.5, max(5, len(d) * .5))); im = ax.imshow(d.values, cmap="YlGnBu", vmin=min(.6, np.nanmin(d.values)), vmax=1, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(["Precision", "Recall", "Specificity", "F1", "ROC-AUC", "Avg. precision"], rotation=30, ha="right"); ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]): ax.text(j, i, f"{d.values[i,j]:.3f}", ha="center", va="center", fontsize=7, color="white" if d.values[i, j] > .93 else "black")
    ax.set_title("Class-wise metric heat-map (test)"); fig.colorbar(im, fraction=.04); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_11_classwise_heatmap.png"))
    fig, ax = plt.subplots(figsize=(max(9, len(d) * .9), 5)); x = np.arange(len(pcl)); r = pcl.recall_sensitivity.values
    ax.bar(x - .2, pcl.f1_score, .4, color=C_TRAIN, label="F1"); ax.bar(x + .2, r, .4, color=C_VAL, label="Recall")
    ax.errorbar(x + .2, r, yerr=[np.maximum(0, r - pcl.recall_CI95_low.values), np.maximum(0, pcl.recall_CI95_high.values - r)], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(pcl["class"], rotation=50, ha="right"); ax.set_ylim(.5, 1.03); ax.set_title("Class-wise F1 and recall with 95% Wilson CI (test)"); ax.legend(); fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "Fig_12_classwise_bars_recall_CI.png"))

def plot_calibration(y, probs, out_dir):
    conf = probs.max(1); ok = (probs.argmax(1) == y).astype(float); e = np.linspace(0, 1, 16); xs, ys, ns = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): xs.append(conf[m].mean()); ys.append(ok[m].mean()); ns.append(int(m.sum()))
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    axs[0].plot([0, 1], [0, 1], "k:", label="perfect"); axs[0].plot(xs, ys, "o-", color=C_TRAIN, label="HEASNet")
    for a, b, n in zip(xs, ys, ns): axs[0].annotate(str(n), (a, b), fontsize=6, xytext=(0, 5), textcoords="offset points", ha="center")
    axs[0].set(xlabel="Mean predicted confidence", ylabel="Observed accuracy", title="Reliability diagram (test; numbers = images per bin)", xlim=(0, 1), ylim=(0, 1.02)); axs[0].legend()
    b = np.linspace(min(conf.min(), .99), 1, 25); axs[1].hist(conf[ok == 1], bins=b, alpha=.7, color=C_TEST, label="correct"); axs[1].hist(conf[ok == 0], bins=b, alpha=.8, color=C_VAL, label="wrong")
    axs[1].set(xlabel="Confidence (max softmax)", ylabel="Images", title="Confidence of correct vs wrong predictions"); axs[1].legend()
    order = np.argsort(-conf); acc_c = np.cumsum(ok[order]) / np.arange(1, len(y) + 1); axs[2].plot(np.arange(1, len(y) + 1) / len(y), acc_c, color=C_TRAIN, lw=2)
    axs[2].set(xlabel="Coverage (fraction kept, most confident first)", ylabel="Accuracy on kept", title="Risk-coverage curve"); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_13_calibration_reliability.png"))
    rows = []
    for t in [0, .5, .6, .7, .8, .9, .95]:
        m = conf >= t
        if m.sum(): rows.append({"confidence_threshold": t, "images_retained": int(m.sum()), "coverage_percent": 100 * m.mean(), "accuracy_on_retained_percent": 100 * ok[m].mean(), "errors_remaining": int((1 - ok[m]).sum())})
    sel = pd.DataFrame(rows); sel.to_csv(os.path.join(out_dir, "selective_accuracy_table.csv"), index=False)
    print("\nSELECTIVE ACCURACY (confidence threshold)"); print(sel.round(2).to_string(index=False))
    cal = {"mean_confidence": float(conf.mean()), "accuracy": float(ok.mean()), "mean_conf_correct": float(conf[ok == 1].mean()), "mean_conf_wrong": float(conf[ok == 0].mean()) if (ok == 0).any() else None}
    return sel, cal

def plot_split_comparison(split_metrics, out_dir, ds):
    keys = ["accuracy", "macro_f1", "macro_roc_auc_ovr", "cohen_kappa", "mcc"]; fig, ax = plt.subplots(figsize=(9.5, 5)); x = np.arange(len(keys))
    for j, (sp, col) in enumerate(zip(["train", "validation", "test"], [C_TRAIN, C_VAL, C_TEST])):
        v = [split_metrics[sp][k] for k in keys]; ax.bar(x + (j - 1) * .27, v, .27, color=col, label=sp)
        for a, b in zip(x + (j - 1) * .27, v): ax.text(a, b + .003, f"{b:.3f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(["Accuracy", "Macro F1", "Macro AUC", "Kappa", "MCC"]); ax.set_ylim(.8, 1.04); ax.set_title(f"{ds}: train vs validation vs test"); ax.legend(); fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "Fig_15_train_val_test_comparison.png"))

def plot_ci_forest(ci, metrics, out_dir, ds):
    ks = list(ci.keys()); fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, k in enumerate(ks):
        v = metrics[k]; lo, hi = ci[k]; ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=C_TRAIN, capsize=4); ax.text(hi + .002, i, f"{v:.4f} [{lo:.4f}, {hi:.4f}]", va="center", fontsize=8)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks); ax.set_xlim(.8, 1.08); ax.set_title(f"{ds}: test metrics with 95% bootstrap CI"); ax.invert_yaxis(); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_25_bootstrap_CI_forest.png"))

def top_confusions(cm, classes, out_dir, k=12):
    rows = [{"true_class": classes[i], "predicted_class": classes[j], "count": int(cm[i, j]), "percent_of_true_class": 100 * cm[i, j] / max(cm[i].sum(), 1)} for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    d = pd.DataFrame(rows).sort_values("count", ascending=False).head(k); d.to_csv(os.path.join(out_dir, "top_confusions.csv"), index=False)
    print("\nTOP CONFUSIONS (test)"); print(d.round(2).to_string(index=False))
    if len(d):
        fig, ax = plt.subplots(figsize=(8, .45 * len(d) + 1.5)); lab = [f"{a} -> {b}" for a, b in zip(d.true_class, d.predicted_class)]
        ax.barh(range(len(d)), d["count"], color=C_VAL); ax.set_yticks(range(len(d))); ax.set_yticklabels(lab, fontsize=8); ax.invert_yaxis(); ax.set_xlabel("Images"); ax.set_title("Most frequent confusions (test)"); fig.tight_layout()
        savefig(fig, os.path.join(out_dir, "Fig_16_top_confusions.png"))
    return d

def load_rgb(p): return Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

def misclassified_gallery(paths, y, probs, classes, out_dir, n=20):
    pred = probs.argmax(1); wrong = np.where(pred != y)[0]
    if len(wrong) == 0: return
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:n]; cols = 5; rows = int(math.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), squeeze=False)
    for a in axes.ravel(): a.axis("off")
    for k, i in enumerate(wrong):
        a = axes[k // cols, k % cols]; a.imshow(load_rgb(paths[i])); a.set_title(f"True: {classes[y[i]]}\nPred: {classes[pred[i]]} ({probs[i].max():.2f})", fontsize=7, color="red")
    fig.suptitle("Most confident misclassified test images"); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_17_misclassified_gallery.png"))
_trapz = getattr(np, "trapezoid", None) or np.trapz
