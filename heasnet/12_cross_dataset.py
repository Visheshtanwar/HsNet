"""HEASNet component: Cross-dataset analysis.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== CROSS-DATASET ========================================
def cross_dataset(names):
    banner("CROSS-DATASET COMPARISON", "=")
    od = os.path.join(OUTPUT_ROOT, "CROSS_DATASET"); os.makedirs(od, exist_ok=True)
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_roc_auc", "macro_roc_auc_ovr", "cohen_kappa", "mcc", "log_loss", "brier_score_probability_MSE", "ece_15_bins"]
    rows = []
    for n in names:
        c = CTX[n]; m = c["R"]["test"]["metrics"]; ci = c["R"]["ci"]
        r = {"dataset": n, "classes": len(c["classes"]), "train": len(c["tr"]), "val": len(c["va"]), "test": len(c["te"]), "params_M": c["complexity"]["parameters_million"], "gmacs": c["complexity"]["gmacs"],
             "best_epoch": c["best_epoch"], "best_val_acc": c["best_val_acc"], "train_minutes": c["timing"]["total_minutes"]}
        r.update({k: m.get(k) for k in keys}); r["acc_ci_low"], r["acc_ci_high"] = ci["accuracy"]; rows.append(r)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(od, "cross_dataset_summary.csv"), index=False); print(d.T.to_string(header=False))
    if len(names) > 1:
        fig, ax = plt.subplots(figsize=(11, 4.8)); w = .8 / len(names); sh = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc_ovr", "cohen_kappa", "mcc"]
        for i, n in enumerate(names):
            v = [CTX[n]["R"]["test"]["metrics"][k] for k in sh]; b = ax.bar(np.arange(len(sh)) + i * w, v, w, label=n, color=[C_TRAIN, C_VAL, C_TEST][i % 3])
            for rect, t in zip(b, v): ax.text(rect.get_x() + rect.get_width() / 2, t + .002, f"{t:.3f}", ha="center", fontsize=7)
        ax.set_xticks(np.arange(len(sh)) + w * (len(names) - 1) / 2); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC", "Kappa", "MCC"]); ax.set_ylim(.85, 1.01); ax.legend(); ax.set_title("Test performance across datasets"); fig.tight_layout(); savefig(fig, os.path.join(od, "Fig_28_cross_dataset_metrics.png"))
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for n, col in zip(names, [C_TRAIN, C_VAL]):
            h = CTX[n]["hist"]; axs[0].plot(h.epoch, h.val_accuracy * 100, label=n, color=col); axs[1].plot(h.epoch, h.val_loss, label=n, color=col)
        axs[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Validation accuracy"); axs[1].set(xlabel="Epoch", ylabel="Validation loss", title="Validation loss"); axs[0].legend(); fig.tight_layout(); savefig(fig, os.path.join(od, "Fig_29_cross_dataset_validation_curves.png"))
    if "HyperKvasir" in CTX and "Kvasir-V2" in CTX:
        try:
            hk, kv = CTX["HyperKvasir"], CTX["Kvasir-V2"]; hh = hash_many(pd.concat([hk["tr"], hk["va"], hk["te"]]).path.tolist(), "HyperKvasir all"); hk_ = hash_many(kv["te"].path.tolist(), "Kvasir test")
            dm = min_hamming(hk_, hh); ov = dm <= DUP_HAMMING; ok = kv["R"]["test"]["probs"].argmax(1) == kv["R"]["test"]["y"]
            res = {"kvasir_test_images": int(len(ov)), "overlap_with_hyperkvasir": int(ov.sum()), "overlap_percent": float(100 * ov.mean()), "kvasir_acc_on_overlap": float(ok[ov].mean()) if ov.any() else None, "kvasir_acc_non_overlap": float(ok[~ov].mean()) if (~ov).any() else None}
            json.dump(res, open(os.path.join(od, "kvasir_vs_hyperkvasir_overlap.json"), "w"), indent=2); print("Kvasir-V2 test vs HyperKvasir overlap:", res)
        except Exception as e: print("cross overlap skipped:", e)
    return d
