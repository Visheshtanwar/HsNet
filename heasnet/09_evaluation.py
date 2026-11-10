"""HEASNet component: Full train/validation/test evaluation.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== FULL EVALUATION OF ONE MODEL ========================================
def evaluate_and_report(model, loaders, dsets, classes, out_dir, ds, tr_df):
    banner(f"{ds}: FULL EVALUATION (train / validation / test)", "-")
    R = {}
    for sp, ld in zip(["train", "validation", "test"], loaders):
        P, Y = predict(model, ld); m, pcl, cm = compute_metrics(Y, P, classes); R[sp] = {"probs": P, "y": Y, "metrics": m, "pcl": pcl, "cm": cm}
        pcl.to_csv(os.path.join(out_dir, f"{sp}_classwise_metrics.csv"), index=False); pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out_dir, f"{sp}_confusion_matrix.csv"))
        np.savez_compressed(os.path.join(out_dir, f"predictions_{sp}.npz"), y_true=Y, probabilities=P, y_pred=P.argmax(1), class_names=np.asarray(classes), paths=np.asarray(dsets[["train", "validation", "test"].index(sp)].paths))
        print(f"\n{sp.upper()} METRICS"); print(pd.Series(m).round(5).to_string()); print(f"\n{sp.upper()} class-wise metrics:"); print(pcl.round(4).to_string(index=False))
    sm = pd.DataFrame([{"split": k, **R[k]["metrics"]} for k in ["train", "validation", "test"]]); sm.to_csv(os.path.join(out_dir, "all_split_metrics.csv"), index=False)
    gap = pd.DataFrame([{"metric": k, "train": R["train"]["metrics"][k], "validation": R["validation"]["metrics"][k], "test": R["test"]["metrics"][k], "train_minus_test": R["train"]["metrics"][k] - R["test"]["metrics"][k]} for k in ["accuracy", "macro_f1", "macro_roc_auc_ovr", "cohen_kappa", "mcc", "log_loss", "ece_15_bins"]])
    gap.to_csv(os.path.join(out_dir, "generalisation_gap_train_val_test.csv"), index=False); print("\nGENERALISATION GAP"); print(gap.round(4).to_string(index=False))
    plot_split_comparison({k: R[k]["metrics"] for k in ["train", "validation", "test"]}, out_dir, ds)
    y, P = R["test"]["y"], R["test"]["probs"]; pred = P.argmax(1); K = len(classes); tm = R["test"]["metrics"]
    ci = bootstrap_all(y, P); maj = Counter(y.tolist()).most_common(1)[0][1] / len(y); tmaj = int(tr_df.target.value_counts().idxmax())
    mc = mcnemar_exact(pred == y, np.full_like(y, tmaj) == y); acc_w = wilson_interval(int((pred == y).sum()), len(y))
    stat = {"bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_95CI": ci, "accuracy_wilson_95CI": list(acc_w), "binomial_vs_uniform_chance_p": float(binomtest(int((pred == y).sum()), len(y), 1 / K, alternative="greater").pvalue),
            "binomial_vs_test_majority_rate_p": float(binomtest(int((pred == y).sum()), len(y), maj, alternative="greater").pvalue), "test_majority_class_rate": float(maj),
            "McNemar_vs_training_majority_baseline": {"baseline_class": classes[tmaj], "only_HEASNet_correct": mc["only_A_correct"], "only_baseline_correct": mc["only_B_correct"], "discordant": mc["discordant"], "exact_p": mc["exact_p"]}}
    json.dump(stat, open(os.path.join(out_dir, "statistical_tests.json"), "w"), indent=2); print("\nTEST STATISTICAL ANALYSIS"); print(json.dumps(stat, indent=2))
    pd.DataFrame([{"metric": k, "estimate": tm[k], "ci_low": v[0], "ci_high": v[1]} for k, v in ci.items()]).to_csv(os.path.join(out_dir, "bootstrap_confidence_intervals.csv"), index=False)
    pd.DataFrame(classification_report(y, pred, labels=np.arange(K), target_names=classes, output_dict=True, zero_division=0)).T.to_csv(os.path.join(out_dir, "test_classification_report_full.csv"))
    plot_confusion(R["test"]["cm"], classes, out_dir, False); plot_confusion(R["test"]["cm"], classes, out_dir, True, num="07")
    plot_roc_pr(y, P, classes, out_dir); plot_classwise(R["test"]["pcl"], out_dir); sel, cal = plot_calibration(y, P, out_dir); plot_ci_forest(ci, tm, out_dir, ds)
    tc = top_confusions(R["test"]["cm"], classes, out_dir); misclassified_gallery(dsets[2].paths, y, P, classes, out_dir)
    R["stat"], R["ci"], R["cal"], R["top_conf"], R["selective"] = stat, ci, cal, tc, sel
    return R
