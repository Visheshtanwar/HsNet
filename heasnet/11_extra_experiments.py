"""HEASNet component: Ablation, baselines, seeds and sweeps.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== OPTIONAL EXTRA EXPERIMENTS ========================================
def run_variant(c, name, builder, seed=SEED, opt="adamw", lr_head=LR_HEAD, lr_bb=LR_BACKBONE, epochs=EPOCHS, warmup=HEAD_WARMUP_EPOCHS):
    od = os.path.join(c["out_dir"], "extras"); os.makedirs(od, exist_ok=True); safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", name); jf = os.path.join(od, f"variant_{safe}.json")
    if os.path.exists(jf): print(f"[{name}] already done, loading"); return json.load(open(jf))
    banner(f"EXTRA RUN: {name} | {c['name']} | seed {seed} | {epochs} epochs | opt {opt} | lr {lr_head}", "-"); seed_everything(seed)
    trl, val, tel, _ = make_loaders(c["tr"], c["va"], c["te"], seed); model = builder().to(DEVICE)
    model, h, be, ba, bp, peak = train_model(model, trl, val, od, safe, lr_head, lr_bb, opt, epochs, min(warmup, epochs), verbose=False, history_name=f"history_{safe}.csv")
    pv, yv = predict(model, val); pt, yt = predict(model, tel); mv, _, _ = compute_metrics(yv, pv, c["classes"]); mt, _, _ = compute_metrics(yt, pt, c["classes"])
    comp = compute_complexity(model, od, safe, save=False); x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
    def f():
        with torch.inference_mode(): model(x)
    ts = _timeit(f, 5 if QUICK else 40, 1); pm = float(torch.cuda.max_memory_allocated() / 1e6) if DEVICE.type == "cuda" else None
    row = {"name": name, "dataset": c["name"], "seed": seed, "optimizer": opt, "lr_head": lr_head, "epochs": epochs, "best_epoch": be, "best_val_accuracy": ba, "params_M": comp["parameters_million"], "gmacs": comp["gmacs"],
           "latency_ms_batch1": float(ts.mean()), "peak_gpu_MB_inference": pm, "peak_gpu_MB_training": peak, "train_minutes": float(h.epoch_seconds.sum() / 60),
           **{f"val_{k}": v for k, v in mv.items() if k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}, **{f"test_{k}": v for k, v in mt.items()}}
    np.savez_compressed(os.path.join(od, f"test_predictions_{safe}.npz"), probs=pt, y=yt); json.dump(row, open(jf, "w"), indent=2, default=str)
    print(f"[{name}] test acc {mt['accuracy']:.4f} | macro F1 {mt['macro_f1']:.4f} | val acc {ba:.4f} | {row['train_minutes']:.1f} min")
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return row

def _pt(c, name): return np.load(os.path.join(c["out_dir"], "extras", f"test_predictions_{re.sub(r'[^A-Za-z0-9_.+-]', '_', name)}.npz"))["probs"]

def run_extras(c):
    ds = c["name"]; K = len(c["classes"]); od = os.path.join(c["out_dir"], "extras"); os.makedirs(od, exist_ok=True); skipped = []; y = c["R"]["test"]["y"]; main_p = c["R"]["test"]["probs"]
    main_row = {"name": "HEASNet (proposed)", "params_M": c["complexity"]["parameters_million"], "gmacs": c["complexity"]["gmacs"], **{f"test_{k}": v for k, v in c["R"]["test"]["metrics"].items()},
                "val_accuracy": c["best_val_acc"], "latency_ms_batch1": float(c["bench"].query("batch_size==1 and precision=='fp32'").mean_ms_per_image.iloc[0])}
    def guard(what, hrs=0.9):
        if not time_left_ok(hrs): skipped.append(what); print(f"!! time budget: skipping {what}"); return False
        return True
    if RUN_ABLATION:
        rows = []
        for nm, se, wi, ax in [("Base model (EfficientNetV2-S)", 0, 0, 0), ("+ SE", 1, 0, 0), ("+ Window attention", 0, 1, 0), ("+ Window + SE", 1, 1, 0), ("+ Axial", 0, 0, 1), ("+ SE + Axial", 1, 0, 1)]:
            if not guard(f"ablation {nm} {ds}"): continue
            r = run_variant(c, f"abl_{nm}", lambda se=se, wi=wi, ax=ax: HEASNet(K, USE_PRETRAINED, bool(se), bool(wi), bool(ax))); r["name"] = nm; rows.append(r)
        if rows:
            for r in rows: r["McNemar_p_vs_full"] = mcnemar_exact(_pt(c, "abl_" + r["name"]).argmax(1) == y, main_p.argmax(1) == y)["exact_p"]
            d = pd.DataFrame(rows + [main_row]); d.to_csv(os.path.join(od, "ablation_results.csv"), index=False); print("\nABLATION RESULTS"); print(d[["name", "params_M", "gmacs", "test_accuracy", "test_macro_f1", "test_macro_roc_auc_ovr", "best_val_accuracy"]].round(4).to_string(index=False))
            fig, ax = plt.subplots(figsize=(9, 4.8)); a = d.test_accuracy.values * 100; ax.barh(range(len(d)), a, color=[C_TRAIN] * (len(d) - 1) + [C_VAL]); ax.set_yticks(range(len(d))); ax.set_yticklabels(d.name); ax.invert_yaxis()
            ax.set_xlim(max(0, a.min() - 3), 100); ax.set_xlabel("Test accuracy (%)"); [ax.text(v + .1, i, f"{v:.2f}", va="center", fontsize=8) for i, v in enumerate(a)]; ax.set_title(f"{ds}: ablation (single run each)"); fig.tight_layout(); savefig(fig, os.path.join(od, "Fig_26_ablation.png"))
    if RUN_BASELINES:
        rows = []
        for nm in ["ResNet50", "EfficientNetV2-S", "ConvNeXt-T", "ViT-B/16"]:
            if not guard(f"baseline {nm} {ds}", 1.1): continue
            rows.append(run_variant(c, f"base_{nm}", lambda nm=nm: BaselineNet(nm, K, USE_PRETRAINED)) | {"name": nm})
        if rows:
            for r in rows: r["McNemar_p_vs_HEASNet"] = mcnemar_exact(_pt(c, "base_" + r["name"]).argmax(1) == y, main_p.argmax(1) == y)["exact_p"]
            d = pd.DataFrame(rows + [main_row]); d.to_csv(os.path.join(od, "baselines_results.csv"), index=False); print("\nBASELINES"); print(d[["name", "params_M", "gmacs", "latency_ms_batch1", "peak_gpu_MB_inference", "test_accuracy", "test_macro_f1", "test_macro_roc_auc_ovr"]].round(4).to_string(index=False))
            fig, ax = plt.subplots(figsize=(8, 4.6)); a = d.test_accuracy.values * 100; ax.bar(d.name, a, color=[C_TRAIN] * (len(d) - 1) + [C_VAL]); ax.set_ylim(max(0, a.min() - 4), 100.5); ax.set_ylabel("Test accuracy (%)")
            [ax.text(i, v + .1, f"{v:.2f}", ha="center", fontsize=8) for i, v in enumerate(a)]; plt.xticks(rotation=20); ax.set_title(f"{ds}: HEASNet vs retrained baselines (same protocol)"); fig.tight_layout(); savefig(fig, os.path.join(od, "Fig_27_baselines.png"))
    if REPEAT_SEEDS:
        rows = [{**main_row, "seed": SEED}]
        for sd in REPEAT_SEEDS:
            if not guard(f"seed {sd} {ds}"): continue
            rows.append(run_variant(c, f"seed{sd}", lambda: HEASNet(K, USE_PRETRAINED), seed=sd))
        if len(rows) > 1:
            d = pd.DataFrame(rows); d.to_csv(os.path.join(od, "seed_runs.csv"), index=False); ks = ["test_accuracy", "test_macro_f1", "test_macro_roc_auc_ovr", "test_cohen_kappa", "test_mcc", "test_ece_15_bins"]
            s = pd.DataFrame({"mean": d[ks].mean(), "sd": d[ks].std(ddof=1)}); s.to_csv(os.path.join(od, "seed_stability_mean_sd.csv")); print("\nSEED STABILITY (mean, SD)"); print(s.round(5).to_string())
    if RUN_SWEEPS:
        rows = []
        for lr in LR_SWEEP:
            if guard(f"lr sweep {lr} {ds}", .6): rows.append(run_variant(c, f"lr_{lr}", lambda: HEASNet(K, USE_PRETRAINED), lr_head=lr, lr_bb=lr * .1, epochs=SWEEP_EPOCHS))
        if rows: d = pd.DataFrame(rows); d.to_csv(os.path.join(od, "sweep_learning_rate.csv"), index=False); print("\nLEARNING-RATE SWEEP (validation)"); print(d[["name", "lr_head", "val_macro_precision", "val_macro_recall", "val_macro_f1", "val_accuracy"]].round(4).to_string(index=False))
        rows = []
        for o in OPT_SWEEP:
            if guard(f"opt sweep {o} {ds}", .6): rows.append(run_variant(c, f"opt_{o}", lambda: HEASNet(K, USE_PRETRAINED), opt=o, epochs=SWEEP_EPOCHS))
        if rows: d = pd.DataFrame(rows); d.to_csv(os.path.join(od, "sweep_optimizer.csv"), index=False); print("\nOPTIMISER SWEEP (validation; AdamW = the 1e-4 row of the learning-rate sweep)"); print(d[["name", "val_macro_precision", "val_macro_recall", "val_macro_f1", "val_accuracy"]].round(4).to_string(index=False))
    if skipped: open(os.path.join(od, "SKIPPED_DUE_TO_TIME_BUDGET.txt"), "w").write("\n".join(skipped)); print("Skipped due to time:", skipped)
