"""HEASNet component: Single-dataset experiment orchestration.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== ONE DATASET ========================================
CTX = {}
def run_dataset(ds, cfg):
    banner(f"STARTING DATASET: {ds}", "#"); seed_everything(SEED)
    od = os.path.join(OUTPUT_ROOT, ds.replace("-", "_")); os.makedirs(od, exist_ok=True)
    root = Path(cfg["root"])
    if not root.exists(): raise FileNotFoundError(f"Dataset path does not exist: {root}")
    print("Dataset found:", root)
    df = discover_hyperkvasir(root) if cfg["layout"] == "predefined_train_test" else discover_kvasir_v2(root)
    if df.label.nunique() != cfg["expected_classes"]: raise RuntimeError(f"{ds}: expected {cfg['expected_classes']} classes, found {df.label.nunique()}")
    df, n_bad = verify_images(df, ds, od)
    classes, tr, va, te = (create_hyperkvasir_splits if cfg["layout"] == "predefined_train_test" else create_kvasir_splits)(df)
    print(f"\nDATA SPLIT SUMMARY\nTotal: {len(df):,} | Train: {len(tr):,} ({len(tr)/len(df):.1%}) | Validation: {len(va):,} ({len(va)/len(df):.1%}) | Test: {len(te):,} ({len(te)/len(df):.1%})")
    if cfg["layout"] == "predefined_train_test": print(f"Augmented copies (*_aug_*): train {int(tr.path.map(is_aug_path).sum())} | validation {int(va.path.map(is_aug_path).sum())} | test {int(te.path.map(is_aug_path).sum())}")
    pd.concat([tr.assign(split="train"), va.assign(split="validation"), te.assign(split="test")]).to_csv(os.path.join(od, "exact_image_split_manifest.csv"), index=False)
    ct, ss, aud, imb = build_split_audit(classes, tr, va, te, od, ds); sample_grid(tr, classes, od, ds)
    trl, val, tel, dsets = make_loaders(tr, va, te)
    model = HEASNet(len(classes), USE_PRETRAINED); print(f"\nModel: HEASNet | classes {len(classes)} | parameters {sum(p.numel() for p in model.parameters())/1e6:.3f} M")
    print(f"Training protocol: {HEAD_WARMUP_EPOCHS} backbone-frozen epochs + {EPOCHS-HEAD_WARMUP_EPOCHS} fine-tuning epochs = {EPOCHS}; AdamW lr {LR_HEAD}/{LR_BACKBONE}, wd {WEIGHT_DECAY}, label smoothing {LABEL_SMOOTHING}")
    comp0 = compute_complexity(model.to(DEVICE), od, "HEASNet")
    best_path = os.path.join(od, "best_HEASNet.pt"); hist_path = os.path.join(od, "training_history.csv"); peak = None
    if REUSE_CHECKPOINT and os.path.exists(best_path) and os.path.exists(hist_path):
        print("Existing checkpoint found: skipping training and re-using it."); model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True)); h = pd.read_csv(hist_path)
        best_ep = int(h.val_accuracy.idxmax() + 1); best_acc = float(h.val_accuracy.max())
    else: model, h, best_ep, best_acc, best_path, peak = train_model(model, trl, val, od, "HEASNet")
    timing = {"total_training_and_validation_seconds": float(h.epoch_seconds.sum()), "total_minutes": float(h.epoch_seconds.sum() / 60), "mean_epoch_seconds": float(h.epoch_seconds.mean()),
              "mean_head_epoch_seconds": float(h.epoch_seconds[:HEAD_WARMUP_EPOCHS].mean()), "mean_finetune_epoch_seconds": float(h.epoch_seconds[HEAD_WARMUP_EPOCHS:].mean()), "peak_gpu_allocated_MB_training": peak,
              "best_epoch": best_ep, "best_val_accuracy": best_acc}
    json.dump(timing, open(os.path.join(od, "training_timing.json"), "w"), indent=2); print("\nTRAINING TIMING"); print(json.dumps(timing, indent=2)); plot_training(h, od, ds, best_ep)
    bench = benchmark_speed(model, od, "HEASNet"); plot_efficiency(comp0, bench, od, ds)
    loaders = [DataLoader(d, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=(NUM_WORKERS > 0)) for d in dsets]
    R = evaluate_and_report(model, loaders, dsets, classes, od, ds, tr)
    tag = f"val{100*best_acc:.2f}_test{100*R['test']['metrics']['accuracy']:.2f}"; shutil.copy(best_path, os.path.join(od, f"HEASNet_best_{tag}.pt"))
    banner(f"{ds}: EXPLAINABLE AI", "-"); gradcam_galleries(model, dsets[2], R["test"]["probs"], R["test"]["y"], classes, od)
    corr = attention_analysis(model, dsets[2], R["test"]["probs"], R["test"]["y"], classes, od); faith = faithfulness(model, dsets[2], R["test"]["probs"], R["test"]["y"], classes, od)
    dup = near_duplicate_analysis(ds, tr, va, te, R["test"]["y"], R["test"]["probs"], od)
    CTX[ds] = dict(name=ds, out_dir=od, classes=classes, tr=tr, va=va, te=te, counts=ct, audit=aud, imbalance=imb, hist=h, timing=timing, R=R, complexity=comp0, bench=bench, corr=corr, faith=faith, dup=dup, best_epoch=best_ep, best_val_acc=best_acc, n_corrupt=n_bad)
    summ = {"dataset": ds, "num_images": int(len(df)), "num_classes": len(classes), "classes": classes, "train": len(tr), "validation": len(va), "test": len(te), "best_epoch": best_ep, "best_val_accuracy": best_acc, "timing": timing,
            "split_metrics": {k: R[k]["metrics"] for k in ["train", "validation", "test"]}, "statistics": R["stat"], "calibration": R["cal"], "near_duplicates": dup, "complexity": {k: v for k, v in comp0.items() if k != "breakdown"}}
    json.dump(summ, open(os.path.join(od, "FINAL_EXPERIMENT_SUMMARY.json"), "w"), indent=2, default=str)
    banner(f"FINAL RESULT: {ds}", "="); print(pd.Series(R["test"]["metrics"]).round(5).to_string()); print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_ep}")
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return summ
