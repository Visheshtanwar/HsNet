"""HEASNet component: Paper-ready tables and workbook generation.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== PAPER TABLES / NUMBERS / XLSX ========================================
def write_paper_tables(names):
    L = []; P = L.append
    P("PAPER-READY TABLES (all numbers computed by this run)\n" + "=" * 70)
    for n in names:
        c = CTX[n]; m = c["R"]["test"]["metrics"]; ci = c["R"]["ci"]; P(f"\n### {n}: overall test metrics (95% bootstrap CI, {BOOTSTRAP_REPS} resamples)")
        for k, v in ci.items(): P(f"  {k:<22} {m[k]:.4f}   [{v[0]:.4f}, {v[1]:.4f}]")
        P(f"  precision/recall/F1 macro: {100*m['macro_precision']:.2f} / {100*m['macro_recall']:.2f} / {100*m['macro_f1']:.2f}   weighted F1 {100*m['weighted_f1']:.2f}")
        P(f"  kappa {m['cohen_kappa']:.4f} | MCC {m['mcc']:.4f} | macro AUC {m['macro_roc_auc_ovr']:.4f} | micro AUC {m['micro_roc_auc']:.4f} | log loss {m['log_loss']:.4f} | Brier(MSE) {m['brier_score_probability_MSE']:.4f} | ECE {m['ece_15_bins']:.4f}")
        P(f"\n  class-wise test metrics:\n" + c["R"]["test"]["pcl"].round(4).to_string(index=False))
        P(f"\n  train/val/test accuracy: " + " / ".join(f"{100*c['R'][s]['metrics']['accuracy']:.2f}" for s in ["train", "validation", "test"]))
        P(f"  best validation acc {100*c['best_val_acc']:.2f}% at epoch {c['best_epoch']} | training {c['timing']['total_minutes']:.1f} min | mean epoch {c['timing']['mean_epoch_seconds']:.1f}s")
        P(f"  params {c['complexity']['parameters_million']:.3f} M | GMACs {c['complexity']['gmacs']} | GFLOPs {c['complexity'].get('gflops')}")
        P("  speed/memory:\n" + c["bench"].round(3).to_string(index=False))
        P("  near-duplicates: " + json.dumps(c["dup"]))
        P("  top confusions:\n" + (c["R"]["top_conf"].to_string(index=False) if hasattr(c["R"]["top_conf"], "to_string") else str(c["R"]["top_conf"])))
    open(os.path.join(OUTPUT_ROOT, "PAPER_READY_TABLES.txt"), "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L[:6]), "\n... (full text in PAPER_READY_TABLES.txt)")

def write_numbers_and_xlsx(names):
    nums = {n: json.load(open(os.path.join(CTX[n]["out_dir"], "FINAL_EXPERIMENT_SUMMARY.json"))) for n in names}
    json.dump(nums, open(os.path.join(OUTPUT_ROOT, "manuscript_numbers.json"), "w"), indent=2, default=str)
    try:
        with pd.ExcelWriter(os.path.join(OUTPUT_ROOT, "ALL_RESULTS_TABLES.xlsx")) as xw:
            for n in names:
                tag = n.replace("-", "")[:12]
                for fn in sorted(os.listdir(CTX[n]["out_dir"])):
                    if fn.endswith(".csv") and os.path.getsize(os.path.join(CTX[n]["out_dir"], fn)) < 2_000_000 and not fn.startswith(("exact_image", "near_dup")):
                        try: pd.read_csv(os.path.join(CTX[n]["out_dir"], fn)).to_excel(xw, sheet_name=f"{tag}_{fn[:-4]}"[:31], index=False)
                        except Exception: pass
        print("Excel workbook written.")
    except Exception as e: print("xlsx skipped:", e)
