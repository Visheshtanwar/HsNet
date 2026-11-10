"""HEASNet component: Experiment entry point.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== MAIN ========================================
def main():
    banner(f"HEASNet FINAL ALL-IN-ONE RUN | device {DEVICE} | {datetime.datetime.now():%Y-%m-%d %H:%M}")
    print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)")
    done = []
    for ds, cfg in DATASETS.items():
        try: run_dataset(ds, cfg); done.append(ds)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"!! {ds} FAILED: {e}")
        try: make_zip()
        except Exception as e: print("zip failed:", e)
    if done:
        try: cross_dataset(done)
        except Exception as e: print("cross-dataset failed:", e)
        try: write_paper_tables(done); write_numbers_and_xlsx(done)
        except Exception as e: print("tables failed:", e)
        make_zip()
    if RUN_EXTRAS:
        for ds in EXTRA_DATASETS:
            if ds in CTX:
                try: banner(f"EXTRA EXPERIMENTS: {ds}", "#"); run_extras(CTX[ds])
                except Exception as e:
                    import traceback; traceback.print_exc(); print(f"!! extras {ds} failed: {e}")
                try: make_zip()
                except Exception as e: print("zip failed:", e)
    banner(f"ALL DONE in {elapsed_h():.2f} h", "=")
    list_outputs(); make_zip(final=True)
    try:
        from IPython.display import FileLink, display; display(FileLink(os.path.relpath(ZIP_BASE + ".zip")))
    except Exception: pass
    print("DOWNLOAD:", ZIP_BASE + ".zip")

main()
