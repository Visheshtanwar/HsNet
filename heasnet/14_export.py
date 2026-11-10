"""HEASNet component: Result archive generation.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== ZIP ========================================
def make_zip(final=False):
    zp = ZIP_BASE + ".zip"
    if os.path.exists(zp): os.remove(zp)
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for r, _, fs in os.walk(OUTPUT_ROOT):
            for f in fs:
                fp = os.path.join(r, f)
                if f.endswith(".pt") and not (INCLUDE_ALL_CHECKPOINTS_IN_ZIP or f == "best_HEASNet.pt"): continue
                if f == "FULL_NOTEBOOK_LOG.txt": _LOG.flush()
                z.write(fp, os.path.relpath(fp, os.path.dirname(OUTPUT_ROOT))); n += 1
    print(f"ZIP written ({'FINAL' if final else 'checkpoint'}): {zp} | {n} files | {os.path.getsize(zp)/1e6:.1f} MB"); return zp

def list_outputs():
    banner("FILES SAVED (relative path, size)", "=")
    tot = 0
    for r, _, fs in sorted(os.walk(OUTPUT_ROOT)):
        for f in sorted(fs):
            s = os.path.getsize(os.path.join(r, f)); tot += s; print(f"{os.path.relpath(os.path.join(r, f), OUTPUT_ROOT):<80} {s/1e3:>10.1f} KB")
    print(f"TOTAL {tot/1e6:.1f} MB")
