"""HEASNet component: Near-duplicate analysis.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== NEAR-DUPLICATES ========================================
HASH = {}
def dhash(path):
    if path in HASH: return HASH[path]
    with Image.open(path) as im: a = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    v = np.packbits((a[:, 1:] > a[:, :-1]).reshape(-1)).view(np.uint64)[0]; HASH[path] = v; return v

def hash_many(paths, label=""):
    out = np.empty(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        out[i] = dhash(p)
        if (i + 1) % 3000 == 0: print(f"   hashed {i+1}/{len(paths)} {label}")
    return out

def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555)); x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F); return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

def min_hamming(a, b, chunk=128):
    out = np.empty(len(a), dtype=np.int64)
    for s in range(0, len(a), chunk): out[s:s + chunk] = _popcount(a[s:s + chunk, None] ^ b[None, :]).min(1).astype(np.int64)
    return out

def near_duplicate_analysis(ds, tr, va, te, y_te, p_te, out_dir):
    banner(f"{ds}: NEAR-DUPLICATE AUDIT (64-bit dHash, Hamming <= {DUP_HAMMING})", "-")
    orig = tr[~tr.path.map(is_aug_path)]; ht = hash_many(orig.path.tolist(), "train originals"); hv = hash_many(va.path.tolist(), "validation"); he = hash_many(te.path.tolist(), "test")
    dte = min_hamming(he, ht); dva = min_hamming(hv, ht); ok = (p_te.argmax(1) == y_te)
    dup = dte <= DUP_HAMMING; ex = dte == 0
    res = {"train_original_images": int(len(orig)), "test_images": int(len(te)), "test_with_near_duplicate_in_train": int(dup.sum()), "test_near_duplicate_percent": float(100 * dup.mean()),
           "test_exact_duplicates": int(ex.sum()), "test_exact_duplicate_percent": float(100 * ex.mean()), "validation_with_near_duplicate_in_train": int((dva <= DUP_HAMMING).sum()),
           "validation_near_duplicate_percent": float(100 * (dva <= DUP_HAMMING).mean()),
           "accuracy_on_test_with_duplicate": float(ok[dup].mean()) if dup.any() else None, "accuracy_on_test_without_duplicate": float(ok[~dup].mean()) if (~dup).any() else None,
           "accuracy_all_test": float(ok.mean()), "n_test_without_duplicate": int((~dup).sum())}
    pd.DataFrame({"path": te.path, "min_hamming_to_train": dte, "near_duplicate": dup, "correct": ok}).to_csv(os.path.join(out_dir, "near_duplicate_per_test_image.csv"), index=False)
    json.dump(res, open(os.path.join(out_dir, "near_duplicate_summary.json"), "w"), indent=2); print(json.dumps(res, indent=2))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)); axs[0].hist(dte, bins=np.arange(0, 33), color=C_TRAIN); axs[0].axvline(DUP_HAMMING + .5, color="r", ls="--"); axs[0].set(xlabel="Min Hamming distance to a training image", ylabel="Test images", title="Near-duplicate distance")
    v = [100 * ok[dup].mean() if dup.any() else 0, 100 * ok[~dup].mean() if (~dup).any() else 0, 100 * ok.mean()]; axs[1].bar(["with duplicate", "without duplicate", "all test"], v, color=[C_VAL, C_TRAIN, C_TEST]); axs[1].set_ylim(80, 101); axs[1].set_ylabel("Accuracy (%)")
    for i, t in enumerate(v): axs[1].text(i, t + .2, f"{t:.2f}", ha="center")
    axs[1].set_title("Accuracy by duplicate status"); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_23_near_duplicate_audit.png")); return res
