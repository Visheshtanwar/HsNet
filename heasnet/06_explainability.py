"""HEASNet component: Grad-CAM, attention maps and faithfulness analysis.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== EXPLAINABLE AI ========================================
def _mm(c): return (c - c.amin((1, 2), keepdim=True)) / (c.amax((1, 2), keepdim=True) - c.amin((1, 2), keepdim=True) + 1e-8)
def _up(m): return F.interpolate(m, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[:, 0]

class XAI:
    """Grad-CAM on the last EfficientNetV2-S feature block + spatial maps of the SE / window / axial branches."""
    def __init__(self, model):
        self.m = model; self.st = {}; self.h = []
        for p in model.parameters(): p.requires_grad_(True)
        self.h.append(model.backbone[-1].register_forward_hook(lambda mod, i, o: self.st.__setitem__("act", o)))
        self.names = []
        for n in ["se", "window_attention", "axial_attention"]:
            mod = getattr(model, n)
            if not isinstance(mod, nn.Identity):
                self.names.append(n); self.h.append(mod.register_forward_hook(lambda mod_, i, o, nn_=n: self.st.__setitem__(nn_, o.detach())))
    def close(self):
        for h in self.h: h.remove()
    def run(self, x, target=None):
        self.m.eval(); x = x.to(DEVICE)
        with torch.enable_grad():
            lo = self.m(x); pred = lo.argmax(1); tg = pred if target is None else torch.as_tensor(target, device=DEVICE)
            act = self.st["act"]; g = torch.autograd.grad(lo.gather(1, tg[:, None]).sum(), act)[0]
        cam = F.relu((g.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True)).detach()
        cam = _mm(_up(cam)); maps = {n: _mm(_up(self.st[n].abs().mean(1, keepdim=True))) for n in self.names}
        return lo.detach(), cam.cpu().numpy(), {k: v.cpu().numpy() for k, v in maps.items()}

def _batch(ds, idx): return torch.stack([ds[i][0] for i in idx])

def gradcam_galleries(model, test_ds, probs, y, classes, out_dir):
    xai = XAI(model); pred = probs.argmax(1); K = len(classes)
    def collect(correct):
        sel = []
        for c in range(K):
            idx = np.where((y == c) & ((pred == c) if correct else (pred != c)))[0]
            if len(idx): sel.extend(idx[np.argsort(-probs[idx].max(1))][:GRADCAM_PER_CLASS].tolist())
        return sel
    for correct, fn, ttl in [(True, "Fig_18_gradcam_correct.png", "Grad-CAM, correctly classified test images"), (False, "Fig_19_gradcam_misclassified.png", "Grad-CAM, misclassified test images (explains the predicted class)")]:
        sel = collect(correct)
        if not sel: continue
        cols = 4; rows = int(math.ceil(len(sel) / cols)); fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.5), squeeze=False)
        for a in axes.ravel(): a.axis("off")
        for s in range(0, len(sel), 16):
            ch = sel[s:s + 16]; lo, cam, _ = xai.run(_batch(test_ds, ch))
            for k, i in enumerate(ch):
                a = axes[(s + k) // cols, (s + k) % cols]; a.imshow(load_rgb(test_ds.paths[i])); a.imshow(cam[k], cmap="jet", alpha=.4)
                p = int(lo[k].argmax()); a.set_title(f"True: {classes[y[i]]}\nPred: {classes[p]}", fontsize=8, color="green" if p == y[i] else "red")
        fig.suptitle(ttl, fontsize=13); fig.tight_layout(); savefig(fig, os.path.join(out_dir, fn))
    xai.close()

def attention_analysis(model, test_ds, probs, y, classes, out_dir):
    """Grad-CAM vs spatial maps of SE / window / axial branches: overlay figure + per-class Pearson correlation."""
    xai = XAI(model); pred = probs.argmax(1); K = len(classes); rows = []; names = xai.names
    lab = {"se": "SE", "window_attention": "WIN", "axial_attention": "AX"}
    ex = []
    for c in range(K):
        idx = np.where(y == c)[0][:CORR_MAX_PER_CLASS]; vals = {n: [] for n in names}
        for s in range(0, len(idx), 16):
            ch = idx[s:s + 16]; _, cam, mp = xai.run(_batch(test_ds, ch), target=torch.as_tensor(y[ch]))
            cf = cam.reshape(len(ch), -1); cf = (cf - cf.mean(1, keepdims=True)) / (cf.std(1, keepdims=True) + 1e-8)
            for n in names:
                mf = mp[n].reshape(len(ch), -1); mf = (mf - mf.mean(1, keepdims=True)) / (mf.std(1, keepdims=True) + 1e-8); vals[n].extend((cf * mf).mean(1).tolist())
        r = {"class": classes[c], "n_images": len(idx)}
        for n in names: v = np.array(vals[n]); r[f"{lab[n]}_corr_mean"] = float(v.mean()); r[f"{lab[n]}_corr_sd"] = float(v.std())
        rows.append(r)
        ok = np.where((y == c) & (pred == c))[0]
        if len(ok) and len(ex) < 8: ex.append(int(ok[0]))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(out_dir, "gradcam_vs_attention_branch_correlation.csv"), index=False)
    print("\nGRAD-CAM vs ATTENTION-BRANCH PEARSON CORRELATION (mean per class)"); print(d.round(3).to_string(index=False))
    cols = [f"{lab[n]}_corr_mean" for n in names]
    fig, ax = plt.subplots(figsize=(6.2, max(4.5, K * .45))); v = d[cols].values; lim = max(.3, np.nanmax(np.abs(v)))
    im = ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto"); ax.grid(False); ax.set_xticks(range(len(cols))); ax.set_xticklabels([lab[n] for n in names]); ax.set_yticks(range(K)); ax.set_yticklabels(classes)
    for i in range(K):
        for j in range(len(cols)): ax.text(j, i, f"{v[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pearson correlation: Grad-CAM vs attention-branch map"); fig.colorbar(im, fraction=.05); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_21_gradcam_attention_correlation_heatmap.png"))
    if ex:
        fig, axes = plt.subplots(len(ex), 2 + len(names), figsize=((2 + len(names)) * 2.8, len(ex) * 2.8), squeeze=False)
        for r_, i in enumerate(ex):
            _, cam, mp = xai.run(_batch(test_ds, [i]), target=torch.as_tensor([y[i]])); img = load_rgb(test_ds.paths[i])
            axes[r_, 0].imshow(img); axes[r_, 0].set_ylabel(classes[y[i]], fontsize=8); axes[r_, 1].imshow(img); axes[r_, 1].imshow(cam[0], cmap="jet", alpha=.45)
            for k, n in enumerate(names): axes[r_, 2 + k].imshow(img); axes[r_, 2 + k].imshow(mp[n][0], cmap="jet", alpha=.45)
            if r_ == 0:
                for k, t in enumerate(["Image", "Grad-CAM"] + [lab[n] + " branch" for n in names]): axes[0, k].set_title(t, fontsize=9)
        for a in axes.ravel(): a.set_xticks([]); a.set_yticks([]); a.grid(False)
        fig.suptitle("Grad-CAM and attention-branch maps (correct test images)"); fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_20_attention_branch_maps.png"))
    xai.close(); return d

@torch.no_grad()
def _prob_curve(model, x, order_rank, c, steps, mode):
    fr = np.linspace(0, 1, steps + 1); HW = IMG_SIZE * IMG_SIZE; imgs = []
    for f in fr:
        m = (order_rank < f * HW).reshape(1, 1, IMG_SIZE, IMG_SIZE).to(x.device)
        imgs.append(x * (~m) if mode == "deletion" else x * m)
    pr = torch.softmax(model(torch.cat(imgs)).float(), 1)[:, c].cpu().numpy(); return fr, pr

def faithfulness(model, test_ds, probs, y, classes, out_dir):
    """Deletion / insertion test of Grad-CAM (lower deletion AUC and higher insertion AUC = more faithful) against random maps."""
    xai = XAI(model); pred = probs.argmax(1); rows = []; curves = {k: [] for k in ["del_cam", "del_rand", "ins_cam", "ins_rand"]}
    g = torch.Generator().manual_seed(SEED)
    for c in range(len(classes)):
        idx = np.where((y == c) & (pred == c))[0][:FAITH_PER_CLASS]
        for i in idx:
            x = _batch(test_ds, [i]); _, cam, _ = xai.run(x); x = x.to(DEVICE); cam = torch.from_numpy(cam[0]).flatten()
            rk = lambda s: torch.argsort(torch.argsort(-s)).view(-1)
            r_cam = rk(cam); r_rand = rk(torch.rand(cam.numel(), generator=g)); out = {"class": classes[c]}
            for mode, tag in [("deletion", "del"), ("insertion", "ins")]:
                for nm, rr in [("cam", r_cam), ("rand", r_rand)]:
                    fr, pr = _prob_curve(model, x, rr, c, FAITH_STEPS, mode); out[f"{tag}_{nm}_auc"] = float(_trapz(pr, fr)); curves[f"{tag}_{nm}"].append(pr)
            rows.append(out)
    xai.close()
    if not rows: return None
    d = pd.DataFrame(rows); d.to_csv(os.path.join(out_dir, "faithfulness_deletion_insertion_per_image.csv"), index=False)
    s = d.groupby("class").mean(numeric_only=True); s.loc["ALL"] = d.mean(numeric_only=True); s.to_csv(os.path.join(out_dir, "faithfulness_deletion_insertion_summary.csv"))
    print("\nGRAD-CAM FAITHFULNESS (deletion AUC lower = better, insertion AUC higher = better; rand = random map)"); print(s.round(3).to_string())
    fr = np.linspace(0, 1, FAITH_STEPS + 1); fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, tag, ttl in [(axs[0], "del", "Deletion: remove most-salient pixels first"), (axs[1], "ins", "Insertion: reveal most-salient pixels first")]:
        for nm, col, lb in [("cam", C_TRAIN, "Grad-CAM"), ("rand", "#888", "random map")]:
            a = np.array(curves[f"{tag}_{nm}"]); ax.plot(fr, a.mean(0), color=col, lw=2, label=f"{lb} (AUC {_trapz(a.mean(0), fr):.3f})"); ax.fill_between(fr, a.mean(0) - a.std(0), a.mean(0) + a.std(0), color=col, alpha=.15)
        ax.set(xlabel="Fraction of pixels", ylabel="Probability of predicted class", title=ttl); ax.legend()
    fig.tight_layout(); savefig(fig, os.path.join(out_dir, "Fig_22_gradcam_faithfulness_deletion_insertion.png")); return s
