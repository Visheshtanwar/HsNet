"""HEASNet component: Training loop, optimizer and checkpoint handling.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== TRAINING ========================================
def make_optimizer(name, groups, wd=WEIGHT_DECAY):
    name = name.lower()
    if name == "adamw": return torch.optim.AdamW(groups, weight_decay=wd)
    if name == "adam": return torch.optim.Adam(groups, weight_decay=wd)
    if name == "sgd": return torch.optim.SGD(groups, momentum=0.9, weight_decay=wd)
    if name == "rmsprop": return torch.optim.RMSprop(groups, weight_decay=wd)
    raise ValueError(name)

def run_train_epoch(model, loader, crit, opt, scaler):
    model.train(); tl = tc = tn = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            lo = model(x); loss = crit(lo, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item() * y.size(0); tc += (lo.argmax(1) == y).sum().item(); tn += y.size(0)
    return tl / tn, tc / tn

@torch.no_grad()
def evaluate_epoch(model, loader, crit):
    model.eval(); tl = tc = tn = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            lo = model(x); loss = crit(lo, y)
        tl += loss.item() * y.size(0); tc += (lo.argmax(1) == y).sum().item(); tn += y.size(0)
    return tl / tn, tc / tn

def train_model(model, tr_loader, va_loader, out_dir, tag="HEASNet", lr_head=LR_HEAD, lr_bb=LR_BACKBONE, opt_name="adamw",
                epochs=EPOCHS, warmup=HEAD_WARMUP_EPOCHS, verbose=True, history_name="training_history.csv", ckpt_name=None):
    model.to(DEVICE); crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    best_path = os.path.join(out_dir, ckpt_name or f"best_{tag}.pt"); hist = []; best_acc, best_ep = -1.0, 0
    for p in model.backbone_parameters(): p.requires_grad = False
    head_opt = make_optimizer(opt_name, [{"params": model.head_parameters(), "lr": lr_head}])
    fine_opt = make_optimizer(opt_name, [{"params": model.backbone_parameters(), "lr": lr_bb}, {"params": model.head_parameters(), "lr": lr_head}])
    if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
    for ep in range(1, epochs + 1):
        if ep == warmup + 1:
            for p in model.backbone_parameters(): p.requires_grad = True
        opt = head_opt if ep <= warmup else fine_opt; t0 = time.time()
        trl, tra = run_train_epoch(model, tr_loader, crit, opt, scaler); val, vaa = evaluate_epoch(model, va_loader, crit)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        sec = time.time() - t0
        hist.append({"epoch": ep, "epoch_seconds": sec, "train_loss": trl, "train_accuracy": tra, "val_loss": val, "val_accuracy": vaa,
                     "learning_rate_head": lr_head, "learning_rate_backbone": 0.0 if ep <= warmup else lr_bb})
        if vaa > best_acc: best_acc, best_ep = vaa, ep; torch.save(model.state_dict(), best_path)
        if verbose or ep == epochs:
            print(f"[{tag}] Epoch {ep:02d}/{epochs} | Train loss {trl:.4f} acc {tra:.4f} | Val loss {val:.4f} acc {vaa:.4f} | Best val {best_acc:.4f} (epoch {best_ep}) | {sec:.1f}s")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    h = pd.DataFrame(hist); h.to_csv(os.path.join(out_dir, history_name), index=False)
    peak = float(torch.cuda.max_memory_allocated() / 1e6) if DEVICE.type == "cuda" else None
    return model, h, best_ep, best_acc, best_path, peak
