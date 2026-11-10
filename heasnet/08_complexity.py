"""HEASNet component: Model complexity and inference benchmarking.

Extracted from the supplied final experiment notebook; execution order is defined in run_experiment.py.
"""

# ======================================== COMPLEXITY AND SPEED ========================================
def compute_complexity(model, out_dir, tag="HEASNet", save=True):
    model.eval().to(DEVICE); x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    tot = sum(p.numel() for p in model.parameters()); res = {"model": tag, "parameters_total": int(tot), "parameters_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "parameters_million": tot / 1e6, "weights_fp32_MB": tot * 4 / 1e6, "weights_fp16_MB": tot * 2 / 1e6, "input_shape": [1, 3, IMG_SIZE, IMG_SIZE], "gmacs": None, "gflops": None, "flop_note": ""}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc: model(x)
        fl = fc.get_total_flops(); res["gflops"] = fl / 1e9; res["gmacs"] = fl / 2e9
        res["flop_note"] = "torch.utils.flop_counter: counts matmul/conv/attention multiply-adds (1 MAC = 2 FLOPs); element-wise ops, norms and activations are not counted."
        if isinstance(model, HEASNet):
            br = []
            with torch.no_grad():
                def cnt(f, inp):
                    with FlopCounterMode(display=False) as c: o = f(inp)
                    return o, c.get_total_flops()
                h, f_ = cnt(model.backbone, x); br.append(("EfficientNetV2-S backbone", model.backbone, f_))
                h, f_ = cnt(model.projection, h); br.append(("1x1 projection", model.projection, f_))
                for nm, mod in [("SE", model.se), ("Window attention", model.window_attention), ("Axial attention (height)", model.axial_attention)]:
                    h, f_ = cnt(mod, h); br.append((nm, mod, f_))
                _, f_ = cnt(model.classifier, h.mean((2, 3))); br.append(("Classifier", model.classifier, f_))
            d = pd.DataFrame([{"module": n, "parameters": sum(p.numel() for p in m.parameters()), "GMACs": f / 2e9} for n, m, f in br]); d["percent_of_GMACs"] = 100 * d.GMACs / d.GMACs.sum()
            if save: d.to_csv(os.path.join(out_dir, f"complexity_breakdown_{tag}.csv"), index=False)
            print(f"\nCOMPLEXITY BREAKDOWN {tag}"); print(d.round(4).to_string(index=False)); res["breakdown"] = d.to_dict("records")
    except Exception as e:
        res["flop_note"] = f"FLOP counter failed: {e!r}"
        try:
            from fvcore.nn import FlopCountAnalysis
            fl = FlopCountAnalysis(model, x); res["gmacs"] = float(fl.total() / 1e9); res["gflops"] = 2 * res["gmacs"]; res["flop_note"] = "fvcore (MAC count)"
        except Exception as e2: res["flop_note"] += f" | fvcore unavailable: {e2!r}"
    if save: json.dump(res, open(os.path.join(out_dir, f"model_complexity_{tag}.json"), "w"), indent=2, default=str)
    print(f"{tag}: params {res['parameters_million']:.3f} M | fp32 {res['weights_fp32_MB']:.1f} MB | GMACs {res['gmacs']} | GFLOPs {res['gflops']}")
    return res

def _timeit(fn, repeats, warmup):
    for _ in range(warmup): fn()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)

def benchmark_speed(model, out_dir, tag="HEASNet", save=True):
    rep, wu = (60, 15) if not QUICK else (3, 1); rows = []; model.eval().to(DEVICE)
    for amp in ([False, True] if DEVICE.type == "cuda" else [False]):
        for bs in [1, 8, 16, 32]:
            x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
            if DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
            def f():
                with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, enabled=amp and DEVICE.type == "cuda"): model(x)
            try: ts = _timeit(f, rep, wu)
            except RuntimeError as e: print("benchmark skipped", bs, e); continue
            rows.append({"model": tag, "device": str(DEVICE), "precision": "AMP fp16" if amp else "fp32", "batch_size": bs, "mean_ms_per_batch": ts.mean(), "median_ms_per_batch": np.median(ts),
                         "std_ms": ts.std(), "p95_ms": np.percentile(ts, 95), "mean_ms_per_image": ts.mean() / bs, "throughput_images_per_s": 1000 * bs / ts.mean(),
                         "peak_gpu_MB_inference": float(torch.cuda.max_memory_allocated() / 1e6) if DEVICE.type == "cuda" else None})
    try:
        cm_ = copy.deepcopy(model).cpu().float().eval(); x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        def fc_():
            with torch.inference_mode(): cm_(x)
        ts = _timeit(fc_, 3 if QUICK else 20, 1)
        rows.append({"model": tag, "device": f"CPU ({torch.get_num_threads()} threads)", "precision": "fp32", "batch_size": 1, "mean_ms_per_batch": ts.mean(), "median_ms_per_batch": np.median(ts), "std_ms": ts.std(),
                     "p95_ms": np.percentile(ts, 95), "mean_ms_per_image": ts.mean(), "throughput_images_per_s": 1000 / ts.mean(), "peak_gpu_MB_inference": None}); del cm_
    except Exception as e: print("CPU benchmark skipped:", e)
    d = pd.DataFrame(rows)
    if save: d.to_csv(os.path.join(out_dir, f"inference_benchmark_{tag}.csv"), index=False)
    print(f"\nINFERENCE SPEED {tag}"); print(d.round(3).to_string(index=False)); return d

def plot_efficiency(comp, bench, out_dir, ds):
    if "breakdown" not in comp: return
    d = pd.DataFrame(comp["breakdown"]); fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    axs[0].barh(d.module, d.GMACs, color=C_TRAIN); axs[0].invert_yaxis(); axs[0].set_xlabel("GMACs"); axs[0].set_title("Compute per module (224x224)")
    axs[1].barh(d.module, d.parameters / 1e6, color=C_VAL); axs[1].invert_yaxis(); axs[1].set_xlabel("Parameters (M)"); axs[1].set_title("Parameters per module")
    g = bench[bench.device == str(DEVICE)] if len(bench) else bench
    for pr, col in [("fp32", C_TRAIN), ("AMP fp16", C_TEST)]:
        s = g[g.precision == pr]
        if len(s): axs[2].plot(s.batch_size, s.throughput_images_per_s, "o-", color=col, label=pr)
    axs[2].set(xlabel="Batch size", ylabel="Images per second", title="Throughput"); axs[2].legend(); fig.suptitle(f"{ds}: model complexity and speed"); fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "Fig_24_complexity_and_speed.png"))
