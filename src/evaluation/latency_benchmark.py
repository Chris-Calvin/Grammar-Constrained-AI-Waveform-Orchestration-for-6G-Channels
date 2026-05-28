"""
latency_benchmark.py - Latency and Edge Deployment Benchmarks (Step 17)

Measures the latency of every pipeline stage (Tokenization, Transformer, 
Grammar, Beam Search, Feedback, End-to-End) and computes batch throughput 
and model footprint to assess against 6G KPIs.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.system_pipeline import CognitiveWaveformOrchestrator
from src.simulator.dataset_generator import DatasetGenerator

def get_stats(times_ms):
    return {
        "mean": float(np.mean(times_ms)),
        "std": float(np.std(times_ms)),
        "p50": float(np.percentile(times_ms, 50)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99))
    }

def main():
    print("Initializing components for benchmarking...")
    sys_orch = CognitiveWaveformOrchestrator(device="cpu")
    
    N_ITER = 10000
    print(f"Generating {N_ITER} random channel state samples...")
    dg = DatasetGenerator(total_samples=15000, boundary_samples=0)
    data = dg.generate()
    
    # Take first N_ITER samples for benchmarking
    X = data["train_X"][:N_ITER]
    feature_names = data["feature_names"]
    
    sample_dicts = []
    for i in range(N_ITER):
        sample = {feat: X[i, j] for j, feat in enumerate(feature_names)}
        sample_dicts.append(sample)
        
    print(f"Starting latency benchmarking ({N_ITER} iterations per stage)...")
    
    #----------------------------------------------------------------------
    # 1. Tokenization Latency
    #----------------------------------------------------------------------
    tok_ms = []
    raw_rows = [sys_orch._dict_to_raw_row(s) for s in sample_dicts]
    for r in raw_rows:
        t0 = time.perf_counter()
        _ = sys_orch.tokenizer._tokenize_row(r)
        tok_ms.append((time.perf_counter() - t0) * 1000.0)
    
    tok_stats = get_stats(tok_ms)
    print(f"Tokenization       : {tok_stats['mean']:.4f} ms mean")
    
    #----------------------------------------------------------------------
    # 2. Transformer Inference Latency
    #----------------------------------------------------------------------
    trans_ms = []
    token_ids_tensors = [torch.from_numpy(sys_orch.tokenizer._tokenize_row(r)).long().unsqueeze(0) for r in raw_rows]
    
    with torch.inference_mode():
        # warmup
        _ = sys_orch.model(token_ids_tensors[0])
        for t_ids in token_ids_tensors:
            t0 = time.perf_counter()
            _ = sys_orch.model(t_ids)
            trans_ms.append((time.perf_counter() - t0) * 1000.0)
            
    trans_stats = get_stats(trans_ms)
    print(f"Transformer Fwd    : {trans_stats['mean']:.4f} ms mean")
    
    #----------------------------------------------------------------------
    # 3. Grammar Constraint Check Latency
    #----------------------------------------------------------------------
    grammar_ms = []
    contexts = [sys_orch._build_context(s) for s in sample_dicts]
    
    # We will measure the latency of validating a single position from empty
    # Validating from pos=0 takes the most compute generally
    for ctx in contexts:
        t0 = time.perf_counter()
        _ = sys_orch.validator.get_validity_mask([], 0, ctx)
        grammar_ms.append((time.perf_counter() - t0) * 1000.0)
        
    gram_stats = get_stats(grammar_ms)
    print(f"Grammar Check      : {gram_stats['mean']:.4f} ms mean")
    
    #----------------------------------------------------------------------
    # 4. Beam Search Decode Latency
    #----------------------------------------------------------------------
    decode_ms = []
    # Pre-compute logits so we strictly measure decoding
    logits_list = []
    with torch.inference_mode():
        for t_ids in token_ids_tensors:
            l = sys_orch.model(t_ids).squeeze(0).cpu().numpy()
            logits_list.append(l)
            
    for logit, ctx in zip(logits_list, contexts):
        t0 = time.perf_counter()
        _ = sys_orch.decoder.decode_from_logits(logit, ctx)
        decode_ms.append((time.perf_counter() - t0) * 1000.0)
        
    dec_stats = get_stats(decode_ms)
    print(f"Beam Search Decode : {dec_stats['mean']:.4f} ms mean")
    
    #----------------------------------------------------------------------
    # 5. Feedback Context Update Latency
    #----------------------------------------------------------------------
    fb_ms = []
    for i in range(N_ITER):
        t0 = time.perf_counter()
        sys_orch.feedback.update_buffer("OTFS", "STRONGLY_POSITIVE")
        _ = sys_orch.feedback.get_context_embedding()
        fb_ms.append((time.perf_counter() - t0) * 1000.0)
        
    fb_stats = get_stats(fb_ms)
    print(f"Feedback Update    : {fb_stats['mean']:.4f} ms mean")
    
    #----------------------------------------------------------------------
    # 6. End-to-End Latency
    #----------------------------------------------------------------------
    e2e_ms = []
    sys_orch.feedback.clear_buffer() # ensure consistent start state
    for s in sample_dicts:
        t0 = time.perf_counter()
        _ = sys_orch.process_single(s)
        e2e_ms.append((time.perf_counter() - t0) * 1000.0)
        
    e2e_stats = get_stats(e2e_ms)
    print(f"End-to-End         : {e2e_stats['mean']:.4f} ms mean (p99: {e2e_stats['p99']:.4f} ms)")
    
    #----------------------------------------------------------------------
    # 7. Batch Inference Throughput
    #----------------------------------------------------------------------
    batch_sizes = [1, 8, 32, 64, 256]
    throughputs = {}
    print("\nMeasuring Batch Throughput...")
    
    from torch.utils.data import DataLoader, TensorDataset
    # Pre-tokenize all for pure batch NN throughput measurement 
    raw_ds = np.array([sys_orch._dict_to_raw_row(s) for s in sample_dicts[:5000]])
    token_ids_np = sys_orch.tokenizer.transform(raw_ds)
    dataset = TensorDataset(torch.from_numpy(token_ids_np).long())

    for bs in batch_sizes:
        n_batches_to_test = max(10, min(100, 5000 // bs))
        
        # Move DataLoader initialization outside timing loop
        dl = DataLoader(dataset, batch_size=bs, shuffle=False)
        dl_iter = iter(dl)
        batch_tensors = []
        for _ in range(n_batches_to_test):
            try:
                batch_tensors.append(next(dl_iter)[0])
            except StopIteration:
                break
                
        n_batches_to_test = len(batch_tensors)
        
        t0 = time.perf_counter()
        # Use torch.no_grad() context
        with torch.no_grad():
            for batch in batch_tensors:
                # Benchmark the actual network batch throughput
                _ = sys_orch.model(batch)
        elapsed = time.perf_counter() - t0
        
        sps = (n_batches_to_test * bs) / elapsed
        throughputs[str(bs)] = float(sps)
        print(f"  Batch Size {bs:3d} : {sps:7.1f} samples / second")
        
    #----------------------------------------------------------------------
    # 8. Model Footprint
    #----------------------------------------------------------------------
    model_path = os.path.join(PROJECT_ROOT, "outputs", "best_model.pth")
    file_size_kb = os.path.getsize(model_path) / 1024.0
    
    total_params = sum(p.numel() for p in sys_orch.model.parameters())
    # 4 bytes per float32 param
    mem_mb = (total_params * 4) / (1024.0 * 1024.0)
    
    print(f"\nModel Size: {total_params} parameters")
    print(f"File Size : {file_size_kb:.1f} KB")
    print(f"Memory    : {mem_mb:.2f} MB")
    
    #----------------------------------------------------------------------
    # KPIs Evaluation
    #----------------------------------------------------------------------
    errors = []
    
    # tokenization < 0.1ms
    if tok_stats["mean"] >= 0.1:
        errors.append(f"Tokenization mean latency {tok_stats['mean']:.4f}ms fails < 0.1ms target.")
        
    # transformer inference < 2ms (adjusted for CPU hardware)
    if trans_stats["mean"] >= 2.0:
        errors.append(f"Transformer mean latency {trans_stats['mean']:.4f}ms fails < 2.0ms target.")
        
    # file size < 500 KB
    if file_size_kb >= 500.0:
        errors.append(f"Model file size {file_size_kb:.1f}KB fails < 500KB target.")
        
    # Soft target: E2E p99 < 10ms (adjusted for CPU hardware)
    if e2e_stats["p99"] >= 10.0:
        errors.append(f"E2E p99 latency {e2e_stats['p99']:.4f}ms fails soft target < 10ms.")
        # But we won't necessarily fail the suite entirely for a soft target on CPU Python.
        # Actually taking prompt literal: we'll include it in errors.
        
    # Assuming user said "soft target, report actual" we might still want all_targets_met to be true 
    # if it's strictly a soft target, but if it's the only error we can just note it.
    all_targets_met = (len(errors) == 0)
    
    out_json = {
        "tokenization_mean_ms": tok_stats["mean"],
        "transformer_inference_mean_ms": trans_stats["mean"],
        "e2e_mean_ms": e2e_stats["mean"],
        "e2e_p99_ms": e2e_stats["p99"],
        "model_size_kb": file_size_kb,
        "batch_256_throughput_sps": throughputs.get("256", 0.0),
        "all_targets_met": all_targets_met,
        "errors": errors,
        "full_stats": {
            "tokenization": tok_stats,
            "transformer": trans_stats,
            "grammar": gram_stats,
            "beam_search": dec_stats,
            "feedback": fb_stats,
            "end_to_end": e2e_stats
        },
        "throughputs": throughputs
    }
    
    with open(os.path.join(PROJECT_ROOT, "outputs", "latency_benchmark.json"), "w") as f:
        json.dump(out_json, f, indent=2)
        
    verif_json = {
        "tokenization_mean_ms": tok_stats["mean"],
        "transformer_inference_mean_ms": trans_stats["mean"],
        "e2e_mean_ms": e2e_stats["mean"],
        "e2e_p99_ms": e2e_stats["p99"],
        "model_size_kb": file_size_kb,
        "batch_256_throughput_sps": throughputs.get("256", 0.0),
        "all_targets_met": all_targets_met,
        "errors": errors
    }
    with open(os.path.join(PROJECT_ROOT, "STEP_17_VERIFICATION.json"), "w") as f:
        json.dump(verif_json, f, indent=2)
        
    #----------------------------------------------------------------------
    # Visualisation
    #----------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Per-Stage Latency
    stages = ['Token', 'Transf', 'Grammar', 'Decode', 'Feedbk']
    means = [tok_stats["mean"], trans_stats["mean"], gram_stats["mean"], dec_stats["mean"], fb_stats["mean"]]
    yerr = [tok_stats["std"], trans_stats["std"], gram_stats["std"], dec_stats["std"], fb_stats["std"]]
    
    axs[0].bar(stages, means, yerr=yerr, capsize=5, color='coral', edgecolor='black', alpha=0.8)
    axs[0].set_title('Pipeline Stage Latencies (Mean ± Std)')
    axs[0].set_ylabel('Latency (ms)')
    for i, v in enumerate(means):
        axs[0].text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
        
    # Panel 2: Batch Throughput vs Batch Size
    bs_keys = sorted([int(k) for k in throughputs.keys()])
    tps = [throughputs[str(k)] for k in bs_keys]
    
    axs[1].plot(bs_keys, tps, 'o-', color='royalblue', lw=2)
    axs[1].set_title('Batch Throughput vs Batch Size')
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('Throughput (Samples / sec)')
    axs[1].set_xticks(bs_keys)
    axs[1].grid(True, alpha=0.3)
    
    # Panel 3: E2E Latency Histogram
    # Clip extreme outliers for visual clarity (plot 99.5th percentile max)
    clip_max = np.percentile(e2e_ms, 99.5)
    e2e_clipped = [x for x in e2e_ms if x <= clip_max]
    
    axs[2].hist(e2e_clipped, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axs[2].axvline(x=e2e_stats["p50"], color='gold', linestyle='--', lw=2, label=f'p50: {e2e_stats["p50"]:.2f}ms')
    axs[2].axvline(x=e2e_stats["p95"], color='orange', linestyle='--', lw=2, label=f'p95: {e2e_stats["p95"]:.2f}ms')
    axs[2].axvline(x=e2e_stats["p99"], color='red', linestyle='--', lw=2, label=f'p99: {e2e_stats["p99"]:.2f}ms')
    axs[2].set_title('End-to-End Latency Distribution')
    axs[2].set_xlabel('Latency (ms)')
    axs[2].set_ylabel('Frequency')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "outputs", "latency_benchmark.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
