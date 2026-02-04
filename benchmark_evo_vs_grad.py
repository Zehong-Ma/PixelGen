#!/usr/bin/env python3
"""
Benchmark: Evolution vs Gradient-Based Training

Compares memory usage and throughput between:
1. Traditional gradient-based training (backprop)
2. Evolutionary optimization (fitness-based)

This demonstrates the key advantage of evolution: no gradient storage.
"""

import torch
import torch.nn as nn
import gc
import time
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_memory_reserved_mb():
    """Get reserved GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024 / 1024
    return 0


def benchmark_gradient_training(model, batch_size=4, num_iters=10):
    """
    Benchmark standard gradient-based training.

    This requires:
    - Forward pass activations (for backward)
    - Gradient storage (same size as parameters)
    - Optimizer state (2x parameters for AdamW)
    """
    print("\n" + "="*60)
    print("GRADIENT-BASED TRAINING BENCHMARK")
    print("="*60)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Create optimizer (AdamW stores 2 tensors per parameter)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy batch
    x = torch.randn(batch_size, 3, 256, 256, device=device, dtype=dtype)
    t = torch.rand(batch_size, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    mem_before = get_gpu_memory_mb()
    print(f"Memory before: {mem_before:.1f} MB")

    # Training loop
    times = []
    for i in range(num_iters):
        start = time.time()

        optimizer.zero_grad()

        # Forward pass
        pred = model(x, t, y)

        # Compute loss (MSE for simplicity)
        loss = ((pred - x) ** 2).mean()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        torch.cuda.synchronize()
        times.append(time.time() - start)

    mem_after = get_gpu_memory_mb()
    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"Memory after: {mem_after:.1f} MB")
    print(f"Peak memory: {mem_peak:.1f} MB")
    print(f"Memory delta: {mem_after - mem_before:.1f} MB")
    print(f"Avg iteration time: {sum(times)/len(times)*1000:.1f} ms")
    print(f"Throughput: {batch_size / (sum(times)/len(times)):.1f} samples/sec")

    return {
        'mem_before': mem_before,
        'mem_after': mem_after,
        'mem_peak': mem_peak,
        'avg_time': sum(times)/len(times),
    }


def benchmark_evolution(model, batch_size=4, num_iters=10):
    """
    Benchmark evolutionary optimization.

    This requires:
    - Forward pass only (no backward)
    - No gradient storage
    - No optimizer state
    """
    print("\n" + "="*60)
    print("EVOLUTIONARY OPTIMIZATION BENCHMARK")
    print("="*60)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Create dummy batch
    x = torch.randn(batch_size, 3, 256, 256, device=device, dtype=dtype)
    t = torch.rand(batch_size, device=device)
    y = torch.randint(0, 1000, (batch_size,), device=device)

    # Disable gradients globally
    torch.set_grad_enabled(False)

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    mem_before = get_gpu_memory_mb()
    print(f"Memory before: {mem_before:.1f} MB")

    # Evolution loop (simulating antithetic pair evaluation)
    times = []
    for i in range(num_iters):
        start = time.time()

        # Forward pass for positive perturbation
        pred_pos = model(x, t, y)

        # Compute fitness (no backward!)
        fitness_pos = -((pred_pos - x) ** 2).mean().item()

        # Forward pass for negative perturbation
        pred_neg = model(x, t, y)

        # Compute fitness
        fitness_neg = -((pred_neg - x) ** 2).mean().item()

        # Compare and vote (simple comparison)
        vote = 1 if fitness_pos > fitness_neg else -1

        torch.cuda.synchronize()
        times.append(time.time() - start)

    mem_after = get_gpu_memory_mb()
    mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"Memory after: {mem_after:.1f} MB")
    print(f"Peak memory: {mem_peak:.1f} MB")
    print(f"Memory delta: {mem_after - mem_before:.1f} MB")
    print(f"Avg iteration time: {sum(times)/len(times)*1000:.1f} ms")
    print(f"Throughput: {2 * batch_size / (sum(times)/len(times)):.1f} samples/sec (2 evals/iter)")

    # Re-enable gradients
    torch.set_grad_enabled(True)

    return {
        'mem_before': mem_before,
        'mem_after': mem_after,
        'mem_peak': mem_peak,
        'avg_time': sum(times)/len(times),
    }


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("="*60)
    print("PIXELGEN: EVOLUTION vs GRADIENT BENCHMARK")
    print("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU (results may not be representative)")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create model
    print("\nLoading JiT model...")
    from src.models.transformer.JiT import JiT_L_16

    model = JiT_L_16(input_size=256, num_classes=1000)
    model = model.to(device=device, dtype=torch.float32)  # Use float32 for compatibility

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Memory for parameters alone
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"Parameter memory: {param_memory:.1f} MB")

    # Run benchmarks
    batch_size = 4

    # Gradient-based training
    model.train()
    grad_results = benchmark_gradient_training(model.clone() if hasattr(model, 'clone') else model,
                                                batch_size=batch_size)

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model for evolution benchmark
    model = JiT_L_16(input_size=256, num_classes=1000)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    # Evolution-based training
    evo_results = benchmark_evolution(model, batch_size=batch_size)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Gradient':<15} {'Evolution':<15} {'Savings':<15}")
    print("-"*60)

    peak_grad = grad_results['mem_peak']
    peak_evo = evo_results['mem_peak']
    savings_pct = (peak_grad - peak_evo) / peak_grad * 100

    print(f"{'Peak Memory (MB)':<25} {peak_grad:<15.1f} {peak_evo:<15.1f} {savings_pct:.1f}%")
    print(f"{'Time per iter (ms)':<25} {grad_results['avg_time']*1000:<15.1f} {evo_results['avg_time']*1000:<15.1f}")

    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print(f"""
Evolution saves ~{savings_pct:.0f}% peak memory by eliminating:
- Gradient storage ({param_memory:.1f} MB)
- Optimizer state (~{2*param_memory:.1f} MB for AdamW)
- Backward pass activations

Trade-off: Evolution requires 2 forward passes per perturbation pair,
but no backward pass. For large models, forward << backward in cost.
""")


if __name__ == '__main__':
    main()
