#!/usr/bin/env python3
"""
PixelGen Speedup Benchmark

Benchmarks various speedup techniques for image generation:
1. Fewer sampling steps (5, 10, 25, 50)
2. Adaptive step scheduling
3. Token merging (ToMe)
4. torch.compile
5. Mixed precision (bfloat16)
6. Batching

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_speedup.py
"""

import torch
import torch.nn as nn
import time
import gc
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_function(fn, name, num_warmup=3, num_runs=10):
    """Benchmark a function with warmup."""
    # Warmup
    for _ in range(num_warmup):
        _ = fn()
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = fn()
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5

    return {
        'name': name,
        'avg_ms': avg * 1000,
        'std_ms': std * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
    }


def main():
    print("=" * 70)
    print("PIXELGEN SPEEDUP BENCHMARK")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\nLoading JiT-L/16 model...")
    from src.models.transformer.JiT import JiT_L_16

    model = JiT_L_16(input_size=256, num_classes=1000)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Test inputs
    batch_size = 1
    x = torch.randn(batch_size, 3, 256, 256, device=device, dtype=torch.float32)
    t = torch.tensor([0.5], device=device)
    y = torch.tensor([42], device=device)

    print("\n" + "=" * 70)
    print("1. SAMPLING STEPS COMPARISON")
    print("=" * 70)

    from src.speedup import FastEulerSampler

    step_counts = [5, 10, 25, 50]
    noise = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    condition = torch.tensor([42], device=device)

    baseline_time = None

    for steps in step_counts:
        from src.speedup.fast_sampler import SamplerConfig
        config = SamplerConfig(num_steps=steps, adaptive_steps=True)
        sampler = FastEulerSampler(config)

        def run_sample():
            return sampler.sample(model, noise.clone(), condition)

        result = benchmark_function(run_sample, f"{steps} steps", num_warmup=2, num_runs=5)

        if steps == 50:
            baseline_time = result['avg_ms']

        speedup = baseline_time / result['avg_ms'] if baseline_time else 1.0
        print(f"  {steps:3d} steps: {result['avg_ms']:7.1f} ms ± {result['std_ms']:5.1f} ms  "
              f"(speedup: {speedup:.2f}x)")

    print("\n" + "=" * 70)
    print("2. ADAPTIVE vs LINEAR SCHEDULING")
    print("=" * 70)

    from src.speedup.fast_sampler import SamplerConfig

    for steps in [25]:
        # Adaptive (cosine)
        config_adaptive = SamplerConfig(num_steps=steps, adaptive_steps=True)
        sampler_adaptive = FastEulerSampler(config_adaptive)

        # Linear
        config_linear = SamplerConfig(num_steps=steps, adaptive_steps=False)
        sampler_linear = FastEulerSampler(config_linear)

        result_adaptive = benchmark_function(
            lambda: sampler_adaptive.sample(model, noise.clone(), condition),
            f"Adaptive {steps} steps", num_warmup=2, num_runs=5
        )

        result_linear = benchmark_function(
            lambda: sampler_linear.sample(model, noise.clone(), condition),
            f"Linear {steps} steps", num_warmup=2, num_runs=5
        )

        print(f"  Adaptive ({steps} steps): {result_adaptive['avg_ms']:.1f} ms")
        print(f"  Linear ({steps} steps):   {result_linear['avg_ms']:.1f} ms")
        print(f"  Note: Same time, but adaptive concentrates steps where quality matters")

    print("\n" + "=" * 70)
    print("3. HEUN (2nd ORDER) vs EULER")
    print("=" * 70)

    for steps in [10, 25]:
        config = SamplerConfig(num_steps=steps)
        sampler = FastEulerSampler(config)

        result_euler = benchmark_function(
            lambda: sampler.sample(model, noise.clone(), condition),
            f"Euler {steps} steps", num_warmup=2, num_runs=5
        )

        result_heun = benchmark_function(
            lambda: sampler.sample_heun(model, noise.clone(), condition),
            f"Heun {steps} steps", num_warmup=2, num_runs=5
        )

        print(f"  Euler ({steps} steps): {result_euler['avg_ms']:.1f} ms")
        print(f"  Heun ({steps} steps):  {result_heun['avg_ms']:.1f} ms (2x NFE, better quality)")

    print("\n" + "=" * 70)
    print("4. BATCHING THROUGHPUT")
    print("=" * 70)

    batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        try:
            noise_batch = torch.randn(bs, 3, 256, 256, device=device, dtype=torch.float32)
            condition_batch = torch.randint(0, 1000, (bs,), device=device)

            config = SamplerConfig(num_steps=25)
            sampler = FastEulerSampler(config)

            def run_batch():
                return sampler.sample(model, noise_batch.clone(), condition_batch)

            result = benchmark_function(run_batch, f"batch={bs}", num_warmup=2, num_runs=3)
            throughput = bs / (result['avg_ms'] / 1000)
            print(f"  Batch {bs}: {result['avg_ms']:.1f} ms total, "
                  f"{result['avg_ms']/bs:.1f} ms/image, {throughput:.2f} images/sec")

            del noise_batch, condition_batch
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"  Batch {bs}: OOM")
            break

    print("\n" + "=" * 70)
    print("5. PRECISION COMPARISON")
    print("=" * 70)

    precisions = [
        ('float32', torch.float32),
        ('float16', torch.float16),
        ('bfloat16', torch.bfloat16),
    ]

    config = SamplerConfig(num_steps=25)

    for name, dtype in precisions:
        try:
            model_prec = model.to(dtype=dtype)
            noise_prec = noise.to(dtype=dtype)

            sampler = FastEulerSampler(config)

            def run_prec():
                return sampler.sample(model_prec, noise_prec.clone(), condition)

            result = benchmark_function(run_prec, name, num_warmup=2, num_runs=5)
            mem = get_gpu_memory_mb()
            print(f"  {name:8s}: {result['avg_ms']:.1f} ms, memory: {mem:.0f} MB")

        except Exception as e:
            print(f"  {name:8s}: Error - {e}")

    # Reset to float32
    model = model.to(dtype=torch.float32)

    print("\n" + "=" * 70)
    print("6. TORCH.COMPILE")
    print("=" * 70)

    try:
        # Baseline
        config = SamplerConfig(num_steps=25)
        sampler = FastEulerSampler(config)

        result_base = benchmark_function(
            lambda: sampler.sample(model, noise.clone(), condition),
            "No compile", num_warmup=2, num_runs=5
        )

        # Compiled
        print("  Compiling model (this may take a moment)...")
        model_compiled = torch.compile(model, mode='reduce-overhead')

        # Longer warmup for compiled model
        result_compiled = benchmark_function(
            lambda: sampler.sample(model_compiled, noise.clone(), condition),
            "torch.compile", num_warmup=5, num_runs=5
        )

        speedup = result_base['avg_ms'] / result_compiled['avg_ms']
        print(f"  Without compile: {result_base['avg_ms']:.1f} ms")
        print(f"  With compile:    {result_compiled['avg_ms']:.1f} ms ({speedup:.2f}x speedup)")

    except Exception as e:
        print(f"  torch.compile failed: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY: RECOMMENDED CONFIGURATION")
    print("=" * 70)
    print("""
    For fastest generation (with quality trade-off):
    - Use 5-10 steps with adaptive scheduling
    - Use bfloat16 precision
    - Batch multiple images together
    - Use torch.compile if generating many images

    For best quality:
    - Use 25-50 steps with Heun solver
    - Use adaptive step scheduling
    - Use float32 precision

    Balanced (recommended default):
    - 25 steps, Euler solver, adaptive scheduling
    - bfloat16 precision
    - Batch size based on available VRAM
    """)

    print("\n" + "=" * 70)
    print("ESTIMATED GENERATION TIMES (256x256)")
    print("=" * 70)
    print("""
    Configuration                      | Time/Image | Quality
    -----------------------------------|------------|--------
    50 steps, float32, Heun            | ~2.0 sec   | Best
    25 steps, float32, Euler           | ~0.5 sec   | Very Good
    25 steps, bfloat16, Euler          | ~0.2 sec   | Very Good
    10 steps, bfloat16, Euler          | ~0.1 sec   | Good
    5 steps, bfloat16, Euler           | ~0.05 sec  | Acceptable
    5 steps, bfloat16, batch=8         | ~0.02 sec  | Acceptable (throughput)
    """)


if __name__ == '__main__':
    main()
