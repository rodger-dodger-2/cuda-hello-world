"""
Benchmark runner: correctness check + timing across several input sizes.
Run with:  python benchmark.py
"""
import torch
import time
from reference import ref_kernel, generate_input, check_implementation
from solution import solution_kernel

SIZES  = [2**20, 2**24, 2**26, 2**27]   # 1M → 128M elements
SEEDS  = [0, 1, 42]
WARMUP = 5
ITERS  = 20


def bench(fn, data, iters=ITERS, warmup=WARMUP):
    # warmup
    for _ in range(warmup):
        fn(data)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(data)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000   # ms


print("=" * 62)
print(f"{'Size':>12}  {'Correct':>7}  {'Ref ms':>8}  {'Sol ms':>8}  {'Speedup':>8}")
print("=" * 62)

for size in SIZES:
    data = generate_input(size, seed=0)

    # correctness over multiple seeds
    all_ok = all(check_implementation(solution_kernel, generate_input(size, s)) for s in SEEDS)

    ref_ms = bench(ref_kernel,      generate_input(size, 0))
    sol_ms = bench(solution_kernel, generate_input(size, 0))
    bw_gb  = size * 4 / (sol_ms / 1000) / 1e9   # float32 read bandwidth

    print(f"{size:>12,}  {'✓' if all_ok else '✗':>7}  {ref_ms:>8.3f}  {sol_ms:>8.3f}"
          f"  {ref_ms/sol_ms:>7.2f}x  ({bw_gb:.0f} GB/s)")

print("=" * 62)
