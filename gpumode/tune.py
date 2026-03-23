"""Grid-search over (num_blocks, BLOCK_SIZE) to find the fastest config."""
import torch
import time
import triton
import triton.language as tl

N = 1 << 27   # 128M elements — representative large size


@triton.jit
def _pass1(x_ptr, par_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid    = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK_SIZE
    acc    = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    base   = pid * BLOCK_SIZE
    while base < n:
        off = base + tl.arange(0, BLOCK_SIZE)
        acc += tl.load(x_ptr + off, mask=off < n, other=0.0).to(tl.float64)
        base += stride
    tl.store(par_ptr + pid, tl.sum(acc, axis=0))


@triton.jit
def _pass2(par_ptr, out_ptr, n_par, BLOCK_SIZE: tl.constexpr):
    off = tl.arange(0, BLOCK_SIZE)
    s   = tl.sum(tl.load(par_ptr + off, mask=off < n_par, other=0.0), axis=0)
    tl.store(out_ptr, s.to(tl.float32))


def bench(blocks, bs, x, warmup=5, iters=30):
    par = torch.empty(blocks, device="cuda", dtype=torch.float64)
    out = torch.empty(1,      device="cuda", dtype=torch.float32)
    p2_bs = triton.next_power_of_2(blocks)
    for _ in range(warmup):
        _pass1[(blocks,)](x, par, N, BLOCK_SIZE=bs)
        _pass2[(1,)](par, out, blocks, BLOCK_SIZE=p2_bs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _pass1[(blocks,)](x, par, N, BLOCK_SIZE=bs)
        _pass2[(1,)](par, out, blocks, BLOCK_SIZE=p2_bs)
    torch.cuda.synchronize()
    ms  = (time.perf_counter() - t0) / iters * 1000
    bw  = N * 4 / (ms / 1000) / 1e9
    ok  = int(out.item()) == N
    return ms, bw, ok


x = torch.ones(N, device="cuda", dtype=torch.float32)

print(f"{'blocks':>8} {'bs':>6} {'ms':>8} {'GB/s':>8} {'ok':>4}")
print("-" * 42)

best_bw, best_cfg = 0, None
for blocks in [512, 1024, 2048, 4096]:
    for bs in [512, 1024, 2048, 4096]:
        ms, bw, ok = bench(blocks, bs, x)
        marker = " ◀" if bw > best_bw else ""
        print(f"{blocks:>8} {bs:>6} {ms:>8.3f} {bw:>8.1f} {'✓' if ok else '✗':>4}{marker}")
        if bw > best_bw:
            best_bw, best_cfg = bw, (blocks, bs)

print("-" * 42)
print(f"Best: blocks={best_cfg[0]}, BLOCK_SIZE={best_cfg[1]}  →  {best_bw:.1f} GB/s")
