# CUDA Hello World

A simple CUDA program that prints GPU device info and runs a kernel across multiple threads.

## Run

```sh
bash run.sh
```

## Expected output

```
=== GPU Info ===
  Device:            NVIDIA T4
  Compute capability: 7.5
  Multiprocessors:    40
  Total global mem:   15109 MB

=== Kernel Output ===
  [thread  0 / 16] Hello from the GPU!
  [thread  1 / 16] Hello from the GPU!
  ...

✓ CUDA hello world succeeded on NVIDIA T4
```
