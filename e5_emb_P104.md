# NVIDIA P104-100 — multilingual-e5-base embedding benchmark

Date: 2026-10-28

## Hardware and software

- GPU: 12× NVIDIA P104-100 8 GB
- CPU: Intel Core i5-7500
- RAM: 16 GB
- SSD: Samsung PM981 NVMe
- OS: Ubuntu 24.04.4 LTS
- Backend: PyTorch
- PyTorch: 2.14.0+cu126
- Model: `intfloat/multilingual-e5-base`
- Embedding dimension: 768
- Dataset: `/mnt/pm981/embedding/lib6/data/chunks_idx.jsonl`
- Dataset size: 352,687 chunks
- Benchmark script: `BUILD_EMBEDDINGS_UNIVERSAL_V1`
- Script version: `2026-04-07 v1.0`

## Benchmark parameters

- `--batch 32`
- `--block-size 5000`
- `--block-start 0`
- `--block-end 2`
- Same input range for every run: chunks `[0,10000)`
- Output directory removed before repeated runs
- Multi-GPU tests used one independent Python process per GPU
- Each GPU process used the same input range `[0,10000)`
- Multi-GPU total throughput is the sum of the individual GPU-process throughput values

## Summary

| GPU count | Power limit / GPU | Block 0 avg, chunks/s/GPU | Block 1 avg, chunks/s/GPU | Mean, chunks/s/GPU | Total throughput, chunks/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 90 W | 21.31 | 20.80 | 21.06 | 21.06 |
| 1 | 105 W | 27.32 | 26.87 | 27.09 | 27.09 |
| 1 | 125 W | 28.71 | 28.31 | 28.51 | 28.51 |
| 1 | 180 W | 29.72 | 29.31 | 29.52 | 29.52 |
| 1 | 215 W | 29.90 | 29.45 | 29.68 | 29.68 |
| 6 | 125 W | 26.96 | 26.37 | 26.66 | 159.99 |
| 8 | 105 W | 24.72 | 24.01 | 24.37 | 194.94 |
| 10 | 105 W | 23.33 | 22.63 | 22.98 | 229.79 |
| 12 | 90 W | 17.36 | 17.67 | 17.51 | 210.16 |
| 12 | 105 W | 21.95 | 21.52 | 21.74 | 260.82 |
| 12 | 115 W | 22.46 | 22.37 | 22.41 | 268.97 |
| 12 | 125 W | 22.98 | 22.76 | 22.87 | 274.42 |
| 12 | 180 W | 24.32 | 24.65 | 24.48 | 293.80 |

## 1 GPU — power-limit sweep

| Power limit | Block 0, chunks/s | Block 1, chunks/s | Mean throughput, chunks/s |
|---:|---:|---:|---:|
| 90 W | 21.31 | 20.80 | 21.06 |
| 105 W | 27.32 | 26.87 | 27.09 |
| 125 W | 28.71 | 28.31 | 28.51 |
| 180 W | 29.72 | 29.31 | 29.52 |
| 215 W | 29.90 | 29.45 | 29.68 |

## 105 W — GPU-count sweep

| GPU count | Mean throughput per GPU, chunks/s | Total throughput, chunks/s |
|---:|---:|---:|
| 1 | 27.09 | 27.09 |
| 8 | 24.37 | 194.94 |
| 10 | 22.98 | 229.79 |
| 12 | 21.74 | 260.82 |

## 12 GPU — power-limit sweep

| Power limit / GPU | Total power-limit budget | Mean throughput per GPU, chunks/s | Total throughput, chunks/s |
|---:|---:|---:|---:|
| 90 W | 1080 W | 17.51 | 210.16 |
| 105 W | 1260 W | 21.74 | 260.82 |
| 115 W | 1380 W | 22.41 | 268.97 |
| 125 W | 1500 W | 22.87 | 274.42 |
| 180 W | 2160 W | 24.48 | 293.80 |

`Total power-limit budget` is the configured per-GPU power limit multiplied by the GPU count. It is not measured wall power.

## 6 GPU @ 125 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 26.94 | 26.29 |
| 1 | 26.94 | 26.23 |
| 2 | 26.96 | 26.48 |
| 3 | 27.02 | 26.57 |
| 4 | 26.98 | 26.48 |
| 5 | 26.92 | 26.16 |

- Block 0 total: 161.76 chunks/s
- Block 1 total: 158.21 chunks/s
- Mean total throughput: 159.99 chunks/s
- Mean per GPU: 26.66 chunks/s

## 8 GPU @ 105 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 24.68 | 23.97 |
| 1 | 24.67 | 23.76 |
| 2 | 25.29 | 24.60 |
| 3 | 25.18 | 24.66 |
| 4 | 24.96 | 24.30 |
| 5 | 23.82 | 23.16 |
| 6 | 24.46 | 23.62 |
| 7 | 24.73 | 24.01 |

- Block 0 total: 197.79 chunks/s
- Block 1 total: 192.08 chunks/s
- Mean total throughput: 194.94 chunks/s
- Mean per GPU: 24.37 chunks/s

## 10 GPU @ 105 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 23.25 | 22.52 |
| 1 | 23.14 | 22.45 |
| 2 | 24.05 | 23.10 |
| 3 | 23.86 | 23.10 |
| 4 | 23.55 | 22.90 |
| 5 | 22.66 | 22.00 |
| 6 | 23.09 | 22.53 |
| 7 | 23.45 | 22.45 |
| 8 | 22.86 | 22.57 |
| 9 | 23.36 | 22.69 |

- Block 0 total: 233.27 chunks/s
- Block 1 total: 226.31 chunks/s
- Mean total throughput: 229.79 chunks/s
- Mean per GPU: 22.98 chunks/s

## 12 GPU @ 90 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 17.78 | 17.39 |
| 1 | 17.36 | 17.27 |
| 2 | 18.03 | 18.59 |
| 3 | 18.47 | 18.33 |
| 4 | 17.79 | 17.97 |
| 5 | 16.11 | 16.97 |
| 6 | 16.54 | 17.55 |
| 7 | 16.57 | 17.38 |
| 8 | 16.96 | 17.35 |
| 9 | 17.84 | 17.73 |
| 10 | 16.25 | 17.19 |
| 11 | 18.56 | 18.34 |

- Block 0 total: 208.26 chunks/s
- Block 1 total: 212.06 chunks/s
- Mean total throughput: 210.16 chunks/s
- Mean per GPU: 17.51 chunks/s

## 12 GPU @ 105 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 22.13 | 21.28 |
| 1 | 22.19 | 21.38 |
| 2 | 22.48 | 21.93 |
| 3 | 22.55 | 21.90 |
| 4 | 21.71 | 21.90 |
| 5 | 21.49 | 20.92 |
| 6 | 21.66 | 21.58 |
| 7 | 21.72 | 21.61 |
| 8 | 21.66 | 21.39 |
| 9 | 22.29 | 21.52 |
| 10 | 21.01 | 21.09 |
| 11 | 22.50 | 21.75 |

- Block 0 total: 263.39 chunks/s
- Block 1 total: 258.25 chunks/s
- Mean total throughput: 260.82 chunks/s
- Mean per GPU: 21.74 chunks/s

## 12 GPU @ 115 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 22.94 | 22.32 |
| 1 | 23.09 | 22.39 |
| 2 | 21.79 | 22.51 |
| 3 | 23.39 | 22.37 |
| 4 | 22.28 | 22.37 |
| 5 | 22.07 | 22.16 |
| 6 | 22.02 | 22.34 |
| 7 | 22.26 | 22.34 |
| 8 | 22.00 | 22.39 |
| 9 | 22.18 | 22.54 |
| 10 | 23.11 | 22.12 |
| 11 | 22.43 | 22.53 |

- Block 0 total: 269.56 chunks/s
- Block 1 total: 268.38 chunks/s
- Mean total throughput: 268.97 chunks/s
- Mean per GPU: 22.41 chunks/s

## 12 GPU @ 125 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 22.79 | 22.73 |
| 1 | 23.31 | 22.85 |
| 2 | 23.28 | 22.85 |
| 3 | 23.31 | 22.99 |
| 4 | 22.83 | 22.71 |
| 5 | 23.59 | 22.91 |
| 6 | 22.71 | 22.72 |
| 7 | 22.90 | 22.73 |
| 8 | 22.61 | 22.74 |
| 9 | 23.26 | 22.27 |
| 10 | 22.41 | 22.70 |
| 11 | 22.76 | 22.88 |

- Block 0 total: 275.76 chunks/s
- Block 1 total: 273.08 chunks/s
- Mean total throughput: 274.42 chunks/s
- Mean per GPU: 22.87 chunks/s

## 12 GPU @ 180 W

| GPU | Block 0, chunks/s | Block 1, chunks/s |
|---:|---:|---:|
| 0 | 24.47 | 24.61 |
| 1 | 23.51 | 24.85 |
| 2 | 24.50 | 24.80 |
| 3 | 24.61 | 24.96 |
| 4 | 24.75 | 24.83 |
| 5 | 24.67 | 24.32 |
| 6 | 23.48 | 23.77 |
| 7 | 24.37 | 24.66 |
| 8 | 24.43 | 24.63 |
| 9 | 24.30 | 24.72 |
| 10 | 23.98 | 24.79 |
| 11 | 24.77 | 24.82 |

- Block 0 total: 291.84 chunks/s
- Block 1 total: 295.76 chunks/s
- Mean total throughput: 293.80 chunks/s
- Mean per GPU: 24.48 chunks/s

## Recorded system measurements

During a 12-GPU run:

- CPU utilization: approximately 100%
- `iowait`: approximately 0%
- RAM: almost fully allocated
- Swap: in use
- Samsung PM981 `nvme0n1` utilization in the recorded `iostat` sample: approximately 1.28%

Recorded GPU clocks:

| Power limit | Core clock | Memory clock |
|---:|---:|---:|
| 125 W | approximately 1.5–1.6 GHz | 5005 MHz |
| 180–215 W | approximately 1.9 GHz | 5005 MHz |
