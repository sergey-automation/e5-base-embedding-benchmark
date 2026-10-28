# NVIDIA P104-100 batch-size benchmark

Date: 2026-10-28

## Test setup

- GPU: NVIDIA P104-100 8 GB
- GPU count: 1
- Power limit: 105 W
- Model: `intfloat/multilingual-e5-base`
- Backend: PyTorch
- PyTorch: 2.14.0+cu126
- OS: Ubuntu 24.04.4 LTS
- CPU: Intel Core i5-7500
- RAM: 16 GB
- SSD: Samsung PM981 NVMe
- Embedding dimension: 768
- Dataset: `/mnt/pm981/embedding/lib6/data/chunks_idx.jsonl`
- Dataset size: 352,687 chunks
- Benchmark script: `BUILD_EMBEDDINGS_UNIVERSAL_V1`
- Script version: `2026-04-07 v1.0`

## Test method

The same first 2,048 chunks were used for every run.

Fixed parameters:

- `--block-size 2048`
- `--block-start 0`
- `--block-end 1`
- `CUDA_VISIBLE_DEVICES=0`
- one benchmark run per batch size

Tested batch sizes:

`1, 4, 8, 16, 32, 64, 128, 256`

Before each run, the corresponding output directory and log file were removed.

## Results

| Batch size | Block size | Throughput, chunks/s | Reported ETA for full dataset |
|---:|---:|---:|---:|
| 1 | 2048 | 38.49 | 02:31:50 |
| 4 | 2048 | 34.86 | 02:47:39 |
| 8 | 2048 | 32.27 | 03:01:06 |
| 16 | 2048 | 30.01 | 03:14:43 |
| 32 | 2048 | 27.50 | 03:32:29 |
| 64 | 2048 | 26.49 | 03:40:38 |
| 128 | 2048 | 25.83 | 03:46:13 |
| 256 | 2048 | 24.27 | 04:00:48 |

## Raw benchmark lines

```text
batch=1    speed_last_block=38.49 chunks/s eta=02:31:50
batch=4    speed_last_block=34.86 chunks/s eta=02:47:39
batch=8    speed_last_block=32.27 chunks/s eta=03:01:06
batch=16   speed_last_block=30.01 chunks/s eta=03:14:43
batch=32   speed_last_block=27.50 chunks/s eta=03:32:29
batch=64   speed_last_block=26.49 chunks/s eta=03:40:38
batch=128  speed_last_block=25.83 chunks/s eta=03:46:13
batch=256  speed_last_block=24.27 chunks/s eta=04:00:48
```

## Command

```bash
cd /mnt/pm981/embedding/lib6
source /mnt/pm981/embedding/lib6/venv/bin/activate

for b in 1 4 8 16 32 64 128 256; do
    rm -rf "/mnt/pm981/embedding/lib6/results/p104_b${b}_block2048_e5_base_torch"
    rm -f "/mnt/pm981/embedding/lib6/logs/p104_b${b}_block2048.log"

    CUDA_VISIBLE_DEVICES=0 \
    python /mnt/pm981/embedding/lib6/scripts/block_embeddings_builder.py \
        --chunks /mnt/pm981/embedding/lib6/data/chunks_idx.jsonl \
        --out-root /mnt/pm981/embedding/lib6/results \
        --base-name "p104_b${b}_block2048" \
        --backend torch \
        --batch "$b" \
        --block-size 2048 \
        --block-start 0 \
        --block-end 1 \
        2>&1 | tee "/mnt/pm981/embedding/lib6/logs/p104_b${b}_block2048.log"
done
```

## Extracting results

```bash
for b in 1 4 8 16 32 64 128 256; do
    printf "batch=%-3s  " "$b"
    grep "speed_last_block" \
        "/mnt/pm981/embedding/lib6/logs/p104_b${b}_block2048.log" | tail -1
done
```
