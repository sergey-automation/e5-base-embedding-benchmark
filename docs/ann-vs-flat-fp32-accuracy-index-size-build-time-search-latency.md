# ANN benchmark vs exact FP32 — comprehensive build, size, search and quality report

**Updated:** 2026-11-05  
**Ground truth:** exact FP32 exhaustive Inner Product scan (IndexFlatIP-equivalent) = **100%**.  
**Status:** reproducible measured data, missing data marked explicitly.  
**Companion historical report:** [`ann-benchmark-vs-hnsw-fp32.md`](ann-benchmark-vs-hnsw-fp32.md).

## 1. Benchmark basis and metric

- Model `intfloat/multilingual-e5-base`; **4,304,468** normalized embedding vectors, dimension **768**, `float32` source data, Inner Product.
- **200 fixed queries**, `K=500`; 66 original FP32 parts scanned once in **91.27 s** to obtain the exact top-500 ID reference. The 91.27 s is a **whole-batch reference scan**, not comparable to CPU ANN per-query median.
- Metric: overlap of returned vector IDs with exact FP32 Top1/5/10/50/500 sets, with Top1 requiring same rank-1 ID. **Not relevance accuracy**. No transformation from HNSW-relative percentages.
- Search settings: IVF `nprobe=32,64,128,256,512,1024`; HNSW `efSearch` as available, including 512–32768. Exact quality comes from `flatip_exact_groundtruth/all_index_results_vs_exact_flatip.csv`.
- Size = serialized index on disk, **not measured resident RAM**. CPU latency values are previous actual search-run measurements, typically **median per query**, not recalculated during exact recall comparison. GPU latency was **not validated**; no GPU time is guessed.
- `—` = data absent or unverified for that exact configuration. A historical approximate figure is explicitly labeled `~`.

## 2. Direct answer: is each search setting timed and each build measured?

**No.** Complete exact-FP32 ranking quality exists for 198 archived result rows (including repeats). Latency is available for each of the CPU IVF/PQ/OPQ/RaBitQ sweeps whose original per-run `summary(*).csv` was preserved, except for older or incomplete RQ runs. HNSW latency is available for all M8/M16 main sweep settings and for a limited set of M32/SQ4 settings; detailed latency files for other HNSW sweeps are not in the supplied evidence. All GPU variants have ranking results but **no validated comparable search latency**. Build statistics are available per variant at differing degrees of completeness; never fill an absent phase with a hypothetical time.

## 3. All index variants: best observed exact FP32 Top10, size, search speed at that setting

| Platform | Index | Size GiB | Best setting | Top1 % | Top10 % | Top500 % | Median ms/query |
|---|---|---|---|---|---|---|---|
| CPU | FP32_HNSW_M32 | 13.420 | `ef=32768` | 98.50 | 99.60 | 99.96 | — |
| CPU | RR768_IVF4096_RaBitQ8 | 3.205 | `nprobe=1024` | 98.50 | 98.85 | 99.32 | 193.44 |
| CPU | sq8 | 4.170 | `ef=32768` | 96.50 | 97.60 | 98.79 | — |
| CPU | sq8_m16 | 3.657 | `ef=32768` | 96.50 | 97.60 | 98.77 | 2817.71 |
| CPU | sq8_m8 | 3.402 | `ef=32768` | 96.50 | 97.35 | 98.38 | 2610.73 |
| CPU | RR768_IVF4096_RaBitQ4 | 1.666 | `nprobe=1024` | 90.50 | 92.80 | 95.41 | 129.60 |
| CPU | IVF_PQ384 | 1.584 | `nprobe=1024` | 82.00 | 90.20 | 92.36 | 624.83 |
| GPU | PQ384_f050_P104 | 1.606 | `nprobe=1024` | 84.00 | 89.60 | 92.51 | — |
| CPU | OPQ_IVF_PQ384 | 1.586 | `nprobe=1024` | 87.00 | 89.45 | 92.44 | 774.91 |
| CPU | RR768_IVF4096_RaBitQ3 | 1.281 | `nprobe=1024` | 88.50 | 88.60 | 92.02 | 121.52 |
| GPU | PQ384_f025_P104 | 1.606 | `nprobe=1024` | 85.00 | 88.30 | 92.30 | — |
| CPU | RR768_IVF4096_RaBitQ2 | 0.896 | `nprobe=512` | 81.50 | 82.30 | 84.70 | 61.81 |
| CPU | IVF_SQ4 | 1.583 | `nprobe=512` | 79.50 | 81.05 | 84.88 | 1048.82 |
| CPU | OPQ_IVF_PQ256 | 1.073 | `nprobe=1024` | 74.50 | 80.30 | 83.35 | 300.31 |
| CPU | IVF_PQ256 | 1.071 | `nprobe=1024` | 78.50 | 78.85 | 83.34 | 301.16 |
| CPU | RR768_IVF4096_RQ256x8 | 1.258 | `nprobe=1024` | 75.00 | 78.55 | 83.03 | 22436.29 |
| CPU | sq4 | 2.630 | `ef=32768` | 80.00 | 78.35 | 81.65 | — |
| CPU | OPQ_IVF_PQ192 | 0.816 | `nprobe=1024` | 67.50 | 74.45 | 77.08 | 209.97 |
| GPU | PQ192_f025_P104 | 0.825 | `nprobe=1024` | 65.00 | 69.30 | 74.83 | — |
| GPU | PQ192_f050_P104 | 0.825 | `nprobe=1024` | 68.00 | 69.05 | 74.69 | — |
| CPU | IVF_PQ192 | 0.814 | `nprobe=1024` | 63.00 | 68.95 | 74.68 | 235.70 |
| CPU | RR768_IVF4096_RaBitQ1 | 0.463 | `nprobe=1024` | 62.50 | 65.95 | 69.66 | 91.84 |
| CPU | OPQ_IVF_PQ128 | 0.560 | `nprobe=1024` | 60.00 | 60.80 | 62.44 | 133.83 |
| GPU | PQ96_f050_P104 | 0.435 | `nprobe=1024` | 40.50 | 44.50 | 53.15 | — |
| GPU | PQ96_f025_P104 | 0.435 | `nprobe=1024` | 41.50 | 41.85 | 52.95 | — |
| GPU | PQ96_f050_CMP90 | 0.435 | `nprobe=1024` | 36.00 | 41.60 | 52.45 | — |


**Selection:** highest observed Top10 even if the search is slow. See section 4 for settings that actually meet the measured `<200 ms` constraint. Selecting best recall does not imply best speed. The `unknown` settings represent legacy cached output whose search parameter was not recorded.

## 4. CPU speed-constrained quality: best measured setting strictly below 200 ms/query

| CPU index | Size GiB | Setting | Exact Top10 % | Exact Top500 % | Median ms |
|---|---|---|---|---|---|
| RR768_IVF4096_RaBitQ8 | 3.205 | `nprobe=1024` | 98.85 | 99.32 | 193.44 |
| sq8_m16 | 3.657 | `ef=4096` | 97.35 | 98.14 | 104.45 |
| sq8_m8 | 3.402 | `ef=4096` | 96.25 | 96.44 | 78.61 |
| RR768_IVF4096_RaBitQ4 | 1.666 | `nprobe=1024` | 92.80 | 95.41 | 129.21 |
| IVF_PQ384 | 1.584 | `nprobe=256` | 89.05 | 91.04 | 164.16 |
| OPQ_IVF_PQ384 | 1.586 | `nprobe=256` | 88.65 | 91.07 | 159.22 |
| RR768_IVF4096_RaBitQ3 | 1.281 | `nprobe=1024` | 88.60 | 92.02 | 121.52 |
| RR768_IVF4096_RaBitQ2 | 0.896 | `nprobe=512` | 82.30 | 84.70 | 61.81 |
| OPQ_IVF_PQ256 | 1.073 | `nprobe=512` | 80.20 | 83.09 | 147.19 |
| IVF_PQ256 | 1.071 | `nprobe=512` | 78.75 | 83.09 | 149.84 |
| IVF_SQ4 | 1.583 | `nprobe=64` | 77.55 | 79.80 | 140.62 |
| OPQ_IVF_PQ192 | 0.816 | `nprobe=512` | 74.35 | 76.91 | 108.87 |
| IVF_PQ192 | 0.814 | `nprobe=512` | 68.85 | 74.47 | 113.87 |
| RR768_IVF4096_RaBitQ1 | 0.463 | `nprobe=1024` | 65.95 | 69.66 | 91.84 |
| OPQ_IVF_PQ128 | 0.560 | `nprobe=1024` | 60.80 | 62.44 | 133.83 |


**Not eligible or not established:** RQ256x8 (all measured timings >200 ms), FP32 HNSW M32 (no verified full search median for the available sweep), SQ8 HNSW M32 and SQ4 HNSW M32 (only historical approximate timings in the supplied report), GPU variants (no validated GPU search latency). These are excluded **only from this constrained table**, not from the full results.

## 5. All 198 exact-reference measurements, with search performance when recorded

Each line preserves the source identity, so repeated runs stay separate. CPU `median`, `mean`, `p95`, `p99`, `QPS` and batch search seconds are copied from the original same-run CSV when present; **quality metrics are direct exact-reference overlap**. For M8/M16 HNSW latency, the source is the historical dedicated comparison; the more detailed distributions were not supplied.

| Platform | Index | Search setting | GiB | Top1 % | Top5 % | Top10 % | Top50 % | Top500 % | Median ms | Mean ms | p95 ms | p99 ms | Batch s | QPS | ID source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CPU | FP32_HNSW_M32 | `unknown` | 13.420 | 94.50 | 93.70 | 94.00 | 94.25 | 88.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_103718\fp32_results.npz` |
| CPU | sq8 | `unknown` | 4.170 | 51.50 | 54.90 | 54.70 | 53.73 | 38.64 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_103718\sq8_results.npz` |
| CPU | FP32_HNSW_M32 | `unknown` | 13.420 | 94.50 | 93.70 | 94.00 | 94.25 | 88.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_125339\fp32_results.npz` |
| CPU | sq8 | `ef=1000` | 4.170 | 94.50 | 96.70 | 95.70 | 97.11 | 95.54 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_125339\sq8_ef1000_results.npz` |
| CPU | sq8 | `ef=128` | 4.170 | 64.50 | 68.10 | 68.15 | 68.65 | 56.44 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_125339\sq8_ef128_results.npz` |
| CPU | sq8 | `ef=256` | 4.170 | 80.00 | 82.10 | 82.25 | 83.84 | 76.08 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_125339\sq8_ef256_results.npz` |
| CPU | sq8 | `ef=500` | 4.170 | 89.00 | 91.40 | 90.95 | 92.42 | 88.55 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_125339\sq8_ef500_results.npz` |
| CPU | sq8 | `ef=64` | 4.170 | 51.50 | 54.90 | 54.70 | 53.73 | 38.64 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_125339\sq8_ef64_results.npz` |
| CPU | FP32_HNSW_M32 | `unknown` | 13.420 | 94.50 | 93.70 | 94.00 | 94.25 | 88.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\fp32_results.npz` |
| CPU | sq8 | `ef=1024` | 4.170 | 94.50 | 96.70 | 95.75 | 97.12 | 95.61 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\sq8_ef1024_results.npz` |
| CPU | sq8 | `ef=16384` | 4.170 | 96.50 | 98.20 | 97.55 | 98.68 | 98.77 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\sq8_ef16384_results.npz` |
| CPU | sq8 | `ef=2048` | 4.170 | 95.50 | 97.60 | 96.95 | 98.02 | 97.69 | ~61.00 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\sq8_ef2048_results.npz` |
| CPU | sq8 | `ef=32768` | 4.170 | 96.50 | 98.30 | 97.60 | 98.69 | 98.79 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\sq8_ef32768_results.npz` |
| CPU | sq8 | `ef=4096` | 4.170 | 96.50 | 98.20 | 97.50 | 98.59 | 98.47 | ~131.00 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\sq8_ef4096_results.npz` |
| CPU | sq8 | `ef=512` | 4.170 | 89.00 | 91.40 | 90.95 | 92.43 | 88.68 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\sq8_ef512_results.npz` |
| CPU | sq8 | `ef=8192` | 4.170 | 96.50 | 98.20 | 97.55 | 98.67 | 98.68 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_133703\sq8_ef8192_results.npz` |
| CPU | sq4 | `ef=1024` | 2.630 | 78.50 | 76.10 | 76.55 | 77.80 | 80.09 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8_sq4\20261031_094848\sq4_ef1024_results.npz` |
| CPU | sq4 | `ef=16384` | 2.630 | 80.00 | 77.80 | 78.35 | 79.32 | 81.63 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8_sq4\20261031_094848\sq4_ef16384_results.npz` |
| CPU | sq4 | `ef=2048` | 2.630 | 80.00 | 78.00 | 78.10 | 79.21 | 81.32 | ~70.00 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8_sq4\20261031_094848\sq4_ef2048_results.npz` |
| CPU | sq4 | `ef=32768` | 2.630 | 80.00 | 77.80 | 78.35 | 79.33 | 81.65 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8_sq4\20261031_094848\sq4_ef32768_results.npz` |
| CPU | sq4 | `ef=4096` | 2.630 | 80.00 | 77.80 | 78.35 | 79.24 | 81.49 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8_sq4\20261031_094848\sq4_ef4096_results.npz` |
| CPU | sq4 | `ef=512` | 2.630 | 71.50 | 71.70 | 72.40 | 74.00 | 75.47 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8_sq4\20261031_094848\sq4_ef512_results.npz` |
| CPU | sq4 | `ef=8192` | 2.630 | 80.00 | 77.80 | 78.35 | 79.31 | 81.59 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_fp32_sq8_sq4\20261031_094848\sq4_ef8192_results.npz` |
| CPU | IVF_PQ192 | `nprobe=1024` | 0.814 | 63.00 | 67.80 | 68.95 | 72.04 | 74.68 | 235.70 | 240.33 | 310.07 | 424.31 | 48.07 | 4.16 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ192_nprobe1024.npz` |
| CPU | IVF_PQ192 | `nprobe=128` | 0.814 | 61.50 | 66.00 | 67.20 | 70.82 | 72.74 | 29.92 | 30.79 | 37.23 | 47.34 | 6.16 | 32.46 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ192_nprobe128.npz` |
| CPU | IVF_PQ192 | `nprobe=256` | 0.814 | 62.00 | 67.00 | 68.20 | 71.58 | 73.93 | 55.95 | 59.47 | 86.03 | 116.61 | 11.90 | 16.81 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ192_nprobe256.npz` |
| CPU | IVF_PQ192 | `nprobe=32` | 0.814 | 59.00 | 63.00 | 64.70 | 67.23 | 67.87 | 9.49 | 9.68 | 11.54 | 14.83 | 1.94 | 103.09 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ192_nprobe32.npz` |
| CPU | IVF_PQ192 | `nprobe=512` | 0.814 | 62.50 | 67.70 | 68.85 | 71.98 | 74.47 | 113.87 | 142.54 | 312.60 | 405.91 | 28.51 | 7.01 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ192_nprobe512.npz` |
| CPU | IVF_PQ192 | `nprobe=64` | 0.814 | 60.50 | 65.00 | 66.05 | 69.56 | 70.82 | 15.82 | 15.95 | 18.42 | 19.46 | 3.19 | 62.65 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ192_nprobe64.npz` |
| CPU | IVF_PQ384 | `nprobe=1024` | 1.584 | 82.00 | 89.60 | 90.20 | 91.44 | 92.36 | 624.83 | 638.59 | 760.52 | 1040.50 | 127.72 | 1.57 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ384_nprobe1024.npz` |
| CPU | IVF_PQ384 | `nprobe=128` | 1.584 | 78.50 | 86.40 | 87.75 | 89.15 | 88.79 | 80.67 | 82.67 | 98.11 | 143.25 | 16.54 | 12.09 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ384_nprobe128.npz` |
| CPU | IVF_PQ384 | `nprobe=256` | 1.584 | 80.50 | 88.10 | 89.05 | 90.62 | 91.04 | 164.16 | 177.12 | 268.94 | 351.78 | 35.43 | 5.65 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ384_nprobe256.npz` |
| CPU | IVF_PQ384 | `nprobe=32` | 1.584 | 74.50 | 81.20 | 82.35 | 83.11 | 80.13 | 25.09 | 25.03 | 29.89 | 33.43 | 5.01 | 39.92 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ384_nprobe32.npz` |
| CPU | IVF_PQ384 | `nprobe=512` | 1.584 | 81.50 | 89.40 | 90.05 | 91.28 | 92.02 | 318.99 | 329.87 | 406.73 | 573.18 | 65.98 | 3.03 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ384_nprobe512.npz` |
| CPU | IVF_PQ384 | `nprobe=64` | 1.584 | 76.50 | 84.10 | 85.35 | 86.78 | 85.25 | 41.83 | 41.93 | 48.71 | 52.17 | 8.39 | 23.84 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\IVF_PQ384_nprobe64.npz` |
| CPU | OPQ_IVF_PQ192 | `nprobe=1024` | 0.816 | 67.50 | 73.20 | 74.45 | 75.58 | 77.08 | 209.97 | 212.19 | 237.34 | 259.39 | 42.44 | 4.71 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ192_nprobe1024.npz` |
| CPU | OPQ_IVF_PQ192 | `nprobe=128` | 0.816 | 65.50 | 71.00 | 72.85 | 74.10 | 75.23 | 33.45 | 34.48 | 41.94 | 64.61 | 6.90 | 28.99 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ192_nprobe128.npz` |
| CPU | OPQ_IVF_PQ192 | `nprobe=256` | 0.816 | 66.50 | 72.40 | 73.90 | 75.02 | 76.36 | 57.77 | 58.17 | 66.99 | 72.12 | 11.64 | 17.19 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ192_nprobe256.npz` |
| CPU | OPQ_IVF_PQ192 | `nprobe=32` | 0.816 | 61.50 | 66.50 | 69.65 | 69.98 | 70.05 | 11.07 | 11.23 | 14.27 | 16.58 | 2.25 | 88.85 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ192_nprobe32.npz` |
| CPU | OPQ_IVF_PQ192 | `nprobe=512` | 0.816 | 67.00 | 73.00 | 74.35 | 75.47 | 76.91 | 108.87 | 109.45 | 124.36 | 133.12 | 21.89 | 9.14 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ192_nprobe512.npz` |
| CPU | OPQ_IVF_PQ192 | `nprobe=64` | 0.816 | 63.00 | 68.90 | 71.15 | 72.54 | 73.20 | 18.34 | 18.92 | 22.41 | 40.21 | 3.79 | 52.82 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ192_nprobe64.npz` |
| CPU | OPQ_IVF_PQ384 | `nprobe=1024` | 1.586 | 87.00 | 89.50 | 89.45 | 91.43 | 92.44 | 774.91 | 975.16 | 2114.64 | 2821.35 | 195.04 | 1.03 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ384_nprobe1024.npz` |
| CPU | OPQ_IVF_PQ384 | `nprobe=128` | 1.586 | 84.00 | 86.50 | 86.95 | 89.32 | 88.86 | 82.37 | 94.84 | 147.50 | 370.57 | 18.97 | 10.54 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ384_nprobe128.npz` |
| CPU | OPQ_IVF_PQ384 | `nprobe=256` | 1.586 | 86.00 | 88.40 | 88.65 | 90.72 | 91.07 | 159.22 | 160.44 | 182.07 | 189.97 | 32.09 | 6.23 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ384_nprobe256.npz` |
| CPU | OPQ_IVF_PQ384 | `nprobe=32` | 1.586 | 79.00 | 81.50 | 81.75 | 83.07 | 80.16 | 29.08 | 30.46 | 42.57 | 51.74 | 6.10 | 32.77 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ384_nprobe32.npz` |
| CPU | OPQ_IVF_PQ384 | `nprobe=512` | 1.586 | 86.50 | 89.40 | 89.30 | 91.24 | 92.08 | 329.10 | 361.14 | 495.68 | 761.44 | 72.23 | 2.77 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ384_nprobe512.npz` |
| CPU | OPQ_IVF_PQ384 | `nprobe=64` | 1.586 | 81.00 | 84.10 | 84.50 | 86.83 | 85.38 | 60.68 | 77.10 | 177.76 | 303.95 | 15.42 | 12.97 | `C:\VAST2_UPLOAD\out\compare_ivfpq_vs_fp32_m32\20261102_153517\OPQ_IVF_PQ384_nprobe64.npz` |
| CPU | IVF_SQ4 | `nprobe=1024` | 1.583 | 80.00 | 80.70 | 80.95 | 83.50 | 85.16 | 2130.64 | 2150.78 | 2459.04 | 3418.41 | 430.16 | 0.46 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\IVF_SQ4_nprobe1024.npz` |
| CPU | IVF_SQ4 | `nprobe=128` | 1.583 | 77.50 | 78.00 | 79.20 | 81.69 | 82.44 | 267.18 | 269.18 | 306.78 | 338.07 | 53.84 | 3.71 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\IVF_SQ4_nprobe128.npz` |
| CPU | IVF_SQ4 | `nprobe=256` | 1.583 | 78.50 | 79.60 | 80.30 | 82.78 | 84.11 | 524.49 | 525.75 | 598.55 | 668.17 | 105.15 | 1.90 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\IVF_SQ4_nprobe256.npz` |
| CPU | IVF_SQ4 | `nprobe=32` | 1.583 | 71.50 | 73.40 | 75.10 | 76.69 | 75.66 | 76.00 | 75.53 | 86.68 | 90.28 | 15.11 | 13.24 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\IVF_SQ4_nprobe32.npz` |
| CPU | IVF_SQ4 | `nprobe=512` | 1.583 | 79.50 | 80.50 | 81.05 | 83.34 | 84.88 | 1048.82 | 1077.02 | 1321.45 | 1620.96 | 215.41 | 0.93 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\IVF_SQ4_nprobe512.npz` |
| CPU | IVF_SQ4 | `nprobe=64` | 1.583 | 75.00 | 76.00 | 77.55 | 79.68 | 79.80 | 140.62 | 141.06 | 160.80 | 168.84 | 28.22 | 7.09 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\IVF_SQ4_nprobe64.npz` |
| CPU | OPQ_IVF_PQ128 | `nprobe=1024` | 0.560 | 60.00 | 59.00 | 60.80 | 61.11 | 62.44 | 133.83 | 134.28 | 145.45 | 149.36 | 26.86 | 7.45 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ128_nprobe1024.npz` |
| CPU | OPQ_IVF_PQ128 | `nprobe=128` | 0.560 | 58.00 | 57.30 | 59.40 | 60.17 | 61.47 | 20.09 | 20.41 | 24.19 | 27.89 | 4.09 | 48.95 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ128_nprobe128.npz` |
| CPU | OPQ_IVF_PQ128 | `nprobe=256` | 0.560 | 59.50 | 58.20 | 60.35 | 60.78 | 62.05 | 37.63 | 38.06 | 43.67 | 46.25 | 7.62 | 26.26 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ128_nprobe256.npz` |
| CPU | OPQ_IVF_PQ128 | `nprobe=32` | 0.560 | 53.50 | 54.30 | 57.00 | 57.52 | 58.56 | 6.64 | 6.59 | 7.35 | 8.22 | 1.32 | 151.25 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ128_nprobe32.npz` |
| CPU | OPQ_IVF_PQ128 | `nprobe=512` | 0.560 | 59.50 | 58.90 | 60.70 | 61.08 | 62.35 | 69.84 | 70.53 | 78.96 | 82.37 | 14.11 | 14.17 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ128_nprobe512.npz` |
| CPU | OPQ_IVF_PQ128 | `nprobe=64` | 0.560 | 56.00 | 56.10 | 58.25 | 58.98 | 60.39 | 10.93 | 10.98 | 12.35 | 12.97 | 2.20 | 90.92 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ128_nprobe64.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=1024` | 1.073 | 74.50 | 78.20 | 80.30 | 81.26 | 83.35 | 300.31 | 301.11 | 324.92 | 335.25 | 60.22 | 3.32 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ256_nprobe1024.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=128` | 1.073 | 72.50 | 76.10 | 78.20 | 79.45 | 80.76 | 41.44 | 41.72 | 47.95 | 50.16 | 8.35 | 23.96 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ256_nprobe128.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=256` | 1.073 | 74.00 | 77.30 | 79.55 | 80.66 | 82.32 | 78.54 | 79.21 | 88.75 | 90.74 | 15.84 | 12.62 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ256_nprobe256.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=32` | 1.073 | 67.00 | 72.20 | 73.80 | 74.71 | 74.19 | 12.91 | 12.81 | 14.52 | 15.39 | 2.57 | 77.96 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ256_nprobe32.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=512` | 1.073 | 74.00 | 78.10 | 80.20 | 81.19 | 83.09 | 147.19 | 148.83 | 165.63 | 169.75 | 29.77 | 6.72 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ256_nprobe512.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=64` | 1.073 | 70.00 | 74.20 | 76.25 | 77.56 | 78.17 | 22.72 | 22.79 | 25.65 | 26.72 | 4.56 | 43.85 | `C:\VAST2_UPLOAD\out\compare_new_cpu_ivf_vs_fp32_m32\20261103_220225\OPQ_IVF_PQ256_nprobe64.npz` |
| CPU | IVF_PQ256 | `nprobe=1024` | 1.071 | 78.50 | 76.60 | 78.85 | 80.94 | 83.34 | 301.16 | 303.52 | 338.05 | 379.69 | 60.71 | 3.29 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\IVF_PQ256_nprobe1024.npz` |
| CPU | IVF_PQ256 | `nprobe=128` | 1.071 | 76.00 | 74.00 | 76.85 | 79.18 | 80.78 | 40.53 | 40.79 | 45.69 | 49.07 | 8.16 | 24.51 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\IVF_PQ256_nprobe128.npz` |
| CPU | IVF_PQ256 | `nprobe=256` | 1.071 | 77.50 | 75.50 | 78.05 | 80.36 | 82.39 | 76.54 | 77.03 | 87.10 | 88.57 | 15.41 | 12.98 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\IVF_PQ256_nprobe256.npz` |
| CPU | IVF_PQ256 | `nprobe=32` | 1.071 | 71.00 | 69.80 | 73.25 | 74.68 | 74.24 | 12.78 | 12.70 | 14.45 | 14.87 | 2.54 | 78.63 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\IVF_PQ256_nprobe32.npz` |
| CPU | IVF_PQ256 | `nprobe=512` | 1.071 | 78.00 | 76.40 | 78.75 | 80.82 | 83.09 | 149.84 | 150.17 | 166.01 | 169.60 | 30.04 | 6.66 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\IVF_PQ256_nprobe512.npz` |
| CPU | IVF_PQ256 | `nprobe=64` | 1.071 | 73.50 | 72.20 | 75.00 | 77.46 | 78.21 | 22.46 | 22.76 | 25.84 | 36.79 | 4.56 | 43.90 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\IVF_PQ256_nprobe64.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=1024` | 1.073 | 74.50 | 78.20 | 80.30 | 81.26 | 83.35 | 298.99 | 301.28 | 326.74 | 338.85 | 60.26 | 3.32 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\OPQ_IVF_PQ256_nprobe1024.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=128` | 1.073 | 72.50 | 76.10 | 78.20 | 79.45 | 80.76 | 43.58 | 44.10 | 52.82 | 59.84 | 8.82 | 22.67 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\OPQ_IVF_PQ256_nprobe128.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=256` | 1.073 | 74.00 | 77.30 | 79.55 | 80.66 | 82.32 | 78.27 | 80.68 | 99.53 | 119.83 | 16.14 | 12.39 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\OPQ_IVF_PQ256_nprobe256.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=32` | 1.073 | 67.00 | 72.20 | 73.80 | 74.71 | 74.19 | 12.72 | 12.69 | 14.54 | 14.79 | 2.54 | 78.66 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\OPQ_IVF_PQ256_nprobe32.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=512` | 1.073 | 74.00 | 78.10 | 80.20 | 81.19 | 83.09 | 155.63 | 156.90 | 176.42 | 186.67 | 31.38 | 6.37 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\OPQ_IVF_PQ256_nprobe512.npz` |
| CPU | OPQ_IVF_PQ256 | `nprobe=64` | 1.073 | 70.00 | 74.20 | 76.25 | 77.56 | 78.17 | 22.59 | 22.62 | 25.71 | 26.66 | 4.53 | 44.19 | `C:\VAST2_UPLOAD\out\compare_pq256_opq256_vs_fp32_m32\20261104_002611\OPQ_IVF_PQ256_nprobe64.npz` |
| CPU | RR768_IVF4096_RaBitQ1 | `nprobe=1024` | 0.463 | 62.50 | 64.10 | 65.95 | 67.53 | 69.66 | 91.84 | 91.82 | 98.41 | 100.31 | 18.37 | 10.89 | `C:\VAST2_UPLOAD\out\compare_rabitq1_vs_fp32_m32\20261104_110337\RR768_IVF4096_RaBitQ1_nprobe1024.npz` |
| CPU | RR768_IVF4096_RaBitQ1 | `nprobe=128` | 0.463 | 60.00 | 62.40 | 64.70 | 66.31 | 68.24 | 13.46 | 13.54 | 15.11 | 15.76 | 2.71 | 73.78 | `C:\VAST2_UPLOAD\out\compare_rabitq1_vs_fp32_m32\20261104_110337\RR768_IVF4096_RaBitQ1_nprobe128.npz` |
| CPU | RR768_IVF4096_RaBitQ1 | `nprobe=256` | 0.463 | 61.50 | 63.70 | 65.45 | 67.15 | 69.12 | 24.62 | 24.79 | 27.39 | 28.60 | 4.96 | 40.32 | `C:\VAST2_UPLOAD\out\compare_rabitq1_vs_fp32_m32\20261104_110337\RR768_IVF4096_RaBitQ1_nprobe256.npz` |
| CPU | RR768_IVF4096_RaBitQ1 | `nprobe=32` | 0.463 | 58.00 | 58.90 | 61.15 | 62.77 | 64.22 | 4.86 | 4.85 | 5.44 | 5.62 | 0.97 | 205.14 | `C:\VAST2_UPLOAD\out\compare_rabitq1_vs_fp32_m32\20261104_110337\RR768_IVF4096_RaBitQ1_nprobe32.npz` |
| CPU | RR768_IVF4096_RaBitQ1 | `nprobe=512` | 0.463 | 62.00 | 64.00 | 65.85 | 67.47 | 69.54 | 47.09 | 47.14 | 51.35 | 52.92 | 9.43 | 21.21 | `C:\VAST2_UPLOAD\out\compare_rabitq1_vs_fp32_m32\20261104_110337\RR768_IVF4096_RaBitQ1_nprobe512.npz` |
| CPU | RR768_IVF4096_RaBitQ1 | `nprobe=64` | 0.463 | 59.00 | 61.10 | 63.35 | 64.99 | 66.61 | 7.77 | 7.78 | 8.62 | 9.31 | 1.56 | 128.25 | `C:\VAST2_UPLOAD\out\compare_rabitq1_vs_fp32_m32\20261104_110337\RR768_IVF4096_RaBitQ1_nprobe64.npz` |
| CPU | RR768_IVF4096_RaBitQ2 | `nprobe=1024` | 0.896 | 82.00 | 81.10 | 82.25 | 84.15 | 84.96 | 112.91 | 114.20 | 130.22 | 143.68 | 22.84 | 8.76 | `C:\VAST2_UPLOAD\out\compare_rabitq2_vs_fp32_m32\20261104_104639\RR768_IVF4096_RaBitQ2_nprobe1024.npz` |
| CPU | RR768_IVF4096_RaBitQ2 | `nprobe=128` | 0.896 | 78.50 | 78.60 | 80.20 | 82.38 | 82.44 | 23.48 | 24.30 | 31.25 | 35.87 | 4.86 | 41.12 | `C:\VAST2_UPLOAD\out\compare_rabitq2_vs_fp32_m32\20261104_104639\RR768_IVF4096_RaBitQ2_nprobe128.npz` |
| CPU | RR768_IVF4096_RaBitQ2 | `nprobe=256` | 0.896 | 80.50 | 80.20 | 81.65 | 83.55 | 83.94 | 36.59 | 37.96 | 47.24 | 52.76 | 7.60 | 26.33 | `C:\VAST2_UPLOAD\out\compare_rabitq2_vs_fp32_m32\20261104_104639\RR768_IVF4096_RaBitQ2_nprobe256.npz` |
| CPU | RR768_IVF4096_RaBitQ2 | `nprobe=32` | 0.896 | 75.00 | 74.20 | 75.35 | 77.17 | 75.61 | 14.21 | 14.74 | 20.77 | 24.48 | 2.96 | 67.68 | `C:\VAST2_UPLOAD\out\compare_rabitq2_vs_fp32_m32\20261104_104639\RR768_IVF4096_RaBitQ2_nprobe32.npz` |
| CPU | RR768_IVF4096_RaBitQ2 | `nprobe=512` | 0.896 | 81.50 | 81.00 | 82.30 | 84.02 | 84.70 | 61.81 | 63.03 | 74.24 | 81.78 | 12.61 | 15.86 | `C:\VAST2_UPLOAD\out\compare_rabitq2_vs_fp32_m32\20261104_104639\RR768_IVF4096_RaBitQ2_nprobe512.npz` |
| CPU | RR768_IVF4096_RaBitQ2 | `nprobe=64` | 0.896 | 77.50 | 76.90 | 78.05 | 80.08 | 79.65 | 18.21 | 20.81 | 32.99 | 64.15 | 4.16 | 48.02 | `C:\VAST2_UPLOAD\out\compare_rabitq2_vs_fp32_m32\20261104_104639\RR768_IVF4096_RaBitQ2_nprobe64.npz` |
| CPU | RR768_IVF4096_RaBitQ3 | `nprobe=1024` | 1.281 | 88.50 | 88.80 | 88.60 | 91.27 | 92.02 | 121.52 | 122.53 | 139.19 | 145.73 | 24.51 | 8.16 | `C:\VAST2_UPLOAD\out\compare_rabitq3_vs_fp32_m32\20261104_105652\RR768_IVF4096_RaBitQ3_nprobe1024.npz` |
| CPU | RR768_IVF4096_RaBitQ3 | `nprobe=128` | 1.281 | 85.50 | 85.80 | 86.15 | 89.19 | 88.64 | 24.52 | 25.43 | 32.99 | 37.88 | 5.09 | 39.31 | `C:\VAST2_UPLOAD\out\compare_rabitq3_vs_fp32_m32\20261104_105652\RR768_IVF4096_RaBitQ3_nprobe128.npz` |
| CPU | RR768_IVF4096_RaBitQ3 | `nprobe=256` | 1.281 | 87.00 | 87.60 | 87.70 | 90.64 | 90.70 | 40.82 | 44.87 | 67.34 | 76.73 | 8.98 | 22.28 | `C:\VAST2_UPLOAD\out\compare_rabitq3_vs_fp32_m32\20261104_105652\RR768_IVF4096_RaBitQ3_nprobe256.npz` |
| CPU | RR768_IVF4096_RaBitQ3 | `nprobe=32` | 1.281 | 79.50 | 80.10 | 80.75 | 82.90 | 79.95 | 15.01 | 16.24 | 25.34 | 29.20 | 3.25 | 61.48 | `C:\VAST2_UPLOAD\out\compare_rabitq3_vs_fp32_m32\20261104_105652\RR768_IVF4096_RaBitQ3_nprobe32.npz` |
| CPU | RR768_IVF4096_RaBitQ3 | `nprobe=512` | 1.281 | 88.00 | 88.70 | 88.50 | 91.11 | 91.69 | 68.12 | 72.66 | 98.86 | 129.66 | 14.53 | 13.76 | `C:\VAST2_UPLOAD\out\compare_rabitq3_vs_fp32_m32\20261104_105652\RR768_IVF4096_RaBitQ3_nprobe512.npz` |
| CPU | RR768_IVF4096_RaBitQ3 | `nprobe=64` | 1.281 | 83.00 | 83.30 | 83.95 | 86.45 | 85.03 | 17.15 | 17.86 | 24.37 | 26.69 | 3.58 | 55.93 | `C:\VAST2_UPLOAD\out\compare_rabitq3_vs_fp32_m32\20261104_105652\RR768_IVF4096_RaBitQ3_nprobe64.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=1024` | 1.666 | 90.50 | 93.50 | 92.80 | 94.93 | 95.41 | 129.60 | 130.33 | 146.71 | 157.19 | 26.07 | 7.67 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102700\RR768_IVF4096_RaBitQ4_nprobe1024.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=128` | 1.666 | 87.00 | 90.40 | 90.10 | 92.41 | 91.41 | 25.80 | 26.80 | 34.53 | 38.84 | 5.36 | 37.30 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102700\RR768_IVF4096_RaBitQ4_nprobe128.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=256` | 1.666 | 89.00 | 92.50 | 91.80 | 94.14 | 93.90 | 40.44 | 42.22 | 52.72 | 58.28 | 8.45 | 23.68 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102700\RR768_IVF4096_RaBitQ4_nprobe256.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=32` | 1.666 | 81.00 | 84.40 | 84.25 | 85.36 | 81.58 | 13.18 | 13.62 | 17.98 | 19.87 | 2.73 | 73.30 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102700\RR768_IVF4096_RaBitQ4_nprobe32.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=512` | 1.666 | 90.00 | 93.40 | 92.60 | 94.75 | 95.01 | 69.95 | 71.57 | 84.64 | 93.49 | 14.32 | 13.97 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102700\RR768_IVF4096_RaBitQ4_nprobe512.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=64` | 1.666 | 84.50 | 88.10 | 87.55 | 89.40 | 87.22 | 17.63 | 18.39 | 24.24 | 27.91 | 3.68 | 54.34 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102700\RR768_IVF4096_RaBitQ4_nprobe64.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=1024` | 1.666 | 90.50 | 93.50 | 92.80 | 94.93 | 95.41 | 129.21 | 130.19 | 147.24 | 156.92 | 26.04 | 7.68 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102841\RR768_IVF4096_RaBitQ4_nprobe1024.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=128` | 1.666 | 87.00 | 90.40 | 90.10 | 92.41 | 91.41 | 26.44 | 27.50 | 36.39 | 39.98 | 5.50 | 36.34 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102841\RR768_IVF4096_RaBitQ4_nprobe128.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=256` | 1.666 | 89.00 | 92.50 | 91.80 | 94.14 | 93.90 | 41.59 | 42.85 | 53.45 | 58.31 | 8.57 | 23.33 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102841\RR768_IVF4096_RaBitQ4_nprobe256.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=32` | 1.666 | 81.00 | 84.40 | 84.25 | 85.36 | 81.58 | 13.71 | 14.25 | 19.74 | 20.94 | 2.85 | 70.08 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102841\RR768_IVF4096_RaBitQ4_nprobe32.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=512` | 1.666 | 90.00 | 93.40 | 92.60 | 94.75 | 95.01 | 73.32 | 78.70 | 98.69 | 198.59 | 15.74 | 12.70 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102841\RR768_IVF4096_RaBitQ4_nprobe512.npz` |
| CPU | RR768_IVF4096_RaBitQ4 | `nprobe=64` | 1.666 | 84.50 | 88.10 | 87.55 | 89.40 | 87.22 | 18.18 | 19.16 | 25.89 | 28.60 | 3.84 | 52.15 | `C:\VAST2_UPLOAD\out\compare_rabitq4_vs_fp32_m32\20261104_102841\RR768_IVF4096_RaBitQ4_nprobe64.npz` |
| CPU | RR768_IVF4096_RaBitQ8 | `nprobe=1024` | 3.205 | 98.50 | 98.70 | 98.85 | 99.51 | 99.32 | 193.44 | 194.62 | 216.83 | 225.45 | 38.93 | 5.14 | `C:\VAST2_UPLOAD\out\compare_rabitq8_vs_fp32_m32\20261104_124840\RR768_IVF4096_RaBitQ8_nprobe1024.npz` |
| CPU | RR768_IVF4096_RaBitQ8 | `nprobe=128` | 3.205 | 94.50 | 94.60 | 95.20 | 95.92 | 93.77 | 33.65 | 34.72 | 43.65 | 47.11 | 6.95 | 28.79 | `C:\VAST2_UPLOAD\out\compare_rabitq8_vs_fp32_m32\20261104_124840\RR768_IVF4096_RaBitQ8_nprobe128.npz` |
| CPU | RR768_IVF4096_RaBitQ8 | `nprobe=256` | 3.205 | 97.00 | 97.30 | 97.55 | 98.18 | 97.00 | 56.45 | 58.06 | 69.37 | 76.23 | 11.62 | 17.22 | `C:\VAST2_UPLOAD\out\compare_rabitq8_vs_fp32_m32\20261104_124840\RR768_IVF4096_RaBitQ8_nprobe256.npz` |
| CPU | RR768_IVF4096_RaBitQ8 | `nprobe=32` | 3.205 | 87.50 | 87.60 | 88.35 | 87.66 | 82.72 | 15.48 | 15.76 | 19.99 | 21.78 | 3.16 | 63.37 | `C:\VAST2_UPLOAD\out\compare_rabitq8_vs_fp32_m32\20261104_124840\RR768_IVF4096_RaBitQ8_nprobe32.npz` |
| CPU | RR768_IVF4096_RaBitQ8 | `nprobe=512` | 3.205 | 98.00 | 98.50 | 98.65 | 99.18 | 98.70 | 102.93 | 104.00 | 119.09 | 128.54 | 20.80 | 9.61 | `C:\VAST2_UPLOAD\out\compare_rabitq8_vs_fp32_m32\20261104_124840\RR768_IVF4096_RaBitQ8_nprobe512.npz` |
| CPU | RR768_IVF4096_RaBitQ8 | `nprobe=64` | 3.205 | 91.50 | 91.70 | 92.15 | 92.35 | 88.94 | 21.71 | 22.47 | 28.68 | 31.50 | 4.50 | 44.48 | `C:\VAST2_UPLOAD\out\compare_rabitq8_vs_fp32_m32\20261104_124840\RR768_IVF4096_RaBitQ8_nprobe64.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=128` | 1.258 | 72.00 | 76.70 | 76.35 | 78.47 | 80.30 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_220455\RR768_IVF4096_RQ256x8_nprobe128.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=256` | 1.258 | 73.50 | 77.90 | 77.80 | 79.73 | 81.95 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_220455\RR768_IVF4096_RQ256x8_nprobe256.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=32` | 1.258 | 68.50 | 72.30 | 72.00 | 73.92 | 73.74 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_220455\RR768_IVF4096_RQ256x8_nprobe32.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=64` | 1.258 | 70.50 | 74.50 | 74.50 | 76.58 | 77.67 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_220455\RR768_IVF4096_RQ256x8_nprobe64.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=1024` | 1.258 | 75.00 | 79.00 | 78.55 | 80.25 | 83.03 | 22436.29 | 23895.08 | 26311.76 | 27907.94 | 4779.02 | 0.04 | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_234644\RR768_IVF4096_RQ256x8_nprobe1024.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=128` | 1.258 | 72.00 | 76.70 | 76.35 | 78.47 | 80.30 | 3274.32 | 3273.79 | 3395.80 | 3606.62 | 654.76 | 0.31 | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_234644\RR768_IVF4096_RQ256x8_nprobe128.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=256` | 1.258 | 73.50 | 77.90 | 77.80 | 79.73 | 81.95 | 6487.86 | 6523.27 | 6687.18 | 7110.85 | 1304.66 | 0.15 | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_234644\RR768_IVF4096_RQ256x8_nprobe256.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=32` | 1.258 | 68.50 | 72.30 | 72.00 | 73.92 | 73.74 | 812.89 | 830.13 | 907.24 | 957.24 | 166.03 | 1.20 | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_234644\RR768_IVF4096_RQ256x8_nprobe32.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=512` | 1.258 | 74.50 | 78.80 | 78.50 | 80.18 | 82.74 | 12997.78 | 13115.97 | 13566.37 | 16877.48 | 2623.20 | 0.08 | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_234644\RR768_IVF4096_RQ256x8_nprobe512.npz` |
| CPU | RR768_IVF4096_RQ256x8 | `nprobe=64` | 1.258 | 70.50 | 74.50 | 74.50 | 76.58 | 77.67 | 1615.91 | 1650.11 | 1787.36 | 1964.82 | 330.03 | 0.61 | `C:\VAST2_UPLOAD\out\compare_rq256x8_vs_fp32_m32\20261104_234644\RR768_IVF4096_RQ256x8_nprobe64.npz` |
| CPU | sq8_m16 | `ef=1024` | 3.657 | 92.00 | 94.10 | 93.30 | 94.89 | 92.76 | 27.21 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m16\20261031_205736\sq8_m16_ef1024_results.npz` |
| CPU | sq8_m16 | `ef=16384` | 3.657 | 96.50 | 98.20 | 97.55 | 98.67 | 98.71 | 796.39 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m16\20261031_205736\sq8_m16_ef16384_results.npz` |
| CPU | sq8_m16 | `ef=2048` | 3.657 | 95.50 | 97.40 | 96.50 | 97.80 | 96.91 | 50.14 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m16\20261031_205736\sq8_m16_ef2048_results.npz` |
| CPU | sq8_m16 | `ef=32768` | 3.657 | 96.50 | 98.30 | 97.60 | 98.68 | 98.77 | 2817.71 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m16\20261031_205736\sq8_m16_ef32768_results.npz` |
| CPU | sq8_m16 | `ef=4096` | 3.657 | 96.50 | 98.10 | 97.35 | 98.49 | 98.14 | 104.45 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m16\20261031_205736\sq8_m16_ef4096_results.npz` |
| CPU | sq8_m16 | `ef=512` | 3.657 | 84.50 | 88.50 | 87.80 | 89.39 | 84.36 | 15.58 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m16\20261031_205736\sq8_m16_ef512_results.npz` |
| CPU | sq8_m16 | `ef=8192` | 3.657 | 96.50 | 98.20 | 97.55 | 98.62 | 98.53 | 258.34 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m16\20261031_205736\sq8_m16_ef8192_results.npz` |
| CPU | sq8_m8 | `ef=1024` | 3.402 | 84.50 | 87.50 | 88.05 | 90.01 | 85.75 | 18.56 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_131633\sq8_m8_ef1024_results.npz` |
| CPU | sq8_m8 | `ef=16384` | 3.402 | 96.50 | 97.80 | 97.20 | 98.26 | 98.12 | 695.07 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_131633\sq8_m8_ef16384_results.npz` |
| CPU | sq8_m8 | `ef=2048` | 3.402 | 92.00 | 94.40 | 93.85 | 95.57 | 93.28 | 35.78 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_131633\sq8_m8_ef2048_results.npz` |
| CPU | sq8_m8 | `ef=32768` | 3.402 | 96.50 | 98.00 | 97.35 | 98.39 | 98.38 | 2610.73 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_131633\sq8_m8_ef32768_results.npz` |
| CPU | sq8_m8 | `ef=4096` | 3.402 | 94.50 | 96.50 | 96.25 | 97.57 | 96.44 | 78.61 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_131633\sq8_m8_ef4096_results.npz` |
| CPU | sq8_m8 | `ef=512` | 3.402 | 74.50 | 78.10 | 78.80 | 80.18 | 72.36 | 10.35 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_131633\sq8_m8_ef512_results.npz` |
| CPU | sq8_m8 | `ef=8192` | 3.402 | 96.00 | 97.40 | 96.90 | 98.00 | 97.59 | 207.38 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_131633\sq8_m8_ef8192_results.npz` |
| CPU | sq8_m8 | `ef=1024` | 3.402 | 84.50 | 87.50 | 88.05 | 90.01 | 85.75 | 18.56 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_164201\sq8_m8_ef1024_results.npz` |
| CPU | sq8_m8 | `ef=16384` | 3.402 | 96.50 | 97.80 | 97.20 | 98.26 | 98.12 | 695.07 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_164201\sq8_m8_ef16384_results.npz` |
| CPU | sq8_m8 | `ef=2048` | 3.402 | 92.00 | 94.40 | 93.85 | 95.57 | 93.28 | 35.78 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_164201\sq8_m8_ef2048_results.npz` |
| CPU | sq8_m8 | `ef=32768` | 3.402 | 96.50 | 98.00 | 97.35 | 98.39 | 98.38 | 2610.73 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_164201\sq8_m8_ef32768_results.npz` |
| CPU | sq8_m8 | `ef=4096` | 3.402 | 94.50 | 96.50 | 96.25 | 97.57 | 96.44 | 78.61 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_164201\sq8_m8_ef4096_results.npz` |
| CPU | sq8_m8 | `ef=512` | 3.402 | 74.50 | 78.10 | 78.80 | 80.18 | 72.36 | 10.35 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_164201\sq8_m8_ef512_results.npz` |
| CPU | sq8_m8 | `ef=8192` | 3.402 | 96.00 | 97.40 | 96.90 | 98.00 | 97.59 | 207.38 | — | — | — | — | — | `C:\VAST2_UPLOAD\out\compare_sq8_m8\20261101_164201\sq8_m8_ef8192_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=1000` | 13.420 | 98.00 | 97.40 | 97.55 | 98.01 | 95.37 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef1000_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=1024` | 13.420 | 98.00 | 97.40 | 97.55 | 98.05 | 95.44 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef1024_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=128` | 13.420 | 94.50 | 93.70 | 94.00 | 94.25 | 88.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef128_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=16384` | 13.420 | 98.50 | 99.50 | 99.55 | 99.91 | 99.91 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef16384_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=2048` | 13.420 | 98.50 | 99.30 | 99.00 | 99.28 | 98.23 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef2048_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=256` | 13.420 | 94.50 | 93.70 | 94.00 | 94.25 | 88.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef256_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=32768` | 13.420 | 98.50 | 99.60 | 99.60 | 99.94 | 99.96 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef32768_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=4096` | 13.420 | 98.50 | 99.50 | 99.45 | 99.74 | 99.36 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef4096_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=500` | 13.420 | 94.50 | 93.70 | 94.00 | 94.25 | 88.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef500_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=512` | 13.420 | 94.50 | 93.70 | 94.00 | 94.29 | 88.91 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef512_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=64` | 13.420 | 94.50 | 93.70 | 94.00 | 94.25 | 88.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef64_results.npz` |
| CPU | FP32_HNSW_M32 | `ef=8192` | 13.420 | 98.50 | 99.50 | 99.50 | 99.87 | 99.76 | — | — | — | — | — | — | `C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\fp32_ef8192_results.npz` |
| GPU | PQ192_f025_P104 | `nprobe=1024` | 0.825 | 65.00 | 67.90 | 69.30 | 71.47 | 74.83 | — | — | — | — | — | — | `PQ192_f025_P104_nprobe1024_top500_u32.raw` |
| GPU | PQ192_f025_P104 | `nprobe=128` | 0.825 | 63.00 | 67.00 | 68.45 | 70.62 | 73.91 | — | — | — | — | — | — | `PQ192_f025_P104_nprobe128_top500_u32.raw` |
| GPU | PQ192_f025_P104 | `nprobe=256` | 0.825 | 63.50 | 67.30 | 68.90 | 71.07 | 74.46 | — | — | — | — | — | — | `PQ192_f025_P104_nprobe256_top500_u32.raw` |
| GPU | PQ192_f025_P104 | `nprobe=32` | 0.825 | 61.00 | 64.90 | 66.05 | 68.08 | 70.50 | — | — | — | — | — | — | `PQ192_f025_P104_nprobe32_top500_u32.raw` |
| GPU | PQ192_f025_P104 | `nprobe=512` | 0.825 | 64.00 | 67.60 | 69.20 | 71.33 | 74.73 | — | — | — | — | — | — | `PQ192_f025_P104_nprobe512_top500_u32.raw` |
| GPU | PQ192_f025_P104 | `nprobe=64` | 0.825 | 63.00 | 66.30 | 67.55 | 69.72 | 72.63 | — | — | — | — | — | — | `PQ192_f025_P104_nprobe64_top500_u32.raw` |
| GPU | PQ192_f050_P104 | `nprobe=1024` | 0.825 | 68.00 | 69.00 | 69.05 | 71.29 | 74.69 | — | — | — | — | — | — | `PQ192_f050_P104_nprobe1024_top500_u32.raw` |
| GPU | PQ192_f050_P104 | `nprobe=128` | 0.825 | 67.00 | 67.70 | 68.30 | 70.55 | 73.72 | — | — | — | — | — | — | `PQ192_f050_P104_nprobe128_top500_u32.raw` |
| GPU | PQ192_f050_P104 | `nprobe=256` | 0.825 | 67.50 | 68.50 | 68.65 | 71.01 | 74.33 | — | — | — | — | — | — | `PQ192_f050_P104_nprobe256_top500_u32.raw` |
| GPU | PQ192_f050_P104 | `nprobe=32` | 0.825 | 65.00 | 65.20 | 65.60 | 68.15 | 70.32 | — | — | — | — | — | — | `PQ192_f050_P104_nprobe32_top500_u32.raw` |
| GPU | PQ192_f050_P104 | `nprobe=512` | 0.825 | 67.50 | 68.70 | 68.85 | 71.20 | 74.59 | — | — | — | — | — | — | `PQ192_f050_P104_nprobe512_top500_u32.raw` |
| GPU | PQ192_f050_P104 | `nprobe=64` | 0.825 | 66.00 | 66.20 | 67.05 | 69.72 | 72.59 | — | — | — | — | — | — | `PQ192_f050_P104_nprobe64_top500_u32.raw` |
| GPU | PQ384_f025_P104 | `nprobe=1024` | 1.606 | 85.00 | 88.20 | 88.30 | 91.19 | 92.30 | — | — | — | — | — | — | `PQ384_f025_P104_nprobe1024_top500_u32.raw` |
| GPU | PQ384_f025_P104 | `nprobe=128` | 1.606 | 84.00 | 86.70 | 86.95 | 89.94 | 90.55 | — | — | — | — | — | — | `PQ384_f025_P104_nprobe128_top500_u32.raw` |
| GPU | PQ384_f025_P104 | `nprobe=256` | 1.606 | 84.50 | 87.60 | 87.85 | 90.73 | 91.59 | — | — | — | — | — | — | `PQ384_f025_P104_nprobe256_top500_u32.raw` |
| GPU | PQ384_f025_P104 | `nprobe=32` | 1.606 | 80.50 | 83.20 | 82.70 | 85.31 | 84.55 | — | — | — | — | — | — | `PQ384_f025_P104_nprobe32_top500_u32.raw` |
| GPU | PQ384_f025_P104 | `nprobe=512` | 1.606 | 84.50 | 88.00 | 88.15 | 91.10 | 92.09 | — | — | — | — | — | — | `PQ384_f025_P104_nprobe512_top500_u32.raw` |
| GPU | PQ384_f025_P104 | `nprobe=64` | 1.606 | 82.50 | 85.40 | 85.40 | 88.38 | 88.47 | — | — | — | — | — | — | `PQ384_f025_P104_nprobe64_top500_u32.raw` |
| GPU | PQ384_f050_P104 | `nprobe=1024` | 1.606 | 84.00 | 88.80 | 89.60 | 91.66 | 92.51 | — | — | — | — | — | — | `PQ384_f050_P104_nprobe1024_top500_u32.raw` |
| GPU | PQ384_f050_P104 | `nprobe=128` | 1.606 | 83.50 | 87.10 | 88.05 | 90.18 | 90.64 | — | — | — | — | — | — | `PQ384_f050_P104_nprobe128_top500_u32.raw` |
| GPU | PQ384_f050_P104 | `nprobe=256` | 1.606 | 84.00 | 87.80 | 88.95 | 91.09 | 91.79 | — | — | — | — | — | — | `PQ384_f050_P104_nprobe256_top500_u32.raw` |
| GPU | PQ384_f050_P104 | `nprobe=32` | 1.606 | 80.50 | 83.10 | 83.95 | 85.36 | 84.45 | — | — | — | — | — | — | `PQ384_f050_P104_nprobe32_top500_u32.raw` |
| GPU | PQ384_f050_P104 | `nprobe=512` | 1.606 | 84.00 | 88.40 | 89.25 | 91.42 | 92.29 | — | — | — | — | — | — | `PQ384_f050_P104_nprobe512_top500_u32.raw` |
| GPU | PQ384_f050_P104 | `nprobe=64` | 1.606 | 83.00 | 85.50 | 86.60 | 88.52 | 88.49 | — | — | — | — | — | — | `PQ384_f050_P104_nprobe64_top500_u32.raw` |
| GPU | PQ96_f025_P104 | `nprobe=1024` | 0.435 | 41.50 | 41.00 | 41.85 | 45.85 | 52.95 | — | — | — | — | — | — | `PQ96_f025_P104_nprobe1024_top500_u32.raw` |
| GPU | PQ96_f025_P104 | `nprobe=128` | 0.435 | 41.50 | 40.50 | 41.50 | 45.63 | 52.69 | — | — | — | — | — | — | `PQ96_f025_P104_nprobe128_top500_u32.raw` |
| GPU | PQ96_f025_P104 | `nprobe=256` | 0.435 | 41.50 | 40.70 | 41.70 | 45.80 | 52.87 | — | — | — | — | — | — | `PQ96_f025_P104_nprobe256_top500_u32.raw` |
| GPU | PQ96_f025_P104 | `nprobe=32` | 0.435 | 41.00 | 39.70 | 40.70 | 44.76 | 51.38 | — | — | — | — | — | — | `PQ96_f025_P104_nprobe32_top500_u32.raw` |
| GPU | PQ96_f025_P104 | `nprobe=512` | 0.435 | 41.00 | 40.90 | 41.75 | 45.82 | 52.93 | — | — | — | — | — | — | `PQ96_f025_P104_nprobe512_top500_u32.raw` |
| GPU | PQ96_f025_P104 | `nprobe=64` | 0.435 | 41.00 | 40.30 | 41.25 | 45.33 | 52.24 | — | — | — | — | — | — | `PQ96_f025_P104_nprobe64_top500_u32.raw` |
| GPU | PQ96_f050_CMP90 | `nprobe=1024` | 0.435 | 36.00 | 41.10 | 41.60 | 45.54 | 52.45 | — | — | — | — | — | — | `PQ96_f050_CMP90_nprobe1024_top500_u32.raw` |
| GPU | PQ96_f050_CMP90 | `nprobe=128` | 0.435 | 35.00 | 40.90 | 41.15 | 45.20 | 52.17 | — | — | — | — | — | — | `PQ96_f050_CMP90_nprobe128_top500_u32.raw` |
| GPU | PQ96_f050_CMP90 | `nprobe=256` | 0.435 | 36.00 | 40.90 | 41.40 | 45.38 | 52.34 | — | — | — | — | — | — | `PQ96_f050_CMP90_nprobe256_top500_u32.raw` |
| GPU | PQ96_f050_CMP90 | `nprobe=32` | 0.435 | 34.50 | 39.70 | 39.95 | 44.14 | 51.00 | — | — | — | — | — | — | `PQ96_f050_CMP90_nprobe32_top500_u32.raw` |
| GPU | PQ96_f050_CMP90 | `nprobe=512` | 0.435 | 36.00 | 41.00 | 41.45 | 45.44 | 52.41 | — | — | — | — | — | — | `PQ96_f050_CMP90_nprobe512_top500_u32.raw` |
| GPU | PQ96_f050_CMP90 | `nprobe=64` | 0.435 | 35.00 | 40.30 | 40.60 | 44.87 | 51.77 | — | — | — | — | — | — | `PQ96_f050_CMP90_nprobe64_top500_u32.raw` |
| GPU | PQ96_f050_P104 | `nprobe=1024` | 0.435 | 40.50 | 44.00 | 44.50 | 47.30 | 53.15 | — | — | — | — | — | — | `PQ96_f050_P104_nprobe1024_top500_u32.raw` |
| GPU | PQ96_f050_P104 | `nprobe=128` | 0.435 | 40.00 | 43.70 | 44.20 | 47.10 | 52.91 | — | — | — | — | — | — | `PQ96_f050_P104_nprobe128_top500_u32.raw` |
| GPU | PQ96_f050_P104 | `nprobe=256` | 0.435 | 40.50 | 43.90 | 44.35 | 47.20 | 53.05 | — | — | — | — | — | — | `PQ96_f050_P104_nprobe256_top500_u32.raw` |
| GPU | PQ96_f050_P104 | `nprobe=32` | 0.435 | 38.50 | 42.40 | 43.25 | 46.39 | 51.70 | — | — | — | — | — | — | `PQ96_f050_P104_nprobe32_top500_u32.raw` |
| GPU | PQ96_f050_P104 | `nprobe=512` | 0.435 | 40.50 | 44.00 | 44.45 | 47.30 | 53.12 | — | — | — | — | — | — | `PQ96_f050_P104_nprobe512_top500_u32.raw` |
| GPU | PQ96_f050_P104 | `nprobe=64` | 0.435 | 39.00 | 43.00 | 43.70 | 46.85 | 52.49 | — | — | — | — | — | — | `PQ96_f050_P104_nprobe64_top500_u32.raw` |


**Coverage:** 120/198 rows have some sourced latency (including historical approximate timings); 78 have no matching timing. Counts include repeated experiments and older cached reference files. Do not infer search speed from the percentages.

## 6. Build performance — archival measured data

Original instrumented measurements are reproduced below, retaining the distinction between a complete wall clock, an instrumented phase sum and a resumed build. No new training was performed.

# 4. Build speed

| Index | Train | Add | Add speed | Serialize / write | Total / core build |
|---|---:|---:|---:|---:|---:|
| SQ8 HNSW M32 | — | — | — | — | **4.714 h** |
| SQ8 HNSW M16 | — | 228.525 min | 313.9 vec/s | — | **3.989 h current run** |
| SQ8 HNSW M8 | — | 151.210 min | 474.4 vec/s | — | ≥151 min |
| SQ4 HNSW M32 | — | 277.438 min | 258.6 vec/s | — | **4.722 h current run** |
| CPU PQ192 | ~12.2 min* | 10.43 min | 6,877.9 vec/s | 4.39 s | **~22 min*** |
| CPU OPQ192 | 49.84 min | 10.75 min | 6,673.0 vec/s | 4.09 s | **60.67 min** |
| CPU PQ384 | ~21 min* | 16.85 min | 4,257.9 vec/s | 8.06 s | **~38 min*** |
| CPU OPQ384 | ~95.1 min* | 17.77 min | 4,037.2 vec/s | 8.45 s | **~113 min*** |
| CPU OPQ128 | Not supplied | Not supplied | Not supplied | Verified | Not supplied |
| CPU OPQ256 | Not supplied | Not supplied | Not supplied | Verified | Not supplied |
| CPU PQ256 | Not supplied | Not supplied | Not supplied | 3.6 s final checkpoint write | Not supplied |
| CPU IVF-SQ4 | Not supplied | Not supplied | Not supplied | Verified | Not supplied |
| CPU RQ256×8 | Not supplied | Not supplied | Not supplied | Final checkpoint 4.5 s | Not supplied |
| CPU RaBitQ1 | 62.81 s | 208.47 s | 20648.4 vec/s | 1.71 s | **4.56 min measured phases** |
| CPU RaBitQ2 | 89.11 s | 325.16 s | 13237.9 vec/s | 3.44 s | **6.98 min measured phases** |
| CPU RaBitQ3 | 61.95 s | 404.38 s | 10644.7 vec/s | 6.58 s | **7.90 min measured phases** |
| CPU RaBitQ4 | 62.17 s | 466.89 s | 9219.4 vec/s | 6.58 s | **8.96 min measured phases** |
| GPU PQ96 f025 P104 | 79.323 s | 98.3 s | 43,787 vec/s | 3.1 s | **3.05 min wall** |
| GPU PQ96 f050 P104 | 180.216 s | 88.3 s | 48,731 vec/s | 3.0 s | **4.60 min wall** |
| GPU PQ96 f050 CMP90 | 257.842 s | 126.4 s | 34,051 vec/s | 2.9 s | **6.49 min wall** |
| GPU PQ192 f025 | 117.339 s | 89.6 s | 48,048 vec/s | 4.7 s | **3.53 min core** |
| GPU PQ192 f050 | 225.124 s | 96.1 s | 44,782 vec/s | 4.7 s | **5.43 min core** |
| GPU PQ384 f025 | 203.797 s | 94.9 s | 45,363 vec/s | 10.4 s | **5.15 min core** |
| GPU PQ384 f050 | 391.362 s | 94.8 s | 45,383 vec/s | 9.8 s | **8.27 min core** |

`*` CPU train/full values marked with an asterisk were reconstructed from logs/timestamps rather than preserved as a direct final timer.

---

### 6.1 Detailed build phases from original JSON summaries

| Factory | Train s (this run) | Add s (this run) | Write s | Verify s | Sum known phases s* | Source JSON |
|---|---|---|---|---|---|---|
| `IVF4096,PQ192` | — | 625.84 | 4.39 | 0.81 | 631.04 | `build_summary(3).json` |
| `IVF4096,PQ384` | — | 1010.94 | 8.06 | 1.21 | 1020.21 | `build_summary(1).json` |
| `OPQ192_768,IVF4096,PQ192` | — | 645.05 | 4.09 | 0.81 | 649.95 | `build_summary.json` |
| `OPQ384_768,IVF4096,PQ384` | — | 1066.22 | 8.45 | 1.09 | 1075.76 | `build_summary(2).json` |
| `RR768,IVF4096,RaBitQ1` | 62.81 | 208.47 | 1.71 | 0.44 | 273.42 | `build_summary(7).json` |
| `RR768,IVF4096,RaBitQ2` | 89.11 | 325.16 | 3.44 | 0.79 | 418.51 | `build_summary(5).json` |
| `RR768,IVF4096,RaBitQ3` | 61.95 | 404.38 | 6.58 | 0.99 | 473.89 | `build_summary(6).json` |
| `RR768,IVF4096,RaBitQ4` | 62.17 | 466.89 | 6.58 | 1.75 | 537.39 | `build_summary(4).json` |
| `RR768,IVF4096,RaBitQ8` | 91.51 | 1415.02 | 15.27 | 17.57 | 1539.37 | `build_summary(8).json` |


`*` **Sum known phases is NOT total build time** when train is missing, build was resumed, checkpoints consumed additional time, or stages occurred in separate runs. Historical end-to-end totals are retained in section 6.

## 7. GPU timing and reproducibility limitations

- Seven GPU variants (PQ96 ×3; PQ192 ×2; PQ384 ×2) and six `nprobe` settings each have directly recalculated exact-FP32 Top1/5/10/50/500. Each GPU raw output is 200×500 `uint32` IDs sorted by descending Inner Product score.
- **No validated GPU per-query latency, mean, percentiles or QPS**. Prior approximate ~0.9–5.6 ms readings are not used as confirmed timings (CUDA synchronization and workspace configuration were not consistently controlled).
- The GPU exporter yielded 42 files with 400,000 bytes each; export format alone does not validate cross-platform query ordering, normalization and ID alignment.
- There is an unresolved disagreement between **historical GPU-to-HNSW percentages** and percentages reconstructed from current saved GPU ID arrays. Exact-reference calculations are reported as measured, but this discrepancy requires cross-platform verification before making hard accuracy conclusions.

## 8. Reference details and caveats

- The previous reference, FP32 HNSW M32, is **approximate** even though stored vectors were FP32. In the saved low-effort legacy result, Top10 versus exact FP32 was **94.00%**. At `ef=32768`, the same index reached **99.60%** Top10 versus exact FP32. These are not contradictory; they used different search effort.
- Exact reference is a full-scanned FP32 top-500 derived from the same normalized embeddings; close FP32 score ties can have implementation-dependent ordering.
- Underlying raw quality table: `all_index_results_vs_exact_flatip.csv` (198 rows). Historical search/build report: `ann-benchmark-vs-hnsw-fp32.md`. Original per-run latency sources include `summary(2..10).csv` and `summary(20261105-072735).csv`; per-build sources include `build_summary*.json`.
- For strict reproducibility, archive the CSV and JSON files and the query-ID reference alongside the report. Do not conflate measured serialized GiB with peak RAM utilization.

## 9. Known measurement gaps

- **GPU search speed:** missing validated search-only CUDA timing for all six `nprobe` settings and all seven GPU indexes.
- **HNSW speed:** missing detailed distribution measurements for most FP32 M32 / SQ8 M32 / SQ4 M32 settings; historical approximate medians are explicitly marked.
- **Build phases:** missing original complete TRAIN/ADD timers for some IVF variants, and RQ256x8; available historical partial values are retained without extrapolation.
- **RAM:** only serialized index-file sizes are established, not RSS, GPU VRAM footprint during search, or peak build RAM.
- **Cross-platform audit:** GPU output reference-query and vector-ID order should be verified against Windows ground-truth preparation.
