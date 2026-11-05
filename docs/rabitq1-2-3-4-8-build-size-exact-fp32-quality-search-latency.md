# RaBitQ 1/2/3/4/8 — index construction, size, exact-FP32 retrieval quality and CPU search time

**Date:** 2026-11-05. **Scope:** measured results for a single 4,304,468-vector dataset. No new measurements were performed to create this document.

## Benchmark conditions

- Embeddings: `intfloat/multilingual-e5-base`; 4,304,468 normalized FP32 vectors; 768 dimensions; inner-product similarity.
- Query basis: the same 200 fixed queries, requested Top500.
- Search quality is intersection of returned IDs with *exact exhaustive FP32* TopK neighbors; Top1 requires matching first ID. This is **not** semantic relevance or answer accuracy.
- `Top1`, `Top5`, `Top10`, `Top50`, `Top500` are percentages (0–100).
- Hardware for the CPU runs: Ryzen 7 5800XT, 32 GiB RAM; FAISS CPU, four OpenMP threads in supplied scripts.
- All index file sizes are **on-disk serialized sizes**, not measured resident RAM. All time values are historical measured values, not predictions.
- A search time of 500 ms is considered here by the observed median and reported p99 only; this does not imply a worst-case bound.
- Build scripts use existing local Windows dataset paths; adjust those paths before running in another environment. Scripts are **original unmodified benchmark artifacts**.

## Index construction: complete recorded phases

| Variant | Bits/dimension | Index GiB | TRAIN s | ADD s | Vectors/s during ADD | Write s | Verify s | Sum measured phases s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RaBitQ1 | 1 | 0.4629 | 62.81 | 208.47 | 20,648 | 1.71 | 0.44 | 273.42 |
| RaBitQ2 | 2 | 0.8959 | 89.11 | 325.16 | 13,238 | 3.44 | 0.79 | 418.51 |
| RaBitQ3 | 3 | 1.2807 | 61.95 | 404.38 | 10,645 | 6.58 | 0.99 | 473.89 |
| RaBitQ4 | 4 | 1.6656 | 62.17 | 466.89 | 9,219 | 6.58 | 1.75 | 537.39 |
| RaBitQ8 | 8 | 3.2050 | 91.51 | 1415.02 | 3,042 | 15.27 | 17.57 | 1539.37 |

**Important:** The phase sum is not a separately verified total end-to-end wall clock; checkpoint operations, loading, setup and pauses may not be represented. The build JSON records each measured phase independently. All five summaries report training on 199,980 vectors, `nlist=4096`, `add_batch=10000`, four OpenMP threads.

## Quality and search latency for every recorded `nprobe`

| Variant | nprobe | Index GiB | Exact Top1 % | Top5 % | Top10 % | Top50 % | Top500 % | Median ms | Mean ms | p95 ms | p99 ms | 200-query batch s | QPS | Run ID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RaBitQ1 | 32 | 0.463 | 58.00 | 58.90 | 61.15 | 62.77 | 64.22 | 4.86 | 4.85 | 5.44 | 5.62 | 0.97 | 205.14 | 20261104_110337 |
| RaBitQ1 | 64 | 0.463 | 59.00 | 61.10 | 63.35 | 64.99 | 66.61 | 7.77 | 7.78 | 8.62 | 9.31 | 1.56 | 128.25 | 20261104_110337 |
| RaBitQ1 | 128 | 0.463 | 60.00 | 62.40 | 64.70 | 66.31 | 68.24 | 13.46 | 13.54 | 15.11 | 15.76 | 2.71 | 73.78 | 20261104_110337 |
| RaBitQ1 | 256 | 0.463 | 61.50 | 63.70 | 65.45 | 67.15 | 69.12 | 24.62 | 24.79 | 27.39 | 28.60 | 4.96 | 40.32 | 20261104_110337 |
| RaBitQ1 | 512 | 0.463 | 62.00 | 64.00 | 65.85 | 67.47 | 69.54 | 47.09 | 47.14 | 51.35 | 52.92 | 9.43 | 21.21 | 20261104_110337 |
| RaBitQ1 | 1024 | 0.463 | 62.50 | 64.10 | 65.95 | 67.53 | 69.66 | 91.84 | 91.82 | 98.41 | 100.31 | 18.37 | 10.89 | 20261104_110337 |
| RaBitQ2 | 32 | 0.896 | 75.00 | 74.20 | 75.35 | 77.17 | 75.61 | 14.21 | 14.74 | 20.77 | 24.48 | 2.96 | 67.68 | 20261104_104639 |
| RaBitQ2 | 64 | 0.896 | 77.50 | 76.90 | 78.05 | 80.08 | 79.65 | 18.21 | 20.81 | 32.99 | 64.15 | 4.16 | 48.02 | 20261104_104639 |
| RaBitQ2 | 128 | 0.896 | 78.50 | 78.60 | 80.20 | 82.38 | 82.44 | 23.48 | 24.30 | 31.25 | 35.87 | 4.86 | 41.12 | 20261104_104639 |
| RaBitQ2 | 256 | 0.896 | 80.50 | 80.20 | 81.65 | 83.55 | 83.94 | 36.59 | 37.96 | 47.24 | 52.76 | 7.60 | 26.33 | 20261104_104639 |
| RaBitQ2 | 512 | 0.896 | 81.50 | 81.00 | 82.30 | 84.02 | 84.70 | 61.81 | 63.03 | 74.24 | 81.78 | 12.61 | 15.86 | 20261104_104639 |
| RaBitQ2 | 1024 | 0.896 | 82.00 | 81.10 | 82.25 | 84.15 | 84.96 | 112.91 | 114.20 | 130.22 | 143.68 | 22.84 | 8.76 | 20261104_104639 |
| RaBitQ3 | 32 | 1.281 | 79.50 | 80.10 | 80.75 | 82.90 | 79.95 | 15.01 | 16.24 | 25.34 | 29.20 | 3.25 | 61.48 | 20261104_105652 |
| RaBitQ3 | 64 | 1.281 | 83.00 | 83.30 | 83.95 | 86.45 | 85.03 | 17.15 | 17.86 | 24.37 | 26.69 | 3.58 | 55.93 | 20261104_105652 |
| RaBitQ3 | 128 | 1.281 | 85.50 | 85.80 | 86.15 | 89.19 | 88.64 | 24.52 | 25.43 | 32.99 | 37.88 | 5.09 | 39.31 | 20261104_105652 |
| RaBitQ3 | 256 | 1.281 | 87.00 | 87.60 | 87.70 | 90.64 | 90.70 | 40.82 | 44.87 | 67.34 | 76.73 | 8.98 | 22.28 | 20261104_105652 |
| RaBitQ3 | 512 | 1.281 | 88.00 | 88.70 | 88.50 | 91.11 | 91.69 | 68.12 | 72.66 | 98.86 | 129.66 | 14.53 | 13.76 | 20261104_105652 |
| RaBitQ3 | 1024 | 1.281 | 88.50 | 88.80 | 88.60 | 91.27 | 92.02 | 121.52 | 122.53 | 139.19 | 145.73 | 24.51 | 8.16 | 20261104_105652 |
| RaBitQ4 | 32 | 1.666 | 81.00 | 84.40 | 84.25 | 85.36 | 81.58 | 13.18 | 13.62 | 17.98 | 19.87 | 2.73 | 73.30 | 20261104_102700 |
| RaBitQ4 | 32 | 1.666 | 81.00 | 84.40 | 84.25 | 85.36 | 81.58 | 13.71 | 14.25 | 19.74 | 20.94 | 2.85 | 70.08 | 20261104_102841 |
| RaBitQ4 | 64 | 1.666 | 84.50 | 88.10 | 87.55 | 89.40 | 87.22 | 17.63 | 18.39 | 24.24 | 27.91 | 3.68 | 54.34 | 20261104_102700 |
| RaBitQ4 | 64 | 1.666 | 84.50 | 88.10 | 87.55 | 89.40 | 87.22 | 18.18 | 19.16 | 25.89 | 28.60 | 3.84 | 52.15 | 20261104_102841 |
| RaBitQ4 | 128 | 1.666 | 87.00 | 90.40 | 90.10 | 92.41 | 91.41 | 25.80 | 26.80 | 34.53 | 38.84 | 5.36 | 37.30 | 20261104_102700 |
| RaBitQ4 | 128 | 1.666 | 87.00 | 90.40 | 90.10 | 92.41 | 91.41 | 26.44 | 27.50 | 36.39 | 39.98 | 5.50 | 36.34 | 20261104_102841 |
| RaBitQ4 | 256 | 1.666 | 89.00 | 92.50 | 91.80 | 94.14 | 93.90 | 40.44 | 42.22 | 52.72 | 58.28 | 8.45 | 23.68 | 20261104_102700 |
| RaBitQ4 | 256 | 1.666 | 89.00 | 92.50 | 91.80 | 94.14 | 93.90 | 41.59 | 42.85 | 53.45 | 58.31 | 8.57 | 23.33 | 20261104_102841 |
| RaBitQ4 | 512 | 1.666 | 90.00 | 93.40 | 92.60 | 94.75 | 95.01 | 69.95 | 71.57 | 84.64 | 93.49 | 14.32 | 13.97 | 20261104_102700 |
| RaBitQ4 | 512 | 1.666 | 90.00 | 93.40 | 92.60 | 94.75 | 95.01 | 73.32 | 78.70 | 98.69 | 198.59 | 15.74 | 12.70 | 20261104_102841 |
| RaBitQ4 | 1024 | 1.666 | 90.50 | 93.50 | 92.80 | 94.93 | 95.41 | 129.60 | 130.33 | 146.71 | 157.19 | 26.07 | 7.67 | 20261104_102700 |
| RaBitQ4 | 1024 | 1.666 | 90.50 | 93.50 | 92.80 | 94.93 | 95.41 | 129.21 | 130.19 | 147.24 | 156.92 | 26.04 | 7.68 | 20261104_102841 |
| RaBitQ8 | 32 | 3.205 | 87.50 | 87.60 | 88.35 | 87.66 | 82.72 | 15.48 | 15.76 | 19.99 | 21.78 | 3.16 | 63.37 | 20261104_124840 |
| RaBitQ8 | 64 | 3.205 | 91.50 | 91.70 | 92.15 | 92.35 | 88.94 | 21.71 | 22.47 | 28.68 | 31.50 | 4.50 | 44.48 | 20261104_124840 |
| RaBitQ8 | 128 | 3.205 | 94.50 | 94.60 | 95.20 | 95.92 | 93.77 | 33.65 | 34.72 | 43.65 | 47.11 | 6.95 | 28.79 | 20261104_124840 |
| RaBitQ8 | 256 | 3.205 | 97.00 | 97.30 | 97.55 | 98.18 | 97.00 | 56.45 | 58.06 | 69.37 | 76.23 | 11.62 | 17.22 | 20261104_124840 |
| RaBitQ8 | 512 | 3.205 | 98.00 | 98.50 | 98.65 | 99.18 | 98.70 | 102.93 | 104.00 | 119.09 | 128.54 | 20.80 | 9.61 | 20261104_124840 |
| RaBitQ8 | 1024 | 3.205 | 98.50 | 98.70 | 98.85 | 99.51 | 99.32 | 193.44 | 194.62 | 216.83 | 225.45 | 38.93 | 5.14 | 20261104_124840 |

## Practical choices under observed 500-ms search requirement

| Size limit | Selected index | nprobe | Exact Top10 % | Top500 % | Median ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| ≤ 0.5 GiB | RaBitQ1 | 1024 | 65.95 | 69.66 | 91.84 | 100.31 |
| ≤ 1.0 GiB | RaBitQ2 | 512 | 82.30 | 84.70 | 61.81 | 81.78 |
| ≤ 1.5 GiB | RaBitQ3 | 1024 | 88.60 | 92.02 | 121.52 | 145.73 |
| ≤ 2.0 GiB | RaBitQ4 | 1024 | 92.80 | 95.41 | 129.60 | 157.19 |
| ≤ 4.0 GiB | RaBitQ8 | 1024 | 98.85 | 99.32 | 193.44 | 225.45 |

These are **RaBitQ-family winners only**, not proof that the same index dominates *every* tested algorithm, every hardware platform or every query set. For the cross-algorithm comparison see `docs/ann-vs-flat-fp32-accuracy-index-size-build-time-search-latency.md` in the parent repository. All five settings satisfy the 500-ms condition by measured p99 on 200 queries; no worst-case SLA was measured.

## Reproducing and interpreting the original scripts

1. The five `build_ivf4096_rabitq*_cpu.py` scripts create indices from 66 existing `.npy` parts, with training, checkpointing, verification and final serialization. They are not self-contained without the original embeddings.
2. The five `benchmark_rabitq*_vs_fp32_m32_cpu.py` scripts execute the six-probe search sweeps and record latency and IDs. **Their baseline is the older FP32 HNSW M32 result**, not the exact ground truth.
3. Exact-FP32 percentages in this report were **recomputed later**, using persisted search result IDs and the exhaustive FP32 reference `flatip_exact_top500.npz`. The included original benchmark scripts do not themselves calculate these exact-FP32 percentages.
4. The original index binaries, source embedding parts, query arrays, exact neighbor array, and archived search `.npz` files are **not included** in this small GitHub package.
5. Local absolute paths, FAISS version and available CPU instruction sets can affect repeatability. Revalidate output after modifying any of these.

## Data sources and traceability

- `data/rabitq/build_summary_rabitq{1,2,3,4,8}.json`: original per-index training, ADD, write, verification timings and exact file bytes.
- `scripts/rabitq/`: original build and benchmark scripts (verbatim copies).
- `docs/ann-vs-flat-fp32-accuracy-index-size-build-time-search-latency.md`: main complete report with all algorithms, 198 archived measurements, original Windows source paths and other historical limitations.
- This standalone document retains **all 36 RaBitQ sweep rows**, including two independent RaBitQ4 search runs.
