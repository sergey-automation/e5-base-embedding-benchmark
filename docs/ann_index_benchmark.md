# ANN Index Benchmark: Size, Build Time, Search Quality, and Search Plateau

This document compares all ANN index variants that were actually built and tested on the same `multilingual-e5-base` embedding corpus.

The focus is on four practical questions:

1. How large is the index?
2. How much ranking quality is retained relative to the FP32 HNSW reference?
3. How long does the index take to build?
4. At what search parameter does additional search effort stop producing meaningful quality gains?

---

## Test basis

- Embedding model: `intfloat/multilingual-e5-base`
- Number of vectors: **4,304,468**
- Embedding dimension: **768**
- Source dtype: **FP32**
- Vectors: normalized
- Similarity: **Inner Product**
- Query set: **200 fixed queries**
- Returned candidates: **K = 500**
- IVF coarse lists: **4096**
- HNSW reference: **FP32 HNSW M32**
- HNSW search sweep: `efSearch = 512, 1024, 2048, 4096, 8192, 16384, 32768`
- IVF search sweep: `nprobe = 32, 64, 128, 256, 512, 1024`

### Quality metric

All `Top1`, `Top5`, `Top10`, `Top50`, and `Top500` percentages are ranking overlap against the existing **FP32 HNSW M32** result set.

They are **not brute-force exact recall**.

The FP32 HNSW M32 index is therefore treated as the **100% ranking reference** for this benchmark.

---

# 1. All tested indexes, grouped by size

Within each size group, the highest measured `Top10` quality is listed first.

## ~0.44 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| IVF4096, PQ96, train fraction 0.50 | P104 GPU | **445.0 MiB** | **44.50%** |
| IVF4096, PQ96, train fraction 0.25 | P104 GPU | **445.1 MiB** | **41.85%** |
| IVF4096, PQ96, train fraction 0.50 | CMP 90HX GPU | **445.1 MiB** | **41.60%** |

## ~0.81–0.83 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| OPQ192 + IVF4096, PQ192 | CPU | **0.816 GiB** | **71.75%** |
| IVF4096, PQ192, train fraction 0.25 | P104 GPU | **0.825 GiB** | **69.30%** |
| IVF4096, PQ192, train fraction 0.50 | P104 GPU | **0.825 GiB** | **69.15%** |
| IVF4096, PQ192 | CPU | **0.814 GiB** | **66.45%** |

## ~1.58–1.61 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| IVF4096, PQ384, train fraction 0.50 | P104 GPU | **1.606 GiB** | **89.85%** |
| IVF4096, PQ384, train fraction 0.25 | P104 GPU | **1.606 GiB** | **88.55%** |
| IVF4096, PQ384 | CPU | **1.584 GiB** | **86.30%** |
| OPQ384 + IVF4096, PQ384 | CPU | **1.586 GiB** | **85.20%** |

## ~2.6 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| SQ4 HNSW M32 | CPU | **2.630 GiB** | **78.20%** |

## ~3.4–4.2 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| SQ8 HNSW M16 | CPU | **3.657 GiB** | **97.30%** |
| SQ8 HNSW M32 | CPU | **4.170 GiB** | **97.25%** |
| SQ8 HNSW M8 | CPU | **3.402 GiB** | **97.00%** |

## Reference

| Index | Platform | Size | Quality |
|---|---|---:|---:|
| FP32 HNSW M32 | CPU | **13.42 GiB** | **100% reference** |

---

# 2. Global quality/size comparison

| Index | Platform | Size | Best Top10 |
|---|---|---:|---:|
| FP32 HNSW M32 | CPU | 13.42 GiB | 100.00% |
| SQ8 HNSW M16 | CPU | 3.657 GiB | 97.30% |
| SQ8 HNSW M32 | CPU | 4.170 GiB | 97.25% |
| SQ8 HNSW M8 | CPU | 3.402 GiB | 97.00% |
| GPU PQ384 f050 | P104 | 1.606 GiB | 89.85% |
| GPU PQ384 f025 | P104 | 1.606 GiB | 88.55% |
| CPU PQ384 | CPU | 1.584 GiB | 86.30% |
| CPU OPQ384 | CPU | 1.586 GiB | 85.20% |
| SQ4 HNSW M32 | CPU | 2.630 GiB | 78.20% |
| CPU OPQ192 | CPU | 0.816 GiB | 71.75% |
| GPU PQ192 f025 | P104 | 0.825 GiB | 69.30% |
| GPU PQ192 f050 | P104 | 0.825 GiB | 69.15% |
| CPU PQ192 | CPU | 0.814 GiB | 66.45% |
| GPU PQ96 f050 | P104 | 445.0 MiB | 44.50% |
| GPU PQ96 f025 | P104 | 445.1 MiB | 41.85% |
| GPU PQ96 f050 | CMP 90HX | 445.1 MiB | 41.60% |

---

# 3. Search-quality plateau table

This is the main table for seeing where extra search effort stops producing meaningful Top10 gains.

## HNSW

| Index | Search parameter | Top10 vs FP32 M32 | Gain from previous step |
|---|---:|---:|---:|
| SQ8 HNSW M32 | ef=512 | 90.00% | — |
|  | ef=1024 | 94.75% | +4.75 pp |
|  | ef=2048 | 96.25% | +1.50 pp |
|  | ef=4096 | 97.20% | +0.95 pp |
|  | ef=8192 | **97.25%** | +0.05 pp |
|  | ef=16384 | 97.25% | +0.00 pp |
|  | ef=32768 | 97.25% | +0.00 pp |
| SQ8 HNSW M16 | ef=512 | 85.80% | — |
|  | ef=1024 | 92.30% | +6.50 pp |
|  | ef=2048 | 96.20% | +3.90 pp |
|  | ef=4096 | 97.15% | +0.95 pp |
|  | ef=8192 | **97.30%** | +0.15 pp |
|  | ef=16384 | 97.25% | -0.05 pp |
|  | ef=32768 | 97.25% | +0.00 pp |
| SQ8 HNSW M8 | ef=512 | 77.50% | — |
|  | ef=1024 | 87.00% | +9.50 pp |
|  | ef=2048 | 93.45% | +6.45 pp |
|  | ef=4096 | 95.90% | +2.45 pp |
|  | ef=8192 | 96.60% | +0.70 pp |
|  | ef=16384 | 96.90% | +0.30 pp |
|  | ef=32768 | **97.00%** | +0.10 pp |
| SQ4 HNSW M32 | ef=512 | 72.40% | — |
|  | ef=1024 | 76.15% | +3.75 pp |
|  | ef=2048 | 78.00% | +1.85 pp |
|  | ef=4096 | 78.15% | +0.15 pp |
|  | ef=8192 | 78.15% | +0.00 pp |
|  | ef=16384 | **78.20%** | +0.05 pp |
|  | ef=32768 | 78.15% | -0.05 pp |

## IVF / PQ

| Index | Search parameter | Top10 vs FP32 M32 | Gain from previous step |
|---|---:|---:|---:|
| CPU PQ192 | nprobe=32 | 55.15% | — |
|  | 64 | 61.20% | +6.05 pp |
|  | 128 | 65.10% | +3.90 pp |
|  | 256 | 66.10% | +1.00 pp |
|  | 512 | **66.45%** | +0.35 pp |
|  | 1024 | 66.35% | -0.10 pp |
| CPU OPQ192 | nprobe=32 | 61.40% | — |
|  | 64 | 66.35% | +4.95 pp |
|  | 128 | 70.70% | +4.35 pp |
|  | 256 | 71.25% | +0.55 pp |
|  | 512 | 71.55% | +0.30 pp |
|  | 1024 | **71.75%** | +0.20 pp |
| GPU PQ192 f025 | nprobe=32 | 66.05% | — |
|  | 64 | 67.55% | +1.50 pp |
|  | 128 | 68.45% | +0.90 pp |
|  | 256 | 68.90% | +0.45 pp |
|  | 512 | 69.20% | +0.30 pp |
|  | 1024 | **69.30%** | +0.10 pp |
| GPU PQ192 f050 | nprobe=32 | 65.70% | — |
|  | 64 | 67.15% | +1.45 pp |
|  | 128 | 68.40% | +1.25 pp |
|  | 256 | 68.75% | +0.35 pp |
|  | 512 | 68.95% | +0.20 pp |
|  | 1024 | **69.15%** | +0.20 pp |
| CPU PQ384 | nprobe=32 | 76.40% | — |
|  | 64 | 82.95% | +6.55 pp |
|  | 128 | 84.80% | +1.85 pp |
|  | 256 | 85.55% | +0.75 pp |
|  | 512 | 86.05% | +0.50 pp |
|  | 1024 | **86.30%** | +0.25 pp |
| CPU OPQ384 | nprobe=32 | 77.00% | — |
|  | 64 | 81.90% | +4.90 pp |
|  | 128 | 83.75% | +1.85 pp |
|  | 256 | 84.70% | +0.95 pp |
|  | 512 | 85.10% | +0.40 pp |
|  | 1024 | **85.20%** | +0.10 pp |
| GPU PQ384 f025 | nprobe=32 | 82.85% | — |
|  | 64 | 85.65% | +2.80 pp |
|  | 128 | 87.20% | +1.55 pp |
|  | 256 | 88.10% | +0.90 pp |
|  | 512 | 88.40% | +0.30 pp |
|  | 1024 | **88.55%** | +0.15 pp |
| GPU PQ384 f050 | nprobe=32 | 84.10% | — |
|  | 64 | 86.80% | +2.70 pp |
|  | 128 | 88.30% | +1.50 pp |
|  | 256 | 89.20% | +0.90 pp |
|  | 512 | 89.50% | +0.30 pp |
|  | 1024 | **89.85%** | +0.35 pp |
| GPU PQ96 f025 | nprobe=32 | 40.70% | — |
|  | 64 | 41.25% | +0.55 pp |
|  | 128 | 41.50% | +0.25 pp |
|  | 256 | 41.70% | +0.20 pp |
|  | 512 | 41.75% | +0.05 pp |
|  | 1024 | **41.85%** | +0.10 pp |
| GPU PQ96 f050 P104 | nprobe=32 | 43.25% | — |
|  | 64 | 43.70% | +0.45 pp |
|  | 128 | 44.20% | +0.50 pp |
|  | 256 | 44.35% | +0.15 pp |
|  | 512 | 44.45% | +0.10 pp |
|  | 1024 | **44.50%** | +0.05 pp |
| GPU PQ96 f050 CMP90 | nprobe=32 | 39.95% | — |
|  | 64 | 40.60% | +0.65 pp |
|  | 128 | 41.15% | +0.55 pp |
|  | 256 | 41.40% | +0.25 pp |
|  | 512 | 41.45% | +0.05 pp |
|  | 1024 | **41.60%** | +0.15 pp |

### Practical plateau points

Using a practical definition of plateau as the point after which additional search effort gives only about `<=0.3–0.5 percentage points` per step:

| Index | Practical plateau begins around | Top10 near plateau |
|---|---:|---:|
| SQ8 HNSW M32 | ef≈4096–8192 | 97.20–97.25% |
| SQ8 HNSW M16 | ef≈4096–8192 | 97.15–97.30% |
| SQ8 HNSW M8 | ef≈8192–16384 | 96.60–96.90% |
| SQ4 HNSW M32 | ef≈4096 | 78.15% |
| CPU PQ192 | nprobe≈256–512 | 66.10–66.45% |
| CPU OPQ192 | nprobe≈256–512 | 71.25–71.55% |
| GPU PQ192 f025 | nprobe≈256–512 | 68.90–69.20% |
| GPU PQ192 f050 | nprobe≈256–512 | 68.75–68.95% |
| CPU PQ384 | nprobe≈256–512 | 85.55–86.05% |
| CPU OPQ384 | nprobe≈256–512 | 84.70–85.10% |
| GPU PQ384 f025 | nprobe≈256–512 | 88.10–88.40% |
| GPU PQ384 f050 | nprobe≈256–512 | 89.20–89.50% |
| GPU PQ96 f025 | nprobe≈64–128 | 41.25–41.50% |
| GPU PQ96 f050 P104 | nprobe≈128–256 | 44.20–44.35% |
| GPU PQ96 f050 CMP90 | nprobe≈128–256 | 41.15–41.40% |

---

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
| GPU PQ96 f025 P104 | 79.323 s | 98.3 s | 43,787 vec/s | 3.1 s | **3.05 min wall** |
| GPU PQ96 f050 P104 | 180.216 s | 88.3 s | 48,731 vec/s | 3.0 s | **4.60 min wall** |
| GPU PQ96 f050 CMP90 | 257.842 s | 126.4 s | 34,051 vec/s | 2.9 s | **6.49 min wall** |
| GPU PQ192 f025 | 117.339 s | 89.6 s | 48,048 vec/s | 4.7 s | **3.53 min core** |
| GPU PQ192 f050 | 225.124 s | 96.1 s | 44,782 vec/s | 4.7 s | **5.43 min core** |
| GPU PQ384 f025 | 203.797 s | 94.9 s | 45,363 vec/s | 10.4 s | **5.15 min core** |
| GPU PQ384 f050 | 391.362 s | 94.8 s | 45,383 vec/s | 9.8 s | **8.27 min core** |

`*` CPU train/full values marked with an asterisk were reconstructed from logs/timestamps rather than preserved as a direct final timer.

---

# 5. Search speed

## CPU HNSW

| Index | Search setting | Top10 | Median latency |
|---|---:|---:|---:|
| SQ8 HNSW M8 | ef=2048 | 93.45% | 35.78 ms |
| SQ8 HNSW M8 | ef=4096 | 95.90% | 78.61 ms |
| SQ8 HNSW M16 | ef=2048 | 96.20% | 50.14 ms |
| SQ8 HNSW M16 | ef=4096 | 97.15% | 104.45 ms |
| SQ8 HNSW M32 | ef=2048 | 96.25% | ~61 ms |
| SQ8 HNSW M32 | ef=4096 | 97.20% | ~131 ms |

## CPU IVF

| Index | Search setting | Top10 | Median latency |
|---|---:|---:|---:|
| CPU PQ192 | nprobe=128 | 65.10% | 29.9 ms |
| CPU OPQ192 | nprobe=128 | 70.70% | 33.4 ms |
| CPU PQ384 | nprobe=64 | 82.95% | 41.83 ms |
| CPU PQ384 | nprobe=128 | 84.80% | 80.67 ms |
| CPU OPQ384 | nprobe=128 | 83.75% | ~82.4 ms |
| CPU PQ384 | nprobe=1024 | 86.30% | 624.8 ms |

### GPU search latency

GPU ranking quality is valid, but final GPU latency is **not yet published as a comparable number**.

The corrected GPU quality benchmark performs CPU-side sorting of `(score, id)` after RAFT search, and RAFT 23.08 also warns that the default CUDA workspace resource may cause significant slowdown.

For this reason, the previously observed raw GPU latency values are intentionally excluded from the comparison table.

A final GPU latency measurement should time only:

- RAFT search
- CUDA synchronization
- no CPU sorting
- no debug output
- fixed workspace-resource configuration

---

# 6. Full GPU search-quality tables

## GPU PQ96 f025 — P104

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 |
|---:|---:|---:|---:|---:|---:|
| 32 | 41.50% | 39.70% | 40.70% | 44.76% | 51.383% |
| 64 | 41.50% | 40.30% | 41.25% | 45.33% | 52.247% |
| 128 | 42.00% | 40.50% | 41.50% | 45.63% | 52.694% |
| 256 | 42.00% | 40.70% | 41.70% | 45.80% | 52.870% |
| 512 | 41.50% | 40.90% | 41.75% | 45.82% | 52.933% |
| 1024 | 42.00% | 41.00% | 41.85% | 45.85% | 52.952% |

## GPU PQ96 f050 — P104

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 |
|---:|---:|---:|---:|---:|---:|
| 32 | 38.50% | 42.40% | 43.25% | 46.39% | 51.703% |
| 64 | 39.00% | 43.00% | 43.70% | 46.85% | 52.496% |
| 128 | 40.00% | 43.70% | 44.20% | 47.10% | 52.914% |
| 256 | 40.50% | 43.90% | 44.35% | 47.20% | 53.055% |
| 512 | 40.50% | 44.00% | 44.45% | 47.30% | 53.125% |
| 1024 | 40.50% | 44.00% | 44.50% | 47.30% | 53.151% |

## GPU PQ96 f050 — CMP 90HX

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 |
|---:|---:|---:|---:|---:|---:|
| 32 | 35.00% | 39.70% | 39.95% | 44.14% | 51.008% |
| 64 | 35.50% | 40.30% | 40.60% | 44.87% | 51.767% |
| 128 | 35.50% | 40.90% | 41.15% | 45.20% | 52.170% |
| 256 | 36.50% | 40.90% | 41.40% | 45.38% | 52.342% |
| 512 | 36.50% | 41.00% | 41.45% | 45.44% | 52.407% |
| 1024 | 36.50% | 41.10% | 41.60% | 45.54% | 52.454% |

## GPU PQ192 f025 — P104

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 |
|---:|---:|---:|---:|---:|---:|
| 32 | 61.50% | 64.90% | 66.05% | 68.08% | 70.505% |
| 64 | 63.50% | 66.30% | 67.55% | 69.72% | 72.633% |
| 128 | 63.50% | 67.00% | 68.45% | 70.62% | 73.916% |
| 256 | 64.00% | 67.30% | 68.90% | 71.07% | 74.465% |
| 512 | 64.50% | 67.60% | 69.20% | 71.33% | 74.737% |
| 1024 | 65.50% | 67.90% | 69.30% | 71.47% | 74.830% |

## GPU PQ192 f050 — P104

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 |
|---:|---:|---:|---:|---:|---:|
| 32 | 66.50% | 65.30% | 65.70% | 68.15% | 70.324% |
| 64 | 67.50% | 66.30% | 67.15% | 69.72% | 72.592% |
| 128 | 68.50% | 67.80% | 68.40% | 70.55% | 73.722% |
| 256 | 69.00% | 68.60% | 68.75% | 71.01% | 74.333% |
| 512 | 69.00% | 68.80% | 68.95% | 71.20% | 74.590% |
| 1024 | 69.50% | 69.10% | 69.15% | 71.29% | 74.685% |

## GPU PQ384 f025 — P104

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 |
|---:|---:|---:|---:|---:|---:|
| 32 | 82.00% | 83.50% | 82.85% | 85.31% | 84.558% |
| 64 | 84.00% | 85.70% | 85.65% | 88.38% | 88.473% |
| 128 | 85.50% | 86.90% | 87.20% | 89.94% | 90.545% |
| 256 | 86.00% | 87.70% | 88.10% | 90.73% | 91.592% |
| 512 | 86.00% | 88.10% | 88.40% | 91.10% | 92.081% |
| 1024 | 86.50% | 88.30% | 88.55% | 91.19% | 92.281% |

## GPU PQ384 f050 — P104

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 |
|---:|---:|---:|---:|---:|---:|
| 32 | 81.50% | 83.40% | 84.10% | 85.36% | 84.462% |
| 64 | 84.00% | 85.80% | 86.80% | 88.52% | 88.496% |
| 128 | 84.50% | 87.40% | 88.30% | 90.18% | 90.644% |
| 256 | 85.00% | 88.10% | 89.20% | 91.09% | 91.790% |
| 512 | 85.00% | 88.70% | 89.50% | 91.42% | 92.278% |
| 1024 | 85.00% | 89.10% | 89.85% | 91.66% | 92.497% |

---

# 7. Important benchmark correction

The first GPU benchmark version incorrectly assumed that RAFT IVF-PQ results were already sorted by rank.

The returned IDs were not ordered by score.

The corrected benchmark explicitly sorts `(score, id)` by **descending Inner Product score** before calculating Top1/Top5/Top10/Top50/Top500.

Example diagnostic result for GPU PQ96 f050 at `nprobe=32`:

| Processing mode | Top1 | Top10 | Top50 | Top500 |
|---|---:|---:|---:|---:|
| Raw returned order | 13.5% | 5.60% | 9.46% | 51.703% |
| Score descending | **38.5%** | **43.20%** | **46.39%** | 51.703% |
| Score ascending | 0.0% | 0.10% | 0.63% | 51.703% |

Therefore, earlier GPU Top1/Top10/Top50 values produced before this correction should not be used.

---

# 8. Additional measured facts

## GPU train-fraction effect

| PQ width | f025 Top10 | f050 Top10 | Difference |
|---|---:|---:|---:|
| PQ96 | 41.85% | **44.50%** | +2.65 pp |
| PQ192 | **69.30%** | 69.15% | -0.15 pp |
| PQ384 | 88.55% | **89.85%** | +1.30 pp |

## Same-size observations

### ~0.82 GiB

- CPU OPQ192: **71.75%**
- GPU PQ192 f025: **69.30%**
- GPU PQ192 f050: **69.15%**
- CPU PQ192: **66.45%**

At this size, CPU OPQ192 has the highest measured Top10 overlap.

### ~1.6 GiB

- GPU PQ384 f050: **89.85%**
- GPU PQ384 f025: **88.55%**
- CPU PQ384: **86.30%**
- CPU OPQ384: **85.20%**

At this size, GPU PQ384 f050 has the highest measured Top10 overlap.

## P104 vs CMP 90HX on the same GPU PQ96 f050 build

Build time:

- P104: **4.60 min**
- CMP 90HX: **6.49 min**

P104 was about **1.43× faster** in this build.

Search quality of the resulting indexes:

- P104-built PQ96 f050: **44.50% Top10**
- CMP90-built PQ96 f050: **41.60% Top10**

The search comparison used the same corrected benchmark methodology.

---

# 9. Summary

Measured size/quality points:

| Size | Best tested index in that size region | Top10 vs FP32 M32 |
|---:|---|---:|
| ~445 MiB | GPU PQ96 f050 P104 | **44.50%** |
| ~0.82 GiB | CPU OPQ192 | **71.75%** |
| ~1.60 GiB | GPU PQ384 f050 P104 | **89.85%** |
| 2.63 GiB | SQ4 HNSW M32 | **78.20%** |
| 3.40–4.17 GiB | SQ8 HNSW | **97.00–97.30%** |
| 13.42 GiB | FP32 HNSW M32 | **100% reference** |

The most important practical plateau observations are:

- HNSW SQ8 M16/M32 is already essentially saturated by `efSearch ≈ 4096–8192`.
- GPU PQ192 is largely saturated by `nprobe ≈ 256–512`.
- GPU PQ384 is largely saturated by `nprobe ≈ 256–512`.
- GPU PQ96 saturates much earlier, around `nprobe ≈ 128–256`.
- Increasing search effort beyond these points adds little quality relative to the added work.

