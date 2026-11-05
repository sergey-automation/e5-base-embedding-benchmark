# ANN index benchmarks — all measured CPU/GPU results (FP32 HNSW M32 reference)

**Updated: 2026-11-05.** This GitHub-ready report preserves all measured ANN index variants, including the completed RQ256×8 run. The sole quality reference in this report is **FP32 HNSW M32 (100%)**. Only experiments using the established FP32 HNSW M32 reference are included.

This document compares ANN index variants actually built and tested on the same `multilingual-e5-base` embedding corpus.

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

The FP32 HNSW M32 index is therefore treated as the **100% ranking reference** for this benchmark. **This is a relative reference, not a claim of mathematically exact recall.**

The displayed size is the **serialized index file size**, not a measured RAM peak. CPU search latency is median per query across the fixed 200-query set; GPU timing is not established as directly comparable. `pp` denotes percentage points.

### Key measured comparisons (same reference)

| Index | File size (GiB) | Top10 vs FP32 HNSW M32 | Search setting | CPU median (ms/query) |
|---|---:|---:|---|---:|
| FP32 HNSW M32 | 13.420 | **100.00% (reference)** | reference | not established |
| SQ8 HNSW M16 | 3.657 | 97.15% | ef=4096 | 104.45 |
| SQ8 HNSW M8 | 3.402 | 95.90% | ef=4096 | 78.61 |
| RaBitQ8 | 3.205 | 93.10% | nprobe=1024 | 193.44 |
| RaBitQ4 | 1.666 | 88.15% | nprobe=1024 | 129.60 |
| RaBitQ3 | 1.281 | 84.50% | nprobe=1024 | 121.52 |
| RaBitQ2 | 0.896 | 79.10% | nprobe=512 | 61.81 |
| RQ256×8 | 1.258 | 75.35% | nprobe=512 | **12997.78** |

All RQ256×8 measurements exceed the 200 ms budget; the complete GPU and all other CPU candidates appear in the detailed tables below.

---

# 0. All measured candidates under the search-time budget (no pruning)

The table below retains **every measured CPU configuration** (no Pareto pruning). For each CPU index, it selects the **highest measured Top10 overlap with median latency strictly below 200 ms**; on a quality tie, the lower-latency setting is chosen. Latency is median **per query**, not time for the full batch of 200 queries. The source for the new/previous CPU IVF rows is the preserved `summary_compact(4).csv` / `summary(4).csv`, `summary_compact(3).csv` and `summary_compact(2).csv` files. Older HNSW measurements come from the existing run reports and are approximate where marked with `~`.

## CPU — measured median search latency <200 ms

| Index | Platform | Index size (GiB) | Top10 vs FP32 M32 | Median search (ms/query) | Search setting |
|---|---|---:|---:|---:|---|
| SQ8 HNSW M32 | CPU | 4.170 | 97.20% | ~131 | `ef=4096` |
| SQ8 HNSW M16 | CPU | 3.657 | 97.15% | 104.45 | `ef=4096` |
| SQ8 HNSW M8 | CPU | 3.402 | 95.90% | 78.61 | `ef=4096` |
| RR768,IVF4096,RaBitQ8 | CPU | 3.2050 | 93.10% | 193.44 | `nprobe=1024` |
| RR768,IVF4096,RaBitQ4 | CPU | 1.6656 | 88.15% | 129.60 | `nprobe=1024` |
| RR768,IVF4096,RaBitQ3 | CPU | 1.2807 | 84.50% | 121.52 | `nprobe=1024` |
| RR768,IVF4096,RaBitQ2 | CPU | 0.8959 | 79.10% | 61.81 | `nprobe=512` |
| RR768,IVF4096,RaBitQ1 | CPU | 0.4629 | 63.50% | 91.84 | `nprobe=1024` |
| IVF4096,PQ384 | CPU | 1.584 | 85.50% | 164.16 | `nprobe=256` |
| OPQ384 + IVF4096,PQ384 | CPU | 1.586 | 84.65% | 159.22 | `nprobe=256` |
| SQ4 HNSW M32 | CPU | 2.630 | 78.00% | ~70 | `ef=2048` |
| OPQ256 + IVF4096,PQ256 | CPU | 1.073 | 76.80% | 147.19 | `nprobe=512` |
| IVF4096,PQ256 | CPU | 1.071 | 75.85% | 149.84 | `nprobe=512` |
| IVF4096,SQ4 | CPU | 1.583 | 75.15% | 140.62 | `nprobe=64` |
| OPQ192 + IVF4096,PQ192 | CPU | 0.816 | 71.70% | 108.87 | `nprobe=512` |
| IVF4096,PQ192 | CPU | 0.814 | 66.45% | 113.87 | `nprobe=512` |
| OPQ128 + IVF4096,PQ128 | CPU | 0.560 | 58.55% | 69.84 | `nprobe=512` |

RQ256×8 was also fully measured, but its **fastest tested** setting (`nprobe=32`) takes **812.89 ms/query**, so it cannot appear in the `<200 ms` table. Its full results are retained below.

No CPU index was excluded for being larger or having lower quality than another. FP32 HNSW M32 is the ranking reference; a comparable median search measurement below 200 ms is **not established** in these records, so it is not presented as an eligible under-200-ms result.

## GPU — candidates with preliminary sub-second observations; exact comparable latency unverified

The earlier GPU benchmark reported **rough raw search latency of ~0.9–5.6 ms**, i.e., well below 1 second, but those figures are **not validated as final comparable timings** (CUDA synchronization/workspace and benchmark measurement protocol were not fully controlled). Thus the table does **not** claim a confirmed `<1 s` guarantee or assign an exact latency to any individual GPU index. It retains all **seven** built/tested GPU variants and their **validated ranking quality** at `nprobe=1024`.

| Index | Platform | Index size (GiB) | Top10 vs FP32 M32 | nprobe | GPU search latency |
|---|---|---:|---:|---:|---|
| GPU PQ384 f050 — P104 | GPU | 1.606 | 89.85% | 1024 | Preliminary `<1 s`; exact time unverified |
| GPU PQ384 f025 — P104 | GPU | 1.606 | 88.55% | 1024 | Preliminary `<1 s`; exact time unverified |
| GPU PQ192 f025 — P104 | GPU | 0.825 | 69.30% | 1024 | Preliminary `<1 s`; exact time unverified |
| GPU PQ192 f050 — P104 | GPU | 0.825 | 69.15% | 1024 | Preliminary `<1 s`; exact time unverified |
| GPU PQ96 f050 — P104 | GPU | 0.435 | 44.50% | 1024 | Preliminary `<1 s`; exact time unverified |
| GPU PQ96 f025 — P104 | GPU | 0.435 | 41.85% | 1024 | Preliminary `<1 s`; exact time unverified |
| GPU PQ96 f050 — CMP 90HX | GPU | 0.435 | 41.60% | 1024 | Preliminary `<1 s`; exact time unverified |

The CPU and GPU latency figures should **not** be treated as a head-to-head speed benchmark. Quality is the overlap of ranked Top10 IDs with FP32 HNSW M32, not brute-force recall.

## Additional completed CPU index builds and search sweeps (2026-11-03 — 2026-11-04)

| New CPU index | Measured size (GiB) | Best Top10 | Best nprobe | Median at best nprobe |
|---|---:|---:|---:|---:|
| OPQ128 + IVF4096,PQ128 | 0.560 | 58.55% | 512 (also 1024) | 69.84 ms |
| OPQ256 + IVF4096,PQ256 | 1.073 | 76.80% | 512 (also 1024) | 147.19 ms |
| IVF4096,PQ256 (without OPQ) | 1.071 | 75.85% | 512 (also 1024) | 149.84 ms |
| IVF4096,SQ4 | 1.583 | 77.65% | 512 | 1048.82 ms |

The last row's best-quality point is **above the 200 ms search budget**; its eligible budget-limited setting is `nprobe=64`, Top10 `75.15%`, median `140.62 ms`. All four indexes have `4,304,468` vectors and were saved and verified by reopening. These are measured results, not predictions.

| nprobe | OPQ128 Top10 | PQ256 Top10 | OPQ256 Top10 | IVF-SQ4 Top10 |
|---:|---:|---:|---:|---:|
| 32 | 55.60% | 71.70% | 72.10% | 73.10% |
| 64 | 56.70% | 73.15% | 74.15% | 75.15% |
| 128 | 57.70% | 74.50% | 75.45% | 76.40% |
| 256 | 58.25% | 75.25% | 76.20% | 77.05% |
| 512 | 58.55% | 75.85% | 76.80% | 77.65% |
| 1024 | 58.55% | 75.85% | 76.80% | 77.50% |

For these four, the increase from `nprobe=512` to `1024` produces no Top10 gain; IVF-SQ4 decreases slightly from 77.65% to 77.50%. This observation is limited to the 200-query benchmark sample.


## Direct experiment: IVF4096,PQ256 vs OPQ256 + IVF4096,PQ256 (2026-11-04)

Measured source: `summary(4).csv` and `summary_compact(4).csv`; same 200 queries, `K=500`, FP32 HNSW M32 ranking reference. Both indexes contain **4,304,468** vectors. Sizes: plain PQ256 **1.071 GiB**, OPQ256 **1.073 GiB** (rounded index file sizes).

| nprobe | PQ256 Top10 | OPQ256 Top10 | OPQ gain (percentage points) | PQ256 median (ms) | OPQ256 median (ms) |
|---:|---:|---:|---:|---:|---:|
| 32 | 71.70% | 72.10% | +0.40 | 12.78 | 12.72 |
| 64 | 73.15% | 74.15% | +1.00 | 22.46 | 22.59 |
| 128 | 74.50% | 75.45% | +0.95 | 40.53 | 43.58 |
| 256 | 75.25% | 76.20% | +0.95 | 76.54 | 78.27 |
| **512** | **75.85%** | **76.80%** | **+0.95** | **149.84** | **155.63** |
| 1024 | 75.85% | 76.80% | +0.95 | 301.16 | 298.99 |

At `nprobe=512`, OPQ improves Top10 from **75.85% to 76.80%**, i.e. **+0.95 percentage points** (relative increase **1.25%**), while the reported index size grows by **0.002 GiB ≈ 2.05 MiB**, or **~0.187%**. The relative Top10 gain is about **6.7 times** the relative memory increase. With 200 queries × 10 results, this equals **19 additional matched Top10 IDs**. The Top10 non-overlap decreases from **24.15% to 23.20%**, a **3.93% relative reduction**. Size figures are rounded to three decimals; ratios based on those figures are approximate.

### All ranking overlap metrics at `nprobe=512` (same measurement run)

| Metric | Plain PQ256 | OPQ256 | Difference (OPQ − PQ) |
|---|---:|---:|---:|
| Top1 | **73.00%** | 69.50% | −3.50 pp |
| Top5 | 73.50% | **74.60%** | +1.10 pp |
| Top10 | 75.85% | **76.80%** | +0.95 pp |
| Top50 | 77.97% | **78.13%** | +0.16 pp |
| Top500 | **78.451%** | 78.427% | −0.024 pp |
| Median latency | **149.84 ms** | 155.63 ms | +5.79 ms |

**Scope of the finding:** OPQ256 has a material incremental **Top10 gain per additional byte** on this corpus, but it is **not uniformly better** across ranking depths: plain PQ256 has higher Top1 and marginally higher Top500. The separate previous OPQ256 run measured **147.19 ms** at `nprobe=512`; the direct head-to-head rerun measured **155.63 ms**. Both figures are retained, not merged or silently substituted. Ranking overlap is against FP32 HNSW M32, not exact brute-force recall.

---

## RaBitQ 1–4 bits: completed CPU builds and search measurements (2026-11-04)

All four configurations use `RR768,IVF4096,RaBitQ{bits}`, Inner Product, 768 dimensions, 4,304,468 embeddings, 199,980 training vectors, four CPU threads, and the same 200 queries (`K=500`). Ranking quality is **overlap relative to FP32 HNSW M32**, not exact brute-force recall. File sizes and build times come from original `build_summary(4–7).json`; search measurements from `summary_compact(5).csv` (RaBitQ4 first run), `(6)` (RaBitQ4 repeat), `(7)` (RaBitQ2), `(8)` (RaBitQ3), `(9)` (RaBitQ1). All reported build times below are sums of instrumented phases, not necessarily end-to-end wall time.

### Size, quality, and latency at equal `nprobe=512`

| Index | Bits/coordinate | Size GiB (measured) | Top1 | Top10 | Top50 | Top500 | Median ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| RaBitQ1 | 1 | 0.4629 | 59.00% | 63.40% | 64.74% | 65.906% | 47.09 |
| RaBitQ2 | 2 | 0.8959 | 77.50% | 79.10% | 80.90% | 79.395% | 61.81 |
| RaBitQ3 | 3 | 1.2807 | 83.50% | 84.45% | 87.05% | 84.890% | 68.12 |
| RaBitQ4 | 4 | 1.6656 | 85.50% | 88.05% | 90.38% | 87.314% | 69.95 |

### Full measured `nprobe` sweep (Top10 overlap and median latency)

| `nprobe` | RaBitQ1 Top10 / ms | RaBitQ2 Top10 / ms | RaBitQ3 Top10 / ms | RaBitQ4 Top10 / ms |
|---:|---:|---:|---:|---:|
| 32 | 59.75% / 4.86 | 73.75% / 14.21 | 78.85% / 15.01 | 82.10% / 13.18 |
| 64 | 61.80% / 7.77 | 76.05% / 18.21 | 81.45% / 17.15 | 84.75% / 17.63 |
| 128 | 62.65% / 13.46 | 77.60% / 23.48 | 83.00% / 24.52 | 86.60% / 25.80 |
| 256 | 63.10% / 24.62 | 78.60% / 36.59 | 83.80% / 40.82 | 87.45% / 40.44 |
| 512 | 63.40% / 47.09 | 79.10% / 61.81 | 84.45% / 68.12 | 88.05% / 69.95 |
| 1024 | 63.50% / 91.84 | 79.05% / 112.91 | 84.50% / 121.52 | 88.15% / 129.60 |

### Measured training and build time

| Index | Training s | ADD s | ADD vectors/s | Final write s | Final verify s | Sum of phases s | Sum of phases min |
|---|---:|---:|---:|---:|---:|---:|---:|
| RaBitQ1 | 62.81 | 208.47 | 20648.4 | 1.71 | 0.44 | **273.42** | **4.56** |
| RaBitQ2 | 89.11 | 325.16 | 13237.9 | 3.44 | 0.79 | **418.51** | **6.98** |
| RaBitQ3 | 61.95 | 404.38 | 10644.7 | 6.58 | 0.99 | **473.89** | **7.90** |
| RaBitQ4 | 62.17 | 466.89 | 9219.4 | 6.58 | 1.75 | **537.39** | **8.96** |

### Marginal Top10 improvement for additional index size (`nprobe=512`)

| Transition | Additional GiB | Top10 gain (percentage points) | Top10 gain per additional GiB |
|---|---:|---:|---:|
| RaBitQ1 → RaBitQ2 | 0.4330 | +15.70 | 36.26 pp/GiB |
| RaBitQ2 → RaBitQ3 | 0.3848 | +5.35 | 13.90 pp/GiB |
| RaBitQ3 → RaBitQ4 | 0.3848 | +3.60 | 9.35 pp/GiB |

RaBitQ4 was benchmarked twice. Top10 results were identical at all `nprobe` points; observed latency changed slightly between the runs. The primary tables report the first run and retain the second run as a reproducibility check rather than averaging unlike runs. At `nprobe=512`, the repeated median was **73.32 ms** (first run **69.95 ms**); at `nprobe=1024`, repeated median **129.21 ms** (first run **129.60 ms**).

---

# 1. All tested indexes, grouped by size

Within each size group, the highest measured `Top10` quality is listed first.

## ~0.44 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| IVF4096, PQ96, train fraction 0.50 | P104 GPU | **445.0 MiB** | **44.50%** |
| IVF4096, PQ96, train fraction 0.25 | P104 GPU | **445.1 MiB** | **41.85%** |
| IVF4096, PQ96, train fraction 0.50 | CMP 90HX GPU | **445.1 MiB** | **41.60%** |

## ~0.46 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| RR768,IVF4096,RaBitQ1 | CPU | **0.4629 GiB** | **63.50%** |

## ~0.56 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| OPQ128 + IVF4096,PQ128 | CPU | **0.560 GiB** | **58.55%** |

## ~0.81–0.83 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| OPQ192 + IVF4096, PQ192 | CPU | **0.816 GiB** | **71.75%** |
| IVF4096, PQ192, train fraction 0.25 | P104 GPU | **0.825 GiB** | **69.30%** |
| IVF4096, PQ192, train fraction 0.50 | P104 GPU | **0.825 GiB** | **69.15%** |
| IVF4096, PQ192 | CPU | **0.814 GiB** | **66.45%** |

## ~0.90 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| RR768,IVF4096,RaBitQ2 | CPU | **0.8959 GiB** | **79.10%** |

## ~1.07 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| OPQ256 + IVF4096,PQ256 | CPU | **1.073 GiB** | **76.80%** |
| IVF4096,PQ256 (without OPQ) | CPU | **1.071 GiB** | **75.85%** |

## ~1.26–1.28 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| RR768,IVF4096,RaBitQ3 | CPU | **1.2807 GiB** | **84.50%** |
| IVF4096,RQ256x8_Nnone | CPU | **1.258 GiB** | **75.35%** |

## ~1.58–1.61 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| IVF4096, PQ384, train fraction 0.50 | P104 GPU | **1.606 GiB** | **89.85%** |
| IVF4096, PQ384, train fraction 0.25 | P104 GPU | **1.606 GiB** | **88.55%** |
| IVF4096, PQ384 | CPU | **1.584 GiB** | **86.30%** |
| OPQ384 + IVF4096, PQ384 | CPU | **1.586 GiB** | **85.20%** |
| IVF4096,SQ4 | CPU | **1.583 GiB** | **77.65%** |

## ~1.67 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| RR768,IVF4096,RaBitQ4 | CPU | **1.6656 GiB** | **88.15%** |

## ~2.6 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| SQ4 HNSW M32 | CPU | **2.630 GiB** | **78.20%** |

## ~3.20 GiB class

| Index | Platform | Size | Best Top10 vs FP32 M32 |
|---|---|---:|---:|
| RR768,IVF4096,RaBitQ8 | CPU | **3.2050 GiB** | **93.10%** |

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
| CPU RaBitQ8 | CPU | 3.2050 GiB | 93.10% |
| GPU PQ384 f050 | P104 | 1.606 GiB | 89.85% |
| GPU PQ384 f025 | P104 | 1.606 GiB | 88.55% |
| CPU RaBitQ4 | CPU | 1.6656 GiB | 88.15% |
| CPU PQ384 | CPU | 1.584 GiB | 86.30% |
| CPU OPQ384 | CPU | 1.586 GiB | 85.20% |
| CPU RaBitQ3 | CPU | 1.2807 GiB | 84.50% |
| CPU RaBitQ2 | CPU | 0.8959 GiB | 79.10% |
| SQ4 HNSW M32 | CPU | 2.630 GiB | 78.20% |
| CPU IVF-SQ4 | CPU | 1.583 GiB | 77.65% |
| CPU OPQ256 | CPU | 1.073 GiB | 76.80% |
| CPU PQ256 (without OPQ) | CPU | 1.071 GiB | 75.85% |
| CPU RQ256×8 | CPU | 1.258 GiB | 75.35% |
| CPU OPQ192 | CPU | 0.816 GiB | 71.75% |
| GPU PQ192 f025 | P104 | 0.825 GiB | 69.30% |
| GPU PQ192 f050 | P104 | 0.825 GiB | 69.15% |
| CPU PQ192 | CPU | 0.814 GiB | 66.45% |
| CPU RaBitQ1 | CPU | 0.4629 GiB | 63.50% |
| CPU OPQ128 | CPU | 0.560 GiB | 58.55% |
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
| CPU PQ256 | nprobe=32 | 71.70% | — |
|  | 64 | 73.15% | +1.45 pp |
|  | 128 | 74.50% | +1.35 pp |
|  | 256 | 75.25% | +0.75 pp |
|  | 512 | **75.85%** | +0.60 pp |
|  | 1024 | 75.85% | +0.00 pp |
| CPU OPQ256 | nprobe=32 | 72.10% | — |
|  | 64 | 74.15% | +2.05 pp |
|  | 128 | 75.45% | +1.30 pp |
|  | 256 | 76.20% | +0.75 pp |
|  | 512 | **76.80%** | +0.60 pp |
|  | 1024 | 76.80% | +0.00 pp |
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

| CPU RaBitQ1 | nprobe=32 | 59.75% | — |
| CPU RaBitQ1 | nprobe=64 | 61.80% | +2.05 pp |
| CPU RaBitQ1 | nprobe=128 | 62.65% | +0.85 pp |
| CPU RaBitQ1 | nprobe=256 | 63.10% | +0.45 pp |
| CPU RaBitQ1 | nprobe=512 | 63.40% | +0.30 pp |
| CPU RaBitQ1 | nprobe=1024 | 63.50% | +0.10 pp |

| CPU RaBitQ2 | nprobe=32 | 73.75% | — |
| CPU RaBitQ2 | nprobe=64 | 76.05% | +2.30 pp |
| CPU RaBitQ2 | nprobe=128 | 77.60% | +1.55 pp |
| CPU RaBitQ2 | nprobe=256 | 78.60% | +1.00 pp |
| CPU RaBitQ2 | nprobe=512 | 79.10% | +0.50 pp |
| CPU RaBitQ2 | nprobe=1024 | 79.05% | -0.05 pp |

| CPU RaBitQ3 | nprobe=32 | 78.85% | — |
| CPU RaBitQ3 | nprobe=64 | 81.45% | +2.60 pp |
| CPU RaBitQ3 | nprobe=128 | 83.00% | +1.55 pp |
| CPU RaBitQ3 | nprobe=256 | 83.80% | +0.80 pp |
| CPU RaBitQ3 | nprobe=512 | 84.45% | +0.65 pp |
| CPU RaBitQ3 | nprobe=1024 | 84.50% | +0.05 pp |

| CPU RaBitQ4 | nprobe=32 | 82.10% | — |
| CPU RaBitQ4 | nprobe=64 | 84.75% | +2.65 pp |
| CPU RaBitQ4 | nprobe=128 | 86.60% | +1.85 pp |
| CPU RaBitQ4 | nprobe=256 | 87.45% | +0.85 pp |
| CPU RaBitQ4 | nprobe=512 | 88.05% | +0.60 pp |
| CPU RaBitQ4 | nprobe=1024 | 88.15% | +0.10 pp |

### Practical plateau points

Using a practical definition of plateau as the point after which additional search effort gives only about `<=0.3–0.5 percentage points` per step:

| Index | Practical plateau begins around | Top10 near plateau |
|---|---:|---:|
| SQ8 HNSW M32 | ef≈4096–8192 | 97.20–97.25% |
| SQ8 HNSW M16 | ef≈4096–8192 | 97.15–97.30% |
| SQ8 HNSW M8 | ef≈8192–16384 | 96.60–96.90% |
| SQ4 HNSW M32 | ef≈4096 | 78.15% |
| CPU RaBitQ1 | nprobe≈512 | 63.40% |
| CPU RaBitQ2 | nprobe≈512 | 79.10% |
| CPU RaBitQ3 | nprobe≈512 | 84.45% |
| CPU RaBitQ4 | nprobe≈512 | 88.05% |
| CPU PQ192 | nprobe≈256–512 | 66.10–66.45% |
| CPU OPQ192 | nprobe≈256–512 | 71.25–71.55% |
| CPU PQ256 | nprobe≈512 | 75.85% |
| CPU OPQ256 | nprobe≈512 | 76.80% |
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
| CPU PQ256 | nprobe=512 | 75.85% | 149.84 ms |
| CPU OPQ256 | nprobe=512 (direct rerun) | 76.80% | 155.63 ms |
| CPU PQ384 | nprobe=1024 | 86.30% | 624.8 ms |
| CPU RQ256×8 | nprobe=32 | 70.60% | 812.89 ms |
| CPU RQ256×8 | nprobe=512 | 75.35% | 12997.78 ms |
| CPU RaBitQ1 | nprobe=512 | 63.40% | 47.09 ms |
| CPU RaBitQ1 | nprobe=1024 | 63.50% | 91.84 ms |
| CPU RaBitQ2 | nprobe=512 | 79.10% | 61.81 ms |
| CPU RaBitQ2 | nprobe=1024 | 79.05% | 112.91 ms |
| CPU RaBitQ3 | nprobe=512 | 84.45% | 68.12 ms |
| CPU RaBitQ3 | nprobe=1024 | 84.50% | 121.52 ms |
| CPU RaBitQ4 | nprobe=512 | 88.05% | 69.95 ms |
| CPU RaBitQ4 | nprobe=1024 | 88.15% | 129.60 ms |

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
| ~0.46 GiB | CPU RaBitQ1 | **63.50%** |
| ~0.56 GiB | CPU OPQ128 | **58.55%** |
| ~0.82 GiB | CPU OPQ192 | **71.75%** |
| ~0.90 GiB | CPU RaBitQ2 | **79.10%** |
| ~1.07 GiB | CPU OPQ256 | **76.80%** (plain PQ256: 75.85%) |
| ~1.26 GiB | CPU RQ256×8 | **75.35%** (12,998 ms at nprobe=512) |
| ~1.28 GiB | CPU RaBitQ3 | **84.50%** |
| ~1.58 GiB | CPU IVF-SQ4 (additional completed candidate) | **77.65%** |
| ~1.60 GiB | GPU PQ384 f050 P104 | **89.85%** |
| ~1.67 GiB | CPU RaBitQ4 | **88.15%** |
| 2.63 GiB | SQ4 HNSW M32 | **78.20%** |
| ~3.205 GiB | CPU RaBitQ8 | **93.10%** |
| 3.40–4.17 GiB | SQ8 HNSW | **97.00–97.30%** |
| 13.42 GiB | FP32 HNSW M32 | **100% reference** |

The most important practical plateau observations are:

- HNSW SQ8 M16/M32 is already essentially saturated by `efSearch ≈ 4096–8192`.
- GPU PQ192 is largely saturated by `nprobe ≈ 256–512`.
- GPU PQ384 is largely saturated by `nprobe ≈ 256–512`.
- GPU PQ96 saturates much earlier, around `nprobe ≈ 128–256`.
- Increasing search effort beyond these points adds little quality relative to the added work.


---

## Update provenance (2026-11-04)

Added the completed **CPU IVF4096,PQ256 without OPQ** build and direct PQ256/OPQ256 CPU search comparison, preserving existing reported measurements. New quality and per-query search latency numbers are transcribed from `summary(4).csv` / `summary_compact(4).csv`. The index build log confirms 4,304,468 vectors and final size 1.071 GiB. No existing historical measurements were recalculated.


## Update provenance (2026-11-04, RaBitQ1–4)

Appended four completed CPU RaBitQ builds and their measured search sweeps without discarding previous variants. Input files: `build_summary(4).json` (RaBitQ4), `(5)` (RaBitQ2), `(6)` (RaBitQ3), `(7)` (RaBitQ1); `summary_compact(5).csv`, `(6)` RaBitQ4 repeat, `(7)` RaBitQ2, `(8)` RaBitQ3, `(9)` RaBitQ1. All sizes are from recorded final index sizes, not estimates.


---

# 10. RaBitQ8 measured CPU results (2026-11-04)

Source files: `summary(10).csv`, `summary_compact(10).csv`, `build_summary(8).json`. These were measured using the same 200 fixed queries and 500 returned candidates; quality denotes ranking overlap against FP32 HNSW M32, not exact brute-force recall.

- Factory: `RR768,IVF4096,RaBitQ8`, inner product, 768 dimensions, 4,304,468 indexed vectors.
- Training set: 199,980 vectors; CPU OpenMP threads: 4.
- Measured index size: **3,441,331,752 bytes = 3.204990 GiB**.
- Times: TRAIN **91.51 s**, ADD **1415.02 s**, final write **15.27 s**, verify **17.57 s**.
- Sum of measured phases (not the complete process wall clock): **1539.37 s = 25.66 min**.
- ADD rate: **3042 vectors/s**.

| nprobe | Top1 | Top5 | Top10 | Top50 | Top500 | Median ms/query |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 83.00% | 85.10% | 86.10% | 85.81% | 81.638% | 15.48 |
| 64 | 87.50% | 88.20% | 89.05% | 89.59% | 85.907% | 21.71 |
| 128 | 90.50% | 90.10% | 91.05% | 91.97% | 88.330% | 33.65 |
| 256 | 91.50% | 91.70% | 92.20% | 93.39% | 88.876% | 56.45 |
| 512 | 92.50% | 92.50% | 92.95% | 94.00% | 88.811% | 102.93 |
| 1024 | 93.00% | 92.70% | 93.10% | 94.01% | 88.770% | 193.44 |

Maximum measured Top10: **93.10%** at `nprobe=1024`, with median **193.44 ms/query**, under the 200 ms selection budget. Top500 is not monotonic over the tested nprobe sweep; no causal explanation is claimed.

## RaBitQ4 versus RaBitQ8 at the same nprobe=512

| Measure | RaBitQ4 | RaBitQ8 | Difference |
|---|---:|---:|---:|
| Index size | 1.6656 GiB | 3.2050 GiB | +1.5394 GiB |
| Top10 | 88.05% | 92.95% | +4.90 pp |
| Median search | 69.95 ms | 102.93 ms | +32.98 ms |

---


# 11. Completed RQ256×8 CPU benchmark (2026-11-05)

**Status: built, saved, reopened and benchmarked.** This section uses the same FP32 HNSW M32 results as its 100% ranking reference. No alternative reference is used in this report.

- Index: `F:\RAG_checkpoints\ivf4096_rq256x8_cpu\ivf4096_rq256x8.index`
- Reported factory: `IVF4096,RQ256x8_Nnone`; benchmark label: `RR768_IVF4096_RQ256x8`. The naming discrepancy is preserved rather than treating the two strings as proof of identical factory definitions.
- `ntotal=4,304,468`, `d=768`, trained and verified after reopening.
- Final serialized index: **1.258 GiB**.
- Search: 4 FAISS CPU threads; same 200 queries; `K=500`.
- Build duration: **not available in supplied final-build excerpt**. Checkpoint writing after part 66: 4.5 s; verification reopening: 1.1 s. These are **not** build-total times.
- Search results from `summary_compact(20261105-072734).csv` and `summary(20261105-072735).csv`.

| nprobe | Top1 (%) | Top5 (%) | Top10 (%) | Top50 (%) | Top500 (%) | Median ms/query | P95 ms/query | Matched-score MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 64.50 | 70.10 | 70.60 | 72.34 | 72.201 | 812.89 | 907.24 | 0.00219941 |
| 64 | 66.50 | 71.80 | 72.50 | 74.63 | 75.226 | 1615.91 | 1787.36 | 0.00217910 |
| 128 | 68.00 | 73.50 | 73.85 | 76.10 | 76.797 | 3274.32 | 3395.80 | 0.00216400 |
| 256 | 68.50 | 74.20 | 74.75 | 77.09 | 77.610 | 6487.86 | 6687.18 | 0.00215812 |
| 512 | 69.50 | 74.90 | 75.35 | 77.44 | 77.976 | 12997.78 | 13566.37 | 0.00215809 |
| 1024 | 70.00 | 75.00 | 75.30 | 77.43 | 78.061 | 22436.29 | 26311.76 | 0.00215810 |

Best measured Top10: **75.35%** (`nprobe=512`, **12.998 seconds/query**). Fastest tested setting: `nprobe=32`, **70.60%** Top10 at **812.89 ms/query**. No tested RQ setting meets the `<200 ms` CPU search-budget condition. The small decline from 75.35% to 75.30% at 1024 is a measured fluctuation on the fixed queries, not a general quality guarantee.

**Same-size practical comparison** (different `nprobe`, both against the *same* FP32 HNSW M32 reference):

| Index | Size GiB | nprobe | Top10 (%) | Median ms/query |
|---|---:|---:|---:|---:|
| RaBitQ3 | 1.2807 | 512 | **84.45** | **68.12** |
| RQ256×8 | 1.258 | 512 | 75.35 | 12997.78 |

RaBitQ3 has **+9.10 percentage points Top10** at these settings, with roughly **191× lower median latency** (`12997.78 / 68.12 ≈ 190.8`). This is an observed result for the tested configuration, not proof that all RQ implementations are inferior.

**Interpretation versus evidence:** the 256-stage RQ decoding cost is a plausible explanation for high latency, **not a measured bottleneck breakdown**. No profiler data is available to assign the delay to a specific kernel or implementation detail.

---

# 12. Other ANN method families (research backlog and completed RQ)

This section is **primarily a backlog of untested families**; RQ is retained as a historical candidate and explicitly marked completed. Unmeasured methods below must not be merged into the measured index ranking. For all candidates, the requirements are a **complete searchable index, approximately 0.4–4 GiB total storage**, no mandatory external FP32 dataset on SSD during search, and median query latency under 200 ms. **None of these conditions has yet been verified for the implementations below**.

| Method/family | Distinct search principle | Storage and RAM condition | Status |
|---|---|---|---|
| NSG + compressed codes | Navigable sparse neighbor graph rather than HNSW | Feasibility of a compact self-contained compressed implementation must be checked; graph links have additional cost | Not tested |
| In-memory Vamana + compressed codes | Vamana graph traversal | Needs a verified implementation retaining graph and compact codes without compulsory FP32 vectors for search | Not tested |
| ScaNN | Learned partitioning / anisotropic quantization and candidate scoring | Quantized-only searchable deployment and actual memory/disk footprint need validation | Not tested |
| IMI + PQ | Multi-index inverted lists, different coarse candidate selection from IVF4096 | Compressed PQ codes can be stored without original FP32 for approximate search; list/storage overhead must be measured | Not tested |
| Residual Quantization (RQ) | Successive residual codebooks | **RQ256×8 tested**; 1.258 GiB, maximum Top10 75.35%, minimum median 812.89 ms | **Tested (RQ256×8)** |
| Additive Quantization / LSQ | Sum of codebook vectors; trained code assignments | Code length configurable; training/search costs and actual footprint unknown | Not tested |
| IVF-PQR | PQ candidate search followed by refinement from an extra code | Compressed base and refinement codes; final index memory must be measured | Not tested |
| LSH | Locality-sensitive hashing instead of graph/IVF traversal | Multiple hash tables can dominate the index; no verified sub-4-GiB configuration here | Not tested |
| Flat compressed scan | Full scan of PQ/SQ codes | Can fit in 0.4–4 GiB, but sub-200-ms performance at N=4.3M is not verified | Not tested |

Conventional SSD-backed DiskANN (Vamana graph and original FP32 vectors on SSD) is **not** counted as satisfying the 0.4–4 GiB **complete index** condition merely because its RAM PQ-cache is small. For this corpus, FP32 embeddings alone require about **12.32 GiB** on disk (4,304,468 × 768 × 4 bytes), before graph/SSD structures.

Future comparison should measure full disk index bytes, resident RAM, Top1/5/10/50/500 overlap, per-query median/p95 and build times under consistent data/query settings. No unverified Top10 numbers are predicted here.

---

## Update provenance (2026-11-04, RaBitQ8 and untested alternatives)

Appended measured RaBitQ8 sweep and build statistics from `summary(10).csv` and `build_summary(8).json`, inserted RaBitQ8 into existing size/quality rankings, and appended a separately labeled inventory of *unmeasured* alternative algorithms. No existing measurements were deleted or overwritten.

---

## Update provenance (2026-11-05, RQ256×8)

This update retains the pre-existing HNSW, SQ, PQ, OPQ, RaBitQ and GPU RAFT observations and adds the completed RQ256×8 CPU run, including all six `nprobe` measurements, verification status, and the 200 ms budget exclusion. All ranking comparisons retain the original FP32 HNSW M32 reference. No missing build times or GPU search latency measurements were fabricated. Earlier approximate or reconstructed build times remain explicitly labeled.
