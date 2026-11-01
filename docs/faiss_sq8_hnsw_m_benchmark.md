# FAISS SQ8 HNSW: Effect of Graph Width `M`

This note isolates the effect of HNSW graph width `M` while keeping vector precision and the rest of the benchmark methodology fixed.

The existing `faiss_hnsw_scalar_quantization_benchmark.md` document remains the full FP32 / SQ8 / SQ4 benchmark. This file focuses specifically on `M = 32`, `16`, and `8`.

## Benchmark setup

Fixed for all SQ8 variants:

```text
database vectors : 4,304,468
embedding model  : intfloat/multilingual-e5-base
dimension        : 768
quantizer        : FAISS ScalarQuantizer.QT_8bit
metric           : inner product
normalization    : L2-normalized
efConstruction   : 200
training vectors : 199,980
add batch        : 10,000

queries          : 200 fixed query embeddings
returned results : Top-500
efSearch sweep   : 512, 1024, 2048, 4096, 8192, 16384, 32768
```

Only `M` was changed.

## Index size vs `M`

| SQ8 index | Size | Reduction vs M32 |
|---|---:|---:|
| HNSW32 | 4.170 GiB | — |
| HNSW16 | 3.657 GiB | 12.3% |
| HNSW8 | 3.402 GiB | 18.4% |

Reducing `M` does not reduce the whole file proportionally because most of the SQ8 index is vector-code storage, not graph storage.

For this dataset, the raw SQ8 codes alone are about `3.079 GiB`.

## Top-10 overlap vs FP32 HNSW reference

| efSearch | SQ8 M32 | SQ8 M16 | SQ8 M8 |
|---:|---:|---:|---:|
| 512 | 90.00% | 85.80% | 77.50% |
| 1024 | 94.75% | 92.30% | 87.00% |
| 2048 | 96.25% | 96.20% | 93.45% |
| 4096 | 97.20% | 97.15% | 95.90% |
| 8192 | 97.25% | 97.30% | 96.60% |
| 16384 | 97.25% | 97.25% | 96.90% |
| 32768 | 97.25% | 97.25% | 97.00% |

At `efSearch=2048`, changing `M=32 -> 16` changed Top-10 overlap only:

```text
96.25% -> 96.20%
```

Changing `M=16 -> 8` reduced it more noticeably:

```text
96.20% -> 93.45%
```

## Search latency

### SQ8 HNSW16

| efSearch | Median latency |
|---:|---:|
| 512 | 15.58 ms |
| 1024 | 27.21 ms |
| 2048 | 50.14 ms |
| 4096 | 104.45 ms |
| 8192 | 258.34 ms |
| 16384 | 796.39 ms |
| 32768 | 2817.71 ms |

### SQ8 HNSW8

| efSearch | Median latency |
|---:|---:|
| 512 | 10.35 ms |
| 1024 | 18.56 ms |
| 2048 | 35.78 ms |
| 4096 | 78.61 ms |
| 8192 | 207.38 ms |
| 16384 | 695.07 ms |
| 32768 | 2610.73 ms |

## Practical comparison at `efSearch=2048`

| Index | Size | Top-10 vs FP32 | Median latency |
|---|---:|---:|---:|
| SQ8 HNSW32 | 4.170 GiB | 96.25% | ~61 ms |
| SQ8 HNSW16 | 3.657 GiB | 96.20% | 50.14 ms |
| SQ8 HNSW8 | 3.402 GiB | 93.45% | 35.78 ms |

`M=16` is the strongest measured trade-off in this experiment:

- 12.3% smaller than M32;
- practically unchanged Top-10 agreement at `efSearch=2048`;
- lower measured latency in this run.

`M=8` is faster at the same `efSearch`, but loses more ranking agreement.

## Can higher `efSearch` compensate for M8?

Measured M8 results:

```text
M8 @ ef=2048 : 93.45% Top-10,  35.78 ms
M8 @ ef=4096 : 95.90% Top-10,  78.61 ms
M8 @ ef=8192 : 96.60% Top-10, 207.38 ms
```

Reference point:

```text
M16 @ ef=2048 : 96.20% Top-10, 50.14 ms
```

M8 can recover most of the ranking agreement by increasing `efSearch`, but once quality is brought back to roughly the M16 level, latency becomes higher than M16.

## Result

For this 4.3M-vector multilingual E5 corpus:

```text
best measured SQ8 operating point:
SQ8 HNSW16 + efSearch=2048
```

It retained essentially the same ranking agreement as HNSW32 while reducing index size and search latency.

HNSW8 is useful when lower latency is more important than maintaining approximately 96% Top-10 overlap with the FP32 HNSW reference.

---

## Context from the full FP32 / SQ8 / SQ4 benchmark

The separate scalar-quantization benchmark compares vector precision as well as HNSW search.

At `efSearch=2048`:

| Index | Size | Top-10 vs FP32 | Median latency |
|---|---:|---:|---:|
| FP32 HNSW reference | 13.42 GiB | reference | ~169 ms |
| SQ8 HNSW32 | 4.170 GiB | 96.25% | ~61 ms |
| SQ8 HNSW16 | 3.657 GiB | 96.20% | 50.14 ms |
| SQ8 HNSW8 | 3.402 GiB | 93.45% | 35.78 ms |
| SQ4 HNSW32 | 2.630 GiB | 78.00% | ~70 ms |

The earlier SQ4 experiment showed that reducing vector precision from SQ8 to SQ4 caused a much larger ranking loss than reducing HNSW graph width from M32 to M16.

See the full scalar-quantization benchmark for FP32 / SQ8 / SQ4 methodology, self-convergence results, similarity-error measurements, and reproducibility details.
