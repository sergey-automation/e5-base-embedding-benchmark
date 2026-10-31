# FAISS HNSW Scalar Quantization Benchmark: FP32 vs SQ8 vs SQ4

This document describes a reproducible benchmark of HNSW-based semantic-search indexes built from the same multilingual E5 embeddings.

It is intended as a supplementary benchmark note for the repository, not as the main project README.

The benchmark compares:

- FP32 HNSW reference search;
- FAISS HNSW with 8-bit scalar quantization (`SQ8`);
- FAISS HNSW with 4-bit scalar quantization (`SQ4`).

The main questions are:

1. How much index size is saved by scalar quantization?
2. How much ranking agreement is lost relative to FP32?
3. Can higher `efSearch` recover the loss?
4. What latency/quality trade-off is obtained at each search width?
5. Is the loss caused mainly by approximate graph traversal or by quantization itself?

---

## 1. Benchmark summary

Dataset used in the original run:

```text
database vectors : 4,304,468
embedding model  : intfloat/multilingual-e5-base
dimension        : 768
embedding dtype  : float32 before index quantization
normalization    : L2-normalized
search metric    : inner product
queries          : 200 fixed query embeddings
returned results : Top-500
```

The same 200 query embeddings were reused for all index variants.

No query embeddings were regenerated between FP32, SQ8, and SQ4 comparisons.

### Main result at `efSearch = 2048`

| Index | Full index size | Top-1 agreement vs FP32 | Top-10 overlap vs FP32 | Top-500 overlap vs FP32 | Median latency |
|---|---:|---:|---:|---:|---:|
| SQ8 HNSW32 | 4.170 GiB | 94.0% | 96.25% | 97.93% | ~61 ms |
| SQ4 HNSW32 | 2.630 GiB | 79.0% | 78.00% | 81.40% | ~70 ms |

The SQ4 index was 36.9% smaller than SQ8, but the ranking agreement dropped substantially.

Increasing SQ4 `efSearch` from `2048` to `32768` changed Top-10 overlap with FP32 only from `78.00%` to `78.15%`, while median latency increased from about `70 ms` to about `3011 ms`.

This indicates that, for this dataset and embedding distribution, the dominant SQ4 quality loss came from 4-bit scalar quantization rather than insufficient HNSW graph traversal.

---

## 2. Repository placement

Recommended location:

```text
docs/faiss_hnsw_scalar_quantization_benchmark.md
```

This benchmark belongs naturally next to the embedding benchmark because it uses the same embedding family and evaluates the next stage of the retrieval pipeline:

```text
text/chunks
    ↓
multilingual-e5-base
    ↓
normalized 768-D embeddings
    ↓
HNSW index construction
    ↓
FP32 / SQ8 / SQ4 comparison
```

---

## 3. Required input data

To reproduce the benchmark on another corpus, prepare a collection of normalized FP32 embeddings.

The original dataset had:

```text
N = 4,304,468
D = 768
```

The source vectors were stored in multiple NumPy files:

```text
embeddings/
    emb_part_000000.npy
    emb_part_000001.npy
    ...
```

A different dataset may use any number of parts.

Each array must have shape:

```python
(n_vectors_in_part, embedding_dimension)
```

and should use:

```python
dtype = np.float32
```

All vectors must use the same embedding model and preprocessing.

### Required normalization

For cosine-style semantic search using inner product:

```python
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
```

The query vectors must be normalized in exactly the same way.

Do not compare normalized and non-normalized runs.

---

## 4. Embedding model

Original benchmark:

```text
intfloat/multilingual-e5-base
dimension = 768
```

For a new corpus, another embedding model may be used, but all compared index variants must be built from exactly the same source embeddings.

If a different model or embedding dimension is used, record it explicitly.

Recommended metadata:

```text
model name
model revision/commit if pinned
embedding dimension
dtype
normalization
pooling method
query/document prefix convention
```

For strict reproducibility, save the final query embeddings used by the benchmark instead of regenerating them for each run.

---

## 5. Software and hardware environment

Original benchmark environment:

```text
OS          : Windows 10 Pro 64-bit, build 19044
CPU         : Intel Core i5-4570, 4 cores / 4 threads, 3.20 GHz
CPU cache   : 1 MiB L2, 6 MiB L3
RAM         : 16 GiB DDR3-1600, 2 × 8 GiB
Python      : 3.13.1
FAISS       : faiss-cpu 1.15.1
NumPy       : 2.2.6
hnswlib     : 0.8.0
FAISS/OpenMP threads : 4
```

The benchmark used storage fast enough to keep index construction and file loading practical, but storage model names are intentionally omitted because they do not affect the in-memory search-quality comparison.

For latency reproduction, the important factors are:

```text
CPU architecture and core count
RAM capacity and memory configuration
FAISS/OpenMP thread count
software versions
whether the index is fully resident in RAM
```

Absolute latency should not be compared across machines without recording those parameters.

For another system, record at least:

```text
operating system
CPU model / cores / threads
RAM capacity and memory type
Python version
FAISS version
NumPy version
hnswlib version
FAISS/OpenMP thread count
```

---

## 6. Index variants

### 6.1 FP32 reference

The original benchmark used an existing FP32 HNSW index as the reference retrieval system.

The stored vectors were FP32.

Important limitation:

> This is an FP32 HNSW reference, not an exact brute-force `IndexFlatIP` ground truth.

Therefore the percentages below are **ranking agreement / overlap with the FP32 HNSW reference**, not mathematical exact-nearest-neighbor recall.

For future experiments, an additional exact `IndexFlatIP` baseline is recommended when memory and runtime permit.

The FP32 reference index was built with `hnswlib`:

```text
library        = hnswlib 0.8.0
space          = cosine
M              = 32
efConstruction = 200
stored vectors = FP32
```

The serialized FP32 index size was:

```text
14,412,499,252 bytes
≈ 13.42 GiB
```

The SQ8 and SQ4 indexes used the same `M=32` and `efConstruction=200`, which keeps the graph-construction settings aligned across the compared variants.

---

### 6.2 SQ8 HNSW32

FAISS index type:

```python
faiss.IndexHNSWSQ(
    dim,
    faiss.ScalarQuantizer.QT_8bit,
    32,
    faiss.METRIC_INNER_PRODUCT,
)
```

Construction parameters:

```text
dim            = 768
M              = 32
efConstruction = 200
metric         = inner product
vectors        = normalized
training sample = up to 200,000 vectors
add batch       = 10,000 vectors
```

Original result:

```text
vectors          = 4,304,468
training vectors = 199,980
build time       = 16,971.27 s
                 ≈ 282.9 min
                 ≈ 4.71 h
build rate       ≈ 253.6 vectors/s
index size       = 4,477,255,414 bytes
                 ≈ 4.170 GiB
```

---

### 6.3 SQ4 HNSW32

FAISS index type:

```python
faiss.IndexHNSWSQ(
    dim,
    faiss.ScalarQuantizer.QT_4bit,
    32,
    faiss.METRIC_INNER_PRODUCT,
)
```

Construction parameters:

```text
dim            = 768
M              = 32
efConstruction = 200
metric         = inner product
vectors        = normalized
training sample = up to 200,000 vectors
add batch       = 10,000 vectors
```

Final original result:

```text
vectors              = 4,304,468
checkpoint interval  = every 2 embedding parts
accumulated add time = 16,646.26 s
index size           = 2,824,339,702 bytes
                     ≈ 2.630 GiB
```

The final index was reopened after writing and verified with:

```text
ntotal = 4,304,468
```

---

## 7. Safe index construction

For multi-hour index builds, do not rely on a single final write.

Use checkpointed construction.

Recommended sequence:

```text
load one embedding part
    ↓
add vectors in fixed-size batches
    ↓
after one or more parts:
    write temporary checkpoint
    ↓
reopen checkpoint
    ↓
verify ntotal
    ↓
atomically replace previous checkpoint
    ↓
continue
```

At minimum verify:

```python
reloaded_index.ntotal == expected_vectors_added
```

Also verify available disk space before each checkpoint.

The original SQ4 experiment first completed all vector insertion but failed during final serialization because the destination disk did not have enough free space. A checkpointed rebuild avoided repeating this failure mode.

This operational failure is not part of the quality comparison, but it is relevant for reproducing large-index builds reliably.

---

## 8. Fixed benchmark queries

Use one immutable query set for every compared index.

Original run:

```text
number of queries = 200
query dimension   = 768
dtype             = float32
```

Recommended files:

```text
queries.txt
query_embeddings.npy
```

The benchmark should load `query_embeddings.npy` directly.

Do not regenerate query embeddings separately for SQ8 or SQ4.

Recommended integrity record:

```bash
sha256sum queries.txt
sha256sum query_embeddings.npy
```

The original query-text SHA-256 was:

```text
16ed6283b45f68aca278de37cdbefca44e340f24b5eaaec22dcf5a3bd94ffe52
```

The query text itself is dataset-specific and does not need to be reused on another corpus. What matters is that one fixed query set is reused across all variants within a benchmark.

---

## 9. Search parameters

The original comparison returned:

```text
k = 500
```

The main `efSearch` sweep was:

```text
512
1024
2048
4096
8192
16384
32768
```

Earlier exploratory FP32/SQ8 runs also included:

```text
64
128
256
500
1000
```

For a clean cross-index comparison, use the common sweep:

```python
EF_VALUES = [512, 1024, 2048, 4096, 8192, 16384, 32768]
K = 500
```

For FAISS HNSW:

```python
index.hnsw.efSearch = ef
scores, labels = index.search(query, K)
```

Search each query individually when measuring per-query latency:

```python
for q in queries:
    t0 = time.perf_counter()
    scores, labels = index.search(q[None, :], K)
    latency_ms = (time.perf_counter() - t0) * 1000
```

This permits calculation of:

```text
mean latency
median latency
p95 latency
```

---

## 10. Saved benchmark output

For each index and each `efSearch`, save at least:

```text
labels : shape [num_queries, K]
scores : shape [num_queries, K]
latency_ms : shape [num_queries]
```

Recommended NumPy format:

```python
np.savez_compressed(
    output_path,
    labels=labels,
    scores=scores,
    latency_ms=latency_ms,
    efSearch=np.asarray([ef], dtype=np.int64),
    k=np.asarray([K], dtype=np.int64),
)
```

Also save a run metadata file containing:

```json
{
  "dataset_vectors": 4304468,
  "dimension": 768,
  "num_queries": 200,
  "k": 500,
  "metric": "inner_product",
  "normalized": true,
  "ef_values": [512, 1024, 2048, 4096, 8192, 16384, 32768],
  "embedding_model": "intfloat/multilingual-e5-base",
  "python_version": "...",
  "faiss_version": "...",
  "hardware": "..."
}
```

---

## 11. Comparison metrics

All metrics must be computed from the same query set and same `K`.

### 11.1 Top-1 exact agreement

For `Q` queries:

```text
Top1_exact =
number of queries where result_A[0] == result_B[0]
--------------------------------------------------
Q
```

---

### 11.2 Top-K overlap

For one query:

```text
overlap@K = |A_K ∩ B_K| / K
```

Average over all queries.

The original benchmark reports:

```text
Top-5 overlap
Top-10 overlap
Top-50 overlap
Top-500 overlap
```

Example implementation:

```python
def overlap_fraction(a, b, topk):
    values = []
    for i in range(a.shape[0]):
        sa = set(map(int, a[i, :topk]))
        sb = set(map(int, b[i, :topk]))
        values.append(len(sa & sb) / topk)
    return float(np.mean(values))
```

---

### 11.3 Similarity MAE on common results

Only IDs present in both Top-K result sets for the same query are compared.

For every shared result ID:

```text
absolute error = |score_A - score_B|
```

Then average over all shared results.

This measures score distortion independently from IDs that disappeared from one result list.

---

### 11.4 Rank delta on common results

For every result ID present in both Top-K lists:

```text
rank delta = |rank_A - rank_B|
```

Report at least:

```text
mean absolute rank delta
median absolute rank delta
number of common results used
```

---

### 11.5 Self-convergence against high `efSearch`

For each quantized index, compare lower `efSearch` results against the same index at `efSearch = 32768`.

Example:

```text
SQ4 ef=512   vs SQ4 ef=32768
SQ4 ef=1024  vs SQ4 ef=32768
SQ4 ef=2048  vs SQ4 ef=32768
...
```

This is essential because it separates two different error sources:

```text
A. graph-search approximation caused by insufficient efSearch
B. representation distortion caused by quantization
```

If a low-ef quantized index already nearly matches its own high-ef result but still differs substantially from FP32, increasing `efSearch` cannot recover the missing ranking agreement.

---

## 12. Original SQ8 results

### FP32 vs SQ8 at the same search width

| ef | Top-1 exact | Top-5 overlap | Top-10 overlap | Top-50 overlap | Top-500 overlap |
|---:|---:|---:|---:|---:|---:|
| 512 | 87.0% | 89.9% | 90.0% | 91.41% | 90.22% |
| 1024 | 93.0% | 95.6% | 94.75% | 96.62% | 96.25% |
| 2048 | 94.0% | 97.4% | 96.25% | 97.88% | 97.93% |
| 4096 | 95.0% | 98.2% | 97.20% | 98.54% | 98.60% |
| 8192 | 95.0% | 98.2% | 97.25% | 98.63% | 98.72% |
| 16384 | 95.0% | 98.2% | 97.25% | 98.67% | 98.78% |
| 32768 | 95.0% | 98.2% | 97.25% | 98.67% | 98.79% |

Similarity MAE for common FP32/SQ8 results stayed approximately:

```text
0.000137–0.000139
```

### SQ8 self-convergence

Top-10 overlap relative to SQ8 `efSearch=32768`:

| SQ8 efSearch | Top-10 overlap |
|---:|---:|
| 512 | ~93% |
| 1024 | 98.05% |
| 2048 | 99.30% |
| 4096 | 99.85% |
| 8192 | 99.95% |
| 16384 | 99.95% |

### SQ8 latency

| efSearch | Median latency |
|---:|---:|
| 2048 | ~61 ms |
| 4096 | ~131 ms |
| 8192 | ~304 ms |
| 16384 | ~895 ms |
| 32768 | ~3126 ms |

For this dataset, `SQ8 + HNSW32 + efSearch=2048` was selected as the practical working point.

---

## 13. Original SQ4 results

### FP32 vs SQ4

| ef | Top-1 exact | Top-5 overlap | Top-10 overlap | Top-50 overlap | Top-500 overlap |
|---:|---:|---:|---:|---:|---:|
| 512 | 70.5% | 70.7% | 72.40% | 73.44% | 75.312% |
| 1024 | 77.0% | 75.3% | 76.15% | 77.50% | 80.073% |
| 2048 | 79.0% | 78.0% | 78.00% | 79.21% | 81.398% |
| 4096 | 79.0% | 77.8% | 78.15% | 79.26% | 81.564% |
| 8192 | 79.0% | 77.8% | 78.15% | 79.31% | 81.601% |
| 16384 | 79.0% | 77.8% | 78.20% | 79.33% | 81.628% |
| 32768 | 79.0% | 77.8% | 78.15% | 79.33% | 81.643% |

### SQ8 vs SQ4

| ef | Top-1 exact | Top-5 overlap | Top-10 overlap | Top-50 overlap | Top-500 overlap |
|---:|---:|---:|---:|---:|---:|
| 512 | 72.5% | 72.6% | 73.90% | 74.66% | 76.421% |
| 1024 | 79.0% | 75.9% | 76.45% | 77.54% | 80.115% |
| 2048 | 80.0% | 78.0% | 78.00% | 79.04% | 81.346% |
| 4096 | 81.0% | 78.1% | 78.50% | 79.26% | 81.518% |
| 8192 | 81.0% | 78.1% | 78.50% | 79.32% | 81.540% |
| 16384 | 81.0% | 78.1% | 78.50% | 79.32% | 81.584% |
| 32768 | 81.0% | 78.1% | 78.45% | 79.32% | 81.592% |

### SQ4 self-convergence and latency

| ef | SQ4 vs own ef=32768 Top-10 | Median latency | p95 latency |
|---:|---:|---:|---:|
| 512 | 90.65% | 22.31 ms | 30.19 ms |
| 1024 | 97.15% | 38.84 ms | 58.11 ms |
| 2048 | 99.55% | 70.09 ms | 99.17 ms |
| 4096 | 99.95% | 134.17 ms | 182.13 ms |
| 8192 | 100.00% | 315.95 ms | 398.01 ms |
| 16384 | 100.00% | 906.23 ms | 1051.76 ms |
| 32768 | 100.00% | 3011.33 ms | 3290.31 ms |

Similarity MAE relative to FP32:

```text
SQ8 : ~0.000137
SQ4 : ~0.00215
```

Ratio:

```text
0.00215 / 0.0001373 ≈ 15.7
```

Thus, for common retrieved IDs, SQ4 produced roughly 15.7× larger score distortion than SQ8 in this dataset.

---

## 14. Index-size analysis

Measured full index sizes:

```text
SQ8 HNSW32 = 4.170 GiB
SQ4 HNSW32 = 2.630 GiB
```

Relative size:

```text
SQ4 / SQ8 ≈ 0.631
```

Full-file saving:

```text
~36.9%
```

The reduction is less than 50% because the HNSW graph and index metadata are not quantized together with the vector codes.

### Approximate SQ8 decomposition

For 4,304,468 vectors at 768 dimensions:

```text
SQ8 vector code = 768 bytes/vector
raw SQ8 codes   ≈ 3.079 GiB
full SQ8 index  ≈ 4.170 GiB
```

Therefore:

```text
HNSW + service overhead ≈ 1.091 GiB
                         ≈ 272 bytes/vector
```

Approximate composition:

```text
~74% SQ8 vector codes
~26% HNSW graph + service data
```

This observation motivates a separate experiment that keeps SQ8 precision but reduces HNSW `M`.

---

## 15. Interpretation

At `efSearch=2048`:

```text
SQ8 vs FP32 Top-10 = 96.25%
SQ4 vs FP32 Top-10 = 78.00%
```

SQ4 at the same search width already matches its own `efSearch=32768` result by:

```text
99.55% Top-10
```

Increasing SQ4 from:

```text
efSearch 2048 → 32768
```

changes FP32 Top-10 agreement only:

```text
78.00% → 78.15%
```

while median latency changes approximately:

```text
70.09 ms → 3011.33 ms
```

or about:

```text
43× slower
```

Therefore, in this experiment, the SQ4 quality ceiling is dominated by quantization error rather than HNSW search width.

This conclusion is specific to:

```text
this embedding model
this corpus
this vector distribution
this FAISS scalar quantizer
this HNSW configuration
```

It should not be generalized to every embedding model or dataset without repeating the benchmark.

---

## 16. How to reproduce the experiment on another dataset

### Step 1 — generate one FP32 embedding corpus

Generate all document/chunk embeddings once.

Store them as FP32 NumPy arrays.

Record:

```text
number of vectors
dimension
model
normalization
dtype
```

---

### Step 2 — create one fixed query set

Select representative search queries.

Generate query embeddings once.

Save:

```text
queries.txt
query_embeddings.npy
```

Never regenerate them between index comparisons.

---

### Step 3 — build the FP32 reference

Build an FP32 search index using the same vectors.

For maximum rigor, consider two references:

```text
1. exact IndexFlatIP ground truth
2. FP32 HNSW operational reference
```

If only HNSW is used, record:

```text
M
efConstruction
efSearch
library/version
```

---

### Step 4 — build SQ8 HNSW

Use the same source vectors.

Example configuration:

```text
ScalarQuantizer = QT_8bit
M = 32
efConstruction = 200
metric = inner product
```

Record:

```text
build time
index size
ntotal after reopen
```

---

### Step 5 — build SQ4 HNSW

Use identical graph parameters:

```text
ScalarQuantizer = QT_4bit
M = 32
efConstruction = 200
metric = inner product
```

Only the scalar-quantizer precision should change.

---

### Step 6 — run identical search sweeps

Recommended:

```text
efSearch = 512, 1024, 2048, 4096, 8192, 16384, 32768
K = 500
```

Run all indexes with the same saved query embeddings.

---

### Step 7 — save raw search outputs

For every index / ef pair, save:

```text
labels
scores
per-query latency
```

Do not keep only aggregate metrics.

Raw outputs make later re-analysis possible without repeating expensive searches.

---

### Step 8 — calculate pairwise agreement

At the same `efSearch`, compare:

```text
FP32 vs SQ8
FP32 vs SQ4
SQ8  vs SQ4
```

Calculate:

```text
Top-1 exact agreement
Top-5 overlap
Top-10 overlap
Top-50 overlap
Top-500 overlap
similarity MAE on common IDs
rank delta on common IDs
```

---

### Step 9 — calculate self-convergence

For each approximate/quantized index:

```text
ef=512   vs same index ef=32768
ef=1024  vs same index ef=32768
ef=2048  vs same index ef=32768
...
```

This is necessary to distinguish insufficient graph search from representation loss.

---

### Step 10 — compare size / quality / latency

Do not evaluate index size alone.

The useful comparison is:

```text
index size
vs
ranking agreement
vs
latency
```

The selected working point should depend on the application's acceptable ranking loss.

---

## 17. Recommended result-directory layout

```text
results/
  run_YYYYMMDD_HHMMSS/
    run_meta.json
    queries.txt
    query_embeddings.npy

    fp32_ef512_results.npz
    fp32_ef1024_results.npz
    ...
    fp32_ef32768_results.npz

    sq8_ef512_results.npz
    sq8_ef1024_results.npz
    ...
    sq8_ef32768_results.npz

    sq4_ef512_results.npz
    sq4_ef1024_results.npz
    ...
    sq4_ef32768_results.npz

    pairwise_same_ef.csv
    sq8_vs_own_ef32768.csv
    sq4_vs_own_ef32768.csv
    summary.json
    summary_compact.csv
```

---

## 18. Reproducibility checklist

Before accepting a benchmark run, verify:

- [ ] all index variants contain the same number of vectors;
- [ ] all use the same embedding dimension;
- [ ] all use the same normalized source embeddings;
- [ ] all use the same similarity metric;
- [ ] all HNSW construction parameters are recorded;
- [ ] query embeddings are identical across variants;
- [ ] query count is identical;
- [ ] `K` is identical;
- [ ] `efSearch` values are identical for pairwise comparisons;
- [ ] raw labels and scores are saved;
- [ ] per-query latency is saved;
- [ ] index size is measured from the final serialized file;
- [ ] serialized indexes are reopened and `ntotal` is verified;
- [ ] software versions are recorded;
- [ ] hardware/thread settings are recorded for latency comparisons;
- [ ] results are described as overlap/agreement unless exact ground truth is available.

---

## 19. Next experiment: reduce HNSW graph size without reducing vector precision

The next experiment keeps SQ8 and changes only:

```text
M = 32 → 16
```

Everything else should remain fixed:

```text
vector representation = SQ8
dimension             = 768
metric                = inner product
normalization         = unchanged
efConstruction        = 200
source embeddings     = identical
query embeddings      = identical
K                     = 500
efSearch sweep        = identical
```

The comparison will become:

```text
FP32 reference
SQ8 HNSW32
SQ8 HNSW16
SQ4 HNSW32
```

This tests whether reducing graph storage provides a better memory/quality trade-off than reducing vector precision from SQ8 to SQ4.

A rough estimate from the measured SQ8 HNSW32 file suggests that SQ8 HNSW16 may be around:

```text
~3.6 GiB
```

but this is only a pre-build estimate.

The actual serialized size must be measured.

---

## 20. Limitations of the current report

The comparison is reproducible at the level of index construction parameters, software versions, query methodology, and recorded hardware.

However:

```text
the FP32 reference is HNSW, not exact IndexFlatIP ground truth;
absolute latency remains machine-dependent;
the exact source corpus is not distributed with this benchmark;
the fixed 200-query set is specific to the original corpus.
```

For a new dataset, the same methodology should be repeated using a new fixed query set generated once and reused across all index variants.

---

## 21. Minimal metadata template for future runs

```json
{
  "dataset": {
    "name": "example",
    "vectors": 1000000,
    "dimension": 768,
    "embedding_model": "intfloat/multilingual-e5-base",
    "dtype": "float32",
    "normalized": true
  },
  "software": {
    "python": "3.x",
    "faiss": "...",
    "numpy": "...",
    "hnswlib": "..."
  },
  "hardware": {
    "cpu": "...",
    "ram": "...",
    "threads": "..."
  },
  "index": {
    "type": "IndexHNSWSQ",
    "quantizer": "QT_8bit",
    "M": 32,
    "efConstruction": 200,
    "metric": "inner_product"
  },
  "benchmark": {
    "queries": 200,
    "k": 500,
    "efSearch": [512, 1024, 2048, 4096, 8192, 16384, 32768]
  }
}
```

---

## 22. Practical conclusion from the original dataset

For the tested 4.3M-vector multilingual E5 corpus:

```text
SQ8 HNSW32:
  size            4.170 GiB
  efSearch        2048
  Top-10 vs FP32  96.25%
  median latency  ~61 ms

SQ4 HNSW32:
  size            2.630 GiB
  efSearch        2048
  Top-10 vs FP32  78.00%
  median latency  ~70 ms
```

SQ4 saved about `1.54 GiB` relative to SQ8 but lost substantially more ranking agreement than could be recovered by increasing `efSearch`.

For this dataset, SQ8 HNSW32 remained the stronger working point pending the SQ8 HNSW16 experiment.
