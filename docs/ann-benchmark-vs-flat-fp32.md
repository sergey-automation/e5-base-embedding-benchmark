# ANN benchmarks vs exact FP32 (IndexFlatIP-equivalent reference)

**Updated:** 2026-11-05  |  **Reference:** exact full FP32 scan = 100%  |  **Source:** `all_index_results_vs_exact_flatip.csv` from `flatip_exact_groundtruth.zip`.

This report directly compares saved Top500 **vector IDs** with exact FP32 top-500 results for the same 200 queries. It does **not** scale previously reported HNSW-relative percentages. The pre-existing [FP32 HNSW reference report](ann-benchmark-vs-hnsw-fp32.md) retains original build and latency measurements; this report preserves those facts by reference and does not rebenchmark latency.

## Basis and metric

- Corpus: `intfloat/multilingual-e5-base`, **4,304,468** L2-normalized FP32 vectors, **768** dimensions; Inner Product similarity.
- **200 fixed queries**, **Top500** retrieved for each query; same IDs and query order are required.
- Exact ground truth computed by full FP32 matrix-vector scan over all 66 data parts, processing query embeddings using the same original input; scan **91.27 seconds total** on CPU (not median per query).
- Percentages mean **ID-set overlap** at the specified Top-k compared with exact FP32 Top-k; Top1 is exact rank-1 identity match. They are not relevance judgments. Comparisons are direct, not products of other recalls.
- Index sizes shown are **previously measured serialized file sizes**, not RAM peak. CPU/GPU search durations below, if mentioned, refer to the earlier benchmark and are not freshly measured with this reference.
- Some indexes/settings have repeat-run result files. The overview selects the best observed Top10 setting per index; the appendix keeps **all 198 CSV rows** including repeats and older points.

## Key findings

- Original cached `FP32 HNSW M32` reference (`fp32_results.npz`): **Top1 94.50%**, **Top10 94.00%**, **Top500 88.758%** against exact FP32.
- The same HNSW FP32 index with a larger search effort (`ef=32768`): **Top10 99.60%**, **Top500 99.959%**. Therefore the previous 94% figure characterized the **saved low-effort reference result**, not the best attainable FP32 HNSW search.
- **RR768 + IVF4096 RaBitQ8** (`nprobe=1024`): **Top10 98.85%** at **3.205 GiB**.
- **RR768 + IVF4096 RaBitQ4** (`nprobe=1024`): **Top10 92.80%** at **1.666 GiB**.
- **SQ8 HNSW M16** (`ef=32768`): **Top10 97.60%** at **3.657 GiB**.
- **GPU PQ384 f050 P104** (`nprobe=1024`): **Top10 89.60%** at **1.606 GiB**.
- **RR768 + IVF4096 RQ256x8** (`nprobe=1024`): **Top10 78.55%** at **1.258 GiB**.

## Best observed Top10 per index (direct exact FP32 reference)

| Platform | Index | Size (GiB) | Setting | Top1 | Top5 | Top10 | Top50 | Top500 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| CPU | FP32 HNSW M32 | 13.420 | `ef=32768` | 98.50% | 99.60% | **99.60%** | 99.94% | 99.959% |
| CPU | RR768 + IVF4096 RaBitQ8 | 3.205 | `nprobe=1024` | 98.50% | 98.70% | **98.85%** | 99.51% | 99.320% |
| CPU | SQ8 HNSW M32 | 4.170 | `ef=32768` | 96.50% | 98.30% | **97.60%** | 98.69% | 98.790% |
| CPU | SQ8 HNSW M16 | 3.657 | `ef=32768` | 96.50% | 98.30% | **97.60%** | 98.68% | 98.769% |
| CPU | SQ8 HNSW M8 | 3.402 | `ef=32768` | 96.50% | 98.00% | **97.35%** | 98.39% | 98.381% |
| CPU | RR768 + IVF4096 RaBitQ4 | 1.666 | `nprobe=1024` | 90.50% | 93.50% | **92.80%** | 94.93% | 95.414% |
| CPU | IVF4096 PQ384 | 1.584 | `nprobe=1024` | 82.00% | 89.60% | **90.20%** | 91.44% | 92.356% |
| GPU | GPU PQ384 f050 P104 | 1.606 | `nprobe=1024` | 84.00% | 88.80% | **89.60%** | 91.66% | 92.512% |
| CPU | OPQ384 + IVF4096 PQ384 | 1.586 | `nprobe=1024` | 87.00% | 89.50% | **89.45%** | 91.43% | 92.438% |
| CPU | RR768 + IVF4096 RaBitQ3 | 1.281 | `nprobe=1024` | 88.50% | 88.80% | **88.60%** | 91.27% | 92.023% |
| GPU | GPU PQ384 f025 P104 | 1.606 | `nprobe=1024` | 85.00% | 88.20% | **88.30%** | 91.19% | 92.295% |
| CPU | RR768 + IVF4096 RaBitQ2 | 0.896 | `nprobe=512` | 81.50% | 81.00% | **82.30%** | 84.02% | 84.703% |
| CPU | IVF4096 SQ4 | 1.583 | `nprobe=512` | 79.50% | 80.50% | **81.05%** | 83.34% | 84.878% |
| CPU | OPQ256 + IVF4096 PQ256 | 1.073 | `nprobe=1024` | 74.50% | 78.20% | **80.30%** | 81.26% | 83.348% |
| CPU | IVF4096 PQ256 | 1.071 | `nprobe=1024` | 78.50% | 76.60% | **78.85%** | 80.94% | 83.344% |
| CPU | RR768 + IVF4096 RQ256x8 | 1.258 | `nprobe=1024` | 75.00% | 79.00% | **78.55%** | 80.25% | 83.030% |
| CPU | SQ4 HNSW M32 | 2.630 | `ef=4096` | 80.00% | 77.80% | **78.35%** | 79.24% | 81.491% |
| CPU | OPQ192 + IVF4096 PQ192 | 0.816 | `nprobe=1024` | 67.50% | 73.20% | **74.45%** | 75.58% | 77.084% |
| GPU | GPU PQ192 f025 P104 | 0.825 | `nprobe=1024` | 65.00% | 67.90% | **69.30%** | 71.47% | 74.832% |
| GPU | GPU PQ192 f050 P104 | 0.825 | `nprobe=1024` | 68.00% | 69.00% | **69.05%** | 71.29% | 74.687% |
| CPU | IVF4096 PQ192 | 0.814 | `nprobe=1024` | 63.00% | 67.80% | **68.95%** | 72.04% | 74.681% |
| CPU | RR768 + IVF4096 RaBitQ1 | 0.463 | `nprobe=1024` | 62.50% | 64.10% | **65.95%** | 67.53% | 69.661% |
| CPU | OPQ128 + IVF4096 PQ128 | 0.560 | `nprobe=1024` | 60.00% | 59.00% | **60.80%** | 61.11% | 62.440% |
| GPU | GPU PQ96 f050 P104 | 0.435 | `nprobe=1024` | 40.50% | 44.00% | **44.50%** | 47.30% | 53.145% |
| GPU | GPU PQ96 f025 P104 | 0.435 | `nprobe=1024` | 41.50% | 41.00% | **41.85%** | 45.85% | 52.947% |
| GPU | GPU PQ96 f050 CMP90 | 0.435 | `nprobe=1024` | 36.00% | 41.10% | **41.60%** | 45.54% | 52.452% |

**Interpretation:** The table picks **highest measured Top10 regardless of latency**. It is not an under-200-ms shortlist. For instance RQ256x8 took ~22.4 seconds/query at `nprobe=1024` in its earlier CPU test; see historical report. GPU timings were not validated for direct CPU/GPU speed comparison.

## FP32 HNSW M32: search effort matters

| Reference version | Setting | Top1 | Top10 | Top500 |
|---|---|---:|---:|---:|
| Original cached ranking | `unknown` (matches saved low-effort reference) | 94.50% | 94.00% | 88.758% |
| FP32 HNSW M32 search sweep | `ef=64` | 94.50% | 94.00% | 88.758% |
| FP32 HNSW M32 search sweep | `ef=128` | 94.50% | 94.00% | 88.758% |
| FP32 HNSW M32 search sweep | `ef=256` | 94.50% | 94.00% | 88.758% |
| FP32 HNSW M32 search sweep | `ef=500` | 94.50% | 94.00% | 88.758% |
| FP32 HNSW M32 search sweep | `ef=512` | 94.50% | 94.00% | 88.914% |
| FP32 HNSW M32 search sweep | `ef=1000` | 98.00% | 97.55% | 95.371% |
| FP32 HNSW M32 search sweep | `ef=1024` | 98.00% | 97.55% | 95.445% |
| FP32 HNSW M32 search sweep | `ef=2048` | 98.50% | 99.00% | 98.233% |
| FP32 HNSW M32 search sweep | `ef=4096` | 98.50% | 99.45% | 99.361% |
| FP32 HNSW M32 search sweep | `ef=8192` | 98.50% | 99.50% | 99.760% |
| FP32 HNSW M32 search sweep | `ef=16384` | 98.50% | 99.55% | 99.908% |
| FP32 HNSW M32 search sweep | `ef=32768` | 98.50% | 99.60% | 99.959% |

## RaBitQ: same search parameter comparison

| Model | Size GiB | nprobe=512 Top10 | nprobe=1024 Top10 | nprobe=1024 Top500 |
|---|---:|---:|---:|---:|
| RaBitQ1 | 0.463 | 65.850% | 65.950% | 69.661% |
| RaBitQ2 | 0.896 | 82.300% | 82.250% | 84.963% |
| RaBitQ3 | 1.281 | 88.500% | 88.600% | 92.023% |
| RaBitQ4 | 1.666 | 92.600% | 92.800% | 95.414% |
| RaBitQ8 | 3.205 | 98.650% | 98.850% | 99.320% |

## GPU cross-check: historical HNSW-relative metric is not interchangeable

The re-computation archive includes `old_hnsw_top*` values for GPU results, but the computed GPU/HNSW overlap at `nprobe=1024` **differs** from percentages in the preceding GitHub report. The direct exact-FP32 values below come from sorted, exported GPU ID files and are computed independently. The historical discrepancy must not be silently reconciled or treated as a reranking correction.

| GPU index | Exact FP32 Top10 | Old HNSW overlap in new CSV | Historical HNSW overlap | Difference (new old − historical), pp |
|---|---:|---:|---:|---:|
| GPU PQ96 f025 P104 | 41.85% | 41.30% | 41.85% | -0.55 |
| GPU PQ96 f050 P104 | 44.50% | 43.75% | 44.50% | -0.75 |
| GPU PQ96 f050 CMP90 | 41.60% | 40.65% | 41.60% | -0.95 |
| GPU PQ192 f025 P104 | 69.30% | 67.00% | 69.30% | -2.30 |
| GPU PQ192 f050 P104 | 69.05% | 66.65% | 69.15% | -2.50 |
| GPU PQ384 f025 P104 | 88.30% | 84.85% | 88.55% | -3.70 |
| GPU PQ384 f050 P104 | 89.60% | 86.00% | 89.85% | -3.85 |

**Validation note:** `.raw` files were exported by a corrected RAFT routine which sorts by decreasing Inner Product score; all 42 GPU files were verified to be 400,000 bytes (200 × 500 × uint32). This file-size check does not independently prove that the GPU query ordering, vector-ID map, and normalization are identical to the Windows corpus. Given the observed historical-reference discrepancy, these assumptions deserve an explicit cross-platform verification before treating GPU comparisons as final. The fact that all 7 GPU methods yield plausible Top10 values is not itself a proof.

## Completeness and provenance

- Source CSV: **198 rows**, **26 distinct named index configurations** (including the FP32 HNSW reference), **42 GPU rows**, **156 CPU rows**.
- Repeated measurements: **25** repeated platform/index/setting combinations; all source rows retained in the appendix.
- The accompanying `all_index_validation_notes.txt` is empty; absence of notes is **not** proof of cross-platform identity validation.
- Previous index build times, sizes and CPU median search durations: [`ann-benchmark-vs-hnsw-fp32.md`](ann-benchmark-vs-hnsw-fp32.md).
- For exact, reproducible computation archive the original `flatip_exact_top500.npz`, the GPU raw-ID ZIP and the CSV alongside code, as appropriate; avoid committing multi-GiB FAISS index binaries.

## Full measurements — all settings, no pruning

Each row is a direct comparison with exact full-scan FP32. For brevity, the appendix reports Top1/5/10/50/500 plus the original `source` to identify the exact archived output. Historical HNSW-relative columns remain preserved in the source CSV.

| Platform | Index | Setting | Top1 | Top5 | Top10 | Top50 | Top500 | Source file |
|---|---|---|---:|---:|---:|---:|---:|---|
| CPU | FP32 HNSW M32 | `unknown` | 94.50% | 93.70% | 94.00% | 94.25% | 88.758% | `fp32_results.npz` |
| CPU | FP32 HNSW M32 | `unknown` | 94.50% | 93.70% | 94.00% | 94.25% | 88.758% | `fp32_results.npz` |
| CPU | FP32 HNSW M32 | `unknown` | 94.50% | 93.70% | 94.00% | 94.25% | 88.758% | `fp32_results.npz` |
| CPU | FP32 HNSW M32 | `ef=64` | 94.50% | 93.70% | 94.00% | 94.25% | 88.758% | `fp32_ef64_results.npz` |
| CPU | FP32 HNSW M32 | `ef=128` | 94.50% | 93.70% | 94.00% | 94.25% | 88.758% | `fp32_ef128_results.npz` |
| CPU | FP32 HNSW M32 | `ef=256` | 94.50% | 93.70% | 94.00% | 94.25% | 88.758% | `fp32_ef256_results.npz` |
| CPU | FP32 HNSW M32 | `ef=500` | 94.50% | 93.70% | 94.00% | 94.25% | 88.758% | `fp32_ef500_results.npz` |
| CPU | FP32 HNSW M32 | `ef=512` | 94.50% | 93.70% | 94.00% | 94.29% | 88.914% | `fp32_ef512_results.npz` |
| CPU | FP32 HNSW M32 | `ef=1000` | 98.00% | 97.40% | 97.55% | 98.01% | 95.371% | `fp32_ef1000_results.npz` |
| CPU | FP32 HNSW M32 | `ef=1024` | 98.00% | 97.40% | 97.55% | 98.05% | 95.445% | `fp32_ef1024_results.npz` |
| CPU | FP32 HNSW M32 | `ef=2048` | 98.50% | 99.30% | 99.00% | 99.28% | 98.233% | `fp32_ef2048_results.npz` |
| CPU | FP32 HNSW M32 | `ef=4096` | 98.50% | 99.50% | 99.45% | 99.74% | 99.361% | `fp32_ef4096_results.npz` |
| CPU | FP32 HNSW M32 | `ef=8192` | 98.50% | 99.50% | 99.50% | 99.87% | 99.760% | `fp32_ef8192_results.npz` |
| CPU | FP32 HNSW M32 | `ef=16384` | 98.50% | 99.50% | 99.55% | 99.91% | 99.908% | `fp32_ef16384_results.npz` |
| CPU | FP32 HNSW M32 | `ef=32768` | 98.50% | 99.60% | 99.60% | 99.94% | 99.959% | `fp32_ef32768_results.npz` |
| CPU | IVF4096 PQ192 | `nprobe=32` | 59.00% | 63.00% | 64.70% | 67.23% | 67.870% | `IVF_PQ192_nprobe32.npz` |
| CPU | IVF4096 PQ192 | `nprobe=64` | 60.50% | 65.00% | 66.05% | 69.56% | 70.821% | `IVF_PQ192_nprobe64.npz` |
| CPU | IVF4096 PQ192 | `nprobe=128` | 61.50% | 66.00% | 67.20% | 70.82% | 72.737% | `IVF_PQ192_nprobe128.npz` |
| CPU | IVF4096 PQ192 | `nprobe=256` | 62.00% | 67.00% | 68.20% | 71.58% | 73.931% | `IVF_PQ192_nprobe256.npz` |
| CPU | IVF4096 PQ192 | `nprobe=512` | 62.50% | 67.70% | 68.85% | 71.98% | 74.470% | `IVF_PQ192_nprobe512.npz` |
| CPU | IVF4096 PQ192 | `nprobe=1024` | 63.00% | 67.80% | 68.95% | 72.04% | 74.681% | `IVF_PQ192_nprobe1024.npz` |
| CPU | IVF4096 PQ256 | `nprobe=32` | 71.00% | 69.80% | 73.25% | 74.68% | 74.239% | `IVF_PQ256_nprobe32.npz` |
| CPU | IVF4096 PQ256 | `nprobe=64` | 73.50% | 72.20% | 75.00% | 77.46% | 78.214% | `IVF_PQ256_nprobe64.npz` |
| CPU | IVF4096 PQ256 | `nprobe=128` | 76.00% | 74.00% | 76.85% | 79.18% | 80.780% | `IVF_PQ256_nprobe128.npz` |
| CPU | IVF4096 PQ256 | `nprobe=256` | 77.50% | 75.50% | 78.05% | 80.36% | 82.391% | `IVF_PQ256_nprobe256.npz` |
| CPU | IVF4096 PQ256 | `nprobe=512` | 78.00% | 76.40% | 78.75% | 80.82% | 83.092% | `IVF_PQ256_nprobe512.npz` |
| CPU | IVF4096 PQ256 | `nprobe=1024` | 78.50% | 76.60% | 78.85% | 80.94% | 83.344% | `IVF_PQ256_nprobe1024.npz` |
| CPU | IVF4096 PQ384 | `nprobe=32` | 74.50% | 81.20% | 82.35% | 83.11% | 80.128% | `IVF_PQ384_nprobe32.npz` |
| CPU | IVF4096 PQ384 | `nprobe=64` | 76.50% | 84.10% | 85.35% | 86.78% | 85.250% | `IVF_PQ384_nprobe64.npz` |
| CPU | IVF4096 PQ384 | `nprobe=128` | 78.50% | 86.40% | 87.75% | 89.15% | 88.791% | `IVF_PQ384_nprobe128.npz` |
| CPU | IVF4096 PQ384 | `nprobe=256` | 80.50% | 88.10% | 89.05% | 90.62% | 91.039% | `IVF_PQ384_nprobe256.npz` |
| CPU | IVF4096 PQ384 | `nprobe=512` | 81.50% | 89.40% | 90.05% | 91.28% | 92.016% | `IVF_PQ384_nprobe512.npz` |
| CPU | IVF4096 PQ384 | `nprobe=1024` | 82.00% | 89.60% | 90.20% | 91.44% | 92.356% | `IVF_PQ384_nprobe1024.npz` |
| CPU | IVF4096 SQ4 | `nprobe=32` | 71.50% | 73.40% | 75.10% | 76.69% | 75.663% | `IVF_SQ4_nprobe32.npz` |
| CPU | IVF4096 SQ4 | `nprobe=64` | 75.00% | 76.00% | 77.55% | 79.68% | 79.803% | `IVF_SQ4_nprobe64.npz` |
| CPU | IVF4096 SQ4 | `nprobe=128` | 77.50% | 78.00% | 79.20% | 81.69% | 82.440% | `IVF_SQ4_nprobe128.npz` |
| CPU | IVF4096 SQ4 | `nprobe=256` | 78.50% | 79.60% | 80.30% | 82.78% | 84.115% | `IVF_SQ4_nprobe256.npz` |
| CPU | IVF4096 SQ4 | `nprobe=512` | 79.50% | 80.50% | 81.05% | 83.34% | 84.878% | `IVF_SQ4_nprobe512.npz` |
| CPU | IVF4096 SQ4 | `nprobe=1024` | 80.00% | 80.70% | 80.95% | 83.50% | 85.155% | `IVF_SQ4_nprobe1024.npz` |
| CPU | OPQ128 + IVF4096 PQ128 | `nprobe=32` | 53.50% | 54.30% | 57.00% | 57.52% | 58.562% | `OPQ_IVF_PQ128_nprobe32.npz` |
| CPU | OPQ128 + IVF4096 PQ128 | `nprobe=64` | 56.00% | 56.10% | 58.25% | 58.98% | 60.392% | `OPQ_IVF_PQ128_nprobe64.npz` |
| CPU | OPQ128 + IVF4096 PQ128 | `nprobe=128` | 58.00% | 57.30% | 59.40% | 60.17% | 61.471% | `OPQ_IVF_PQ128_nprobe128.npz` |
| CPU | OPQ128 + IVF4096 PQ128 | `nprobe=256` | 59.50% | 58.20% | 60.35% | 60.78% | 62.049% | `OPQ_IVF_PQ128_nprobe256.npz` |
| CPU | OPQ128 + IVF4096 PQ128 | `nprobe=512` | 59.50% | 58.90% | 60.70% | 61.08% | 62.346% | `OPQ_IVF_PQ128_nprobe512.npz` |
| CPU | OPQ128 + IVF4096 PQ128 | `nprobe=1024` | 60.00% | 59.00% | 60.80% | 61.11% | 62.440% | `OPQ_IVF_PQ128_nprobe1024.npz` |
| CPU | OPQ192 + IVF4096 PQ192 | `nprobe=32` | 61.50% | 66.50% | 69.65% | 69.98% | 70.052% | `OPQ_IVF_PQ192_nprobe32.npz` |
| CPU | OPQ192 + IVF4096 PQ192 | `nprobe=64` | 63.00% | 68.90% | 71.15% | 72.54% | 73.196% | `OPQ_IVF_PQ192_nprobe64.npz` |
| CPU | OPQ192 + IVF4096 PQ192 | `nprobe=128` | 65.50% | 71.00% | 72.85% | 74.10% | 75.231% | `OPQ_IVF_PQ192_nprobe128.npz` |
| CPU | OPQ192 + IVF4096 PQ192 | `nprobe=256` | 66.50% | 72.40% | 73.90% | 75.02% | 76.364% | `OPQ_IVF_PQ192_nprobe256.npz` |
| CPU | OPQ192 + IVF4096 PQ192 | `nprobe=512` | 67.00% | 73.00% | 74.35% | 75.47% | 76.906% | `OPQ_IVF_PQ192_nprobe512.npz` |
| CPU | OPQ192 + IVF4096 PQ192 | `nprobe=1024` | 67.50% | 73.20% | 74.45% | 75.58% | 77.084% | `OPQ_IVF_PQ192_nprobe1024.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=32` | 67.00% | 72.20% | 73.80% | 74.71% | 74.189% | `OPQ_IVF_PQ256_nprobe32.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=32` | 67.00% | 72.20% | 73.80% | 74.71% | 74.189% | `OPQ_IVF_PQ256_nprobe32.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=64` | 70.00% | 74.20% | 76.25% | 77.56% | 78.170% | `OPQ_IVF_PQ256_nprobe64.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=64` | 70.00% | 74.20% | 76.25% | 77.56% | 78.170% | `OPQ_IVF_PQ256_nprobe64.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=128` | 72.50% | 76.10% | 78.20% | 79.45% | 80.757% | `OPQ_IVF_PQ256_nprobe128.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=128` | 72.50% | 76.10% | 78.20% | 79.45% | 80.757% | `OPQ_IVF_PQ256_nprobe128.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=256` | 74.00% | 77.30% | 79.55% | 80.66% | 82.319% | `OPQ_IVF_PQ256_nprobe256.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=256` | 74.00% | 77.30% | 79.55% | 80.66% | 82.319% | `OPQ_IVF_PQ256_nprobe256.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=512` | 74.00% | 78.10% | 80.20% | 81.19% | 83.086% | `OPQ_IVF_PQ256_nprobe512.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=512` | 74.00% | 78.10% | 80.20% | 81.19% | 83.086% | `OPQ_IVF_PQ256_nprobe512.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=1024` | 74.50% | 78.20% | 80.30% | 81.26% | 83.348% | `OPQ_IVF_PQ256_nprobe1024.npz` |
| CPU | OPQ256 + IVF4096 PQ256 | `nprobe=1024` | 74.50% | 78.20% | 80.30% | 81.26% | 83.348% | `OPQ_IVF_PQ256_nprobe1024.npz` |
| CPU | OPQ384 + IVF4096 PQ384 | `nprobe=32` | 79.00% | 81.50% | 81.75% | 83.07% | 80.164% | `OPQ_IVF_PQ384_nprobe32.npz` |
| CPU | OPQ384 + IVF4096 PQ384 | `nprobe=64` | 81.00% | 84.10% | 84.50% | 86.83% | 85.378% | `OPQ_IVF_PQ384_nprobe64.npz` |
| CPU | OPQ384 + IVF4096 PQ384 | `nprobe=128` | 84.00% | 86.50% | 86.95% | 89.32% | 88.858% | `OPQ_IVF_PQ384_nprobe128.npz` |
| CPU | OPQ384 + IVF4096 PQ384 | `nprobe=256` | 86.00% | 88.40% | 88.65% | 90.72% | 91.068% | `OPQ_IVF_PQ384_nprobe256.npz` |
| CPU | OPQ384 + IVF4096 PQ384 | `nprobe=512` | 86.50% | 89.40% | 89.30% | 91.24% | 92.078% | `OPQ_IVF_PQ384_nprobe512.npz` |
| CPU | OPQ384 + IVF4096 PQ384 | `nprobe=1024` | 87.00% | 89.50% | 89.45% | 91.43% | 92.438% | `OPQ_IVF_PQ384_nprobe1024.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=32` | 68.50% | 72.30% | 72.00% | 73.92% | 73.738% | `RR768_IVF4096_RQ256x8_nprobe32.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=32` | 68.50% | 72.30% | 72.00% | 73.92% | 73.738% | `RR768_IVF4096_RQ256x8_nprobe32.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=64` | 70.50% | 74.50% | 74.50% | 76.58% | 77.674% | `RR768_IVF4096_RQ256x8_nprobe64.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=64` | 70.50% | 74.50% | 74.50% | 76.58% | 77.674% | `RR768_IVF4096_RQ256x8_nprobe64.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=128` | 72.00% | 76.70% | 76.35% | 78.47% | 80.295% | `RR768_IVF4096_RQ256x8_nprobe128.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=128` | 72.00% | 76.70% | 76.35% | 78.47% | 80.295% | `RR768_IVF4096_RQ256x8_nprobe128.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=256` | 73.50% | 77.90% | 77.80% | 79.73% | 81.949% | `RR768_IVF4096_RQ256x8_nprobe256.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=256` | 73.50% | 77.90% | 77.80% | 79.73% | 81.949% | `RR768_IVF4096_RQ256x8_nprobe256.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=512` | 74.50% | 78.80% | 78.50% | 80.18% | 82.741% | `RR768_IVF4096_RQ256x8_nprobe512.npz` |
| CPU | RR768 + IVF4096 RQ256x8 | `nprobe=1024` | 75.00% | 79.00% | 78.55% | 80.25% | 83.030% | `RR768_IVF4096_RQ256x8_nprobe1024.npz` |
| CPU | RR768 + IVF4096 RaBitQ1 | `nprobe=32` | 58.00% | 58.90% | 61.15% | 62.77% | 64.224% | `RR768_IVF4096_RaBitQ1_nprobe32.npz` |
| CPU | RR768 + IVF4096 RaBitQ1 | `nprobe=64` | 59.00% | 61.10% | 63.35% | 64.99% | 66.614% | `RR768_IVF4096_RaBitQ1_nprobe64.npz` |
| CPU | RR768 + IVF4096 RaBitQ1 | `nprobe=128` | 60.00% | 62.40% | 64.70% | 66.31% | 68.236% | `RR768_IVF4096_RaBitQ1_nprobe128.npz` |
| CPU | RR768 + IVF4096 RaBitQ1 | `nprobe=256` | 61.50% | 63.70% | 65.45% | 67.15% | 69.117% | `RR768_IVF4096_RaBitQ1_nprobe256.npz` |
| CPU | RR768 + IVF4096 RaBitQ1 | `nprobe=512` | 62.00% | 64.00% | 65.85% | 67.47% | 69.543% | `RR768_IVF4096_RaBitQ1_nprobe512.npz` |
| CPU | RR768 + IVF4096 RaBitQ1 | `nprobe=1024` | 62.50% | 64.10% | 65.95% | 67.53% | 69.661% | `RR768_IVF4096_RaBitQ1_nprobe1024.npz` |
| CPU | RR768 + IVF4096 RaBitQ2 | `nprobe=32` | 75.00% | 74.20% | 75.35% | 77.17% | 75.607% | `RR768_IVF4096_RaBitQ2_nprobe32.npz` |
| CPU | RR768 + IVF4096 RaBitQ2 | `nprobe=64` | 77.50% | 76.90% | 78.05% | 80.08% | 79.652% | `RR768_IVF4096_RaBitQ2_nprobe64.npz` |
| CPU | RR768 + IVF4096 RaBitQ2 | `nprobe=128` | 78.50% | 78.60% | 80.20% | 82.38% | 82.438% | `RR768_IVF4096_RaBitQ2_nprobe128.npz` |
| CPU | RR768 + IVF4096 RaBitQ2 | `nprobe=256` | 80.50% | 80.20% | 81.65% | 83.55% | 83.942% | `RR768_IVF4096_RaBitQ2_nprobe256.npz` |
| CPU | RR768 + IVF4096 RaBitQ2 | `nprobe=512` | 81.50% | 81.00% | 82.30% | 84.02% | 84.703% | `RR768_IVF4096_RaBitQ2_nprobe512.npz` |
| CPU | RR768 + IVF4096 RaBitQ2 | `nprobe=1024` | 82.00% | 81.10% | 82.25% | 84.15% | 84.963% | `RR768_IVF4096_RaBitQ2_nprobe1024.npz` |
| CPU | RR768 + IVF4096 RaBitQ3 | `nprobe=32` | 79.50% | 80.10% | 80.75% | 82.90% | 79.948% | `RR768_IVF4096_RaBitQ3_nprobe32.npz` |
| CPU | RR768 + IVF4096 RaBitQ3 | `nprobe=64` | 83.00% | 83.30% | 83.95% | 86.45% | 85.030% | `RR768_IVF4096_RaBitQ3_nprobe64.npz` |
| CPU | RR768 + IVF4096 RaBitQ3 | `nprobe=128` | 85.50% | 85.80% | 86.15% | 89.19% | 88.636% | `RR768_IVF4096_RaBitQ3_nprobe128.npz` |
| CPU | RR768 + IVF4096 RaBitQ3 | `nprobe=256` | 87.00% | 87.60% | 87.70% | 90.64% | 90.703% | `RR768_IVF4096_RaBitQ3_nprobe256.npz` |
| CPU | RR768 + IVF4096 RaBitQ3 | `nprobe=512` | 88.00% | 88.70% | 88.50% | 91.11% | 91.690% | `RR768_IVF4096_RaBitQ3_nprobe512.npz` |
| CPU | RR768 + IVF4096 RaBitQ3 | `nprobe=1024` | 88.50% | 88.80% | 88.60% | 91.27% | 92.023% | `RR768_IVF4096_RaBitQ3_nprobe1024.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=32` | 81.00% | 84.40% | 84.25% | 85.36% | 81.584% | `RR768_IVF4096_RaBitQ4_nprobe32.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=32` | 81.00% | 84.40% | 84.25% | 85.36% | 81.584% | `RR768_IVF4096_RaBitQ4_nprobe32.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=64` | 84.50% | 88.10% | 87.55% | 89.40% | 87.215% | `RR768_IVF4096_RaBitQ4_nprobe64.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=64` | 84.50% | 88.10% | 87.55% | 89.40% | 87.215% | `RR768_IVF4096_RaBitQ4_nprobe64.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=128` | 87.00% | 90.40% | 90.10% | 92.41% | 91.414% | `RR768_IVF4096_RaBitQ4_nprobe128.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=128` | 87.00% | 90.40% | 90.10% | 92.41% | 91.414% | `RR768_IVF4096_RaBitQ4_nprobe128.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=256` | 89.00% | 92.50% | 91.80% | 94.14% | 93.896% | `RR768_IVF4096_RaBitQ4_nprobe256.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=256` | 89.00% | 92.50% | 91.80% | 94.14% | 93.896% | `RR768_IVF4096_RaBitQ4_nprobe256.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=512` | 90.00% | 93.40% | 92.60% | 94.75% | 95.011% | `RR768_IVF4096_RaBitQ4_nprobe512.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=512` | 90.00% | 93.40% | 92.60% | 94.75% | 95.011% | `RR768_IVF4096_RaBitQ4_nprobe512.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=1024` | 90.50% | 93.50% | 92.80% | 94.93% | 95.414% | `RR768_IVF4096_RaBitQ4_nprobe1024.npz` |
| CPU | RR768 + IVF4096 RaBitQ4 | `nprobe=1024` | 90.50% | 93.50% | 92.80% | 94.93% | 95.414% | `RR768_IVF4096_RaBitQ4_nprobe1024.npz` |
| CPU | RR768 + IVF4096 RaBitQ8 | `nprobe=32` | 87.50% | 87.60% | 88.35% | 87.66% | 82.721% | `RR768_IVF4096_RaBitQ8_nprobe32.npz` |
| CPU | RR768 + IVF4096 RaBitQ8 | `nprobe=64` | 91.50% | 91.70% | 92.15% | 92.35% | 88.942% | `RR768_IVF4096_RaBitQ8_nprobe64.npz` |
| CPU | RR768 + IVF4096 RaBitQ8 | `nprobe=128` | 94.50% | 94.60% | 95.20% | 95.92% | 93.768% | `RR768_IVF4096_RaBitQ8_nprobe128.npz` |
| CPU | RR768 + IVF4096 RaBitQ8 | `nprobe=256` | 97.00% | 97.30% | 97.55% | 98.18% | 96.999% | `RR768_IVF4096_RaBitQ8_nprobe256.npz` |
| CPU | RR768 + IVF4096 RaBitQ8 | `nprobe=512` | 98.00% | 98.50% | 98.65% | 99.18% | 98.702% | `RR768_IVF4096_RaBitQ8_nprobe512.npz` |
| CPU | RR768 + IVF4096 RaBitQ8 | `nprobe=1024` | 98.50% | 98.70% | 98.85% | 99.51% | 99.320% | `RR768_IVF4096_RaBitQ8_nprobe1024.npz` |
| CPU | SQ4 HNSW M32 | `ef=512` | 71.50% | 71.70% | 72.40% | 74.00% | 75.470% | `sq4_ef512_results.npz` |
| CPU | SQ4 HNSW M32 | `ef=1024` | 78.50% | 76.10% | 76.55% | 77.80% | 80.093% | `sq4_ef1024_results.npz` |
| CPU | SQ4 HNSW M32 | `ef=2048` | 80.00% | 78.00% | 78.10% | 79.21% | 81.317% | `sq4_ef2048_results.npz` |
| CPU | SQ4 HNSW M32 | `ef=4096` | 80.00% | 77.80% | 78.35% | 79.24% | 81.491% | `sq4_ef4096_results.npz` |
| CPU | SQ4 HNSW M32 | `ef=8192` | 80.00% | 77.80% | 78.35% | 79.31% | 81.593% | `sq4_ef8192_results.npz` |
| CPU | SQ4 HNSW M32 | `ef=16384` | 80.00% | 77.80% | 78.35% | 79.32% | 81.634% | `sq4_ef16384_results.npz` |
| CPU | SQ4 HNSW M32 | `ef=32768` | 80.00% | 77.80% | 78.35% | 79.33% | 81.646% | `sq4_ef32768_results.npz` |
| CPU | SQ8 HNSW M32 | `unknown` | 51.50% | 54.90% | 54.70% | 53.73% | 38.641% | `sq8_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=64` | 51.50% | 54.90% | 54.70% | 53.73% | 38.641% | `sq8_ef64_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=128` | 64.50% | 68.10% | 68.15% | 68.65% | 56.435% | `sq8_ef128_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=256` | 80.00% | 82.10% | 82.25% | 83.84% | 76.083% | `sq8_ef256_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=500` | 89.00% | 91.40% | 90.95% | 92.42% | 88.548% | `sq8_ef500_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=512` | 89.00% | 91.40% | 90.95% | 92.43% | 88.680% | `sq8_ef512_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=1000` | 94.50% | 96.70% | 95.70% | 97.11% | 95.540% | `sq8_ef1000_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=1024` | 94.50% | 96.70% | 95.75% | 97.12% | 95.608% | `sq8_ef1024_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=2048` | 95.50% | 97.60% | 96.95% | 98.02% | 97.694% | `sq8_ef2048_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=4096` | 96.50% | 98.20% | 97.50% | 98.59% | 98.473% | `sq8_ef4096_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=8192` | 96.50% | 98.20% | 97.55% | 98.67% | 98.684% | `sq8_ef8192_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=16384` | 96.50% | 98.20% | 97.55% | 98.68% | 98.770% | `sq8_ef16384_results.npz` |
| CPU | SQ8 HNSW M32 | `ef=32768` | 96.50% | 98.30% | 97.60% | 98.69% | 98.790% | `sq8_ef32768_results.npz` |
| CPU | SQ8 HNSW M16 | `ef=512` | 84.50% | 88.50% | 87.80% | 89.39% | 84.358% | `sq8_m16_ef512_results.npz` |
| CPU | SQ8 HNSW M16 | `ef=1024` | 92.00% | 94.10% | 93.30% | 94.89% | 92.756% | `sq8_m16_ef1024_results.npz` |
| CPU | SQ8 HNSW M16 | `ef=2048` | 95.50% | 97.40% | 96.50% | 97.80% | 96.908% | `sq8_m16_ef2048_results.npz` |
| CPU | SQ8 HNSW M16 | `ef=4096` | 96.50% | 98.10% | 97.35% | 98.49% | 98.138% | `sq8_m16_ef4096_results.npz` |
| CPU | SQ8 HNSW M16 | `ef=8192` | 96.50% | 98.20% | 97.55% | 98.62% | 98.531% | `sq8_m16_ef8192_results.npz` |
| CPU | SQ8 HNSW M16 | `ef=16384` | 96.50% | 98.20% | 97.55% | 98.67% | 98.710% | `sq8_m16_ef16384_results.npz` |
| CPU | SQ8 HNSW M16 | `ef=32768` | 96.50% | 98.30% | 97.60% | 98.68% | 98.769% | `sq8_m16_ef32768_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=512` | 74.50% | 78.10% | 78.80% | 80.18% | 72.363% | `sq8_m8_ef512_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=512` | 74.50% | 78.10% | 78.80% | 80.18% | 72.363% | `sq8_m8_ef512_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=1024` | 84.50% | 87.50% | 88.05% | 90.01% | 85.754% | `sq8_m8_ef1024_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=1024` | 84.50% | 87.50% | 88.05% | 90.01% | 85.754% | `sq8_m8_ef1024_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=2048` | 92.00% | 94.40% | 93.85% | 95.57% | 93.281% | `sq8_m8_ef2048_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=2048` | 92.00% | 94.40% | 93.85% | 95.57% | 93.281% | `sq8_m8_ef2048_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=4096` | 94.50% | 96.50% | 96.25% | 97.57% | 96.438% | `sq8_m8_ef4096_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=4096` | 94.50% | 96.50% | 96.25% | 97.57% | 96.438% | `sq8_m8_ef4096_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=8192` | 96.00% | 97.40% | 96.90% | 98.00% | 97.593% | `sq8_m8_ef8192_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=8192` | 96.00% | 97.40% | 96.90% | 98.00% | 97.593% | `sq8_m8_ef8192_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=16384` | 96.50% | 97.80% | 97.20% | 98.26% | 98.122% | `sq8_m8_ef16384_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=16384` | 96.50% | 97.80% | 97.20% | 98.26% | 98.122% | `sq8_m8_ef16384_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=32768` | 96.50% | 98.00% | 97.35% | 98.39% | 98.381% | `sq8_m8_ef32768_results.npz` |
| CPU | SQ8 HNSW M8 | `ef=32768` | 96.50% | 98.00% | 97.35% | 98.39% | 98.381% | `sq8_m8_ef32768_results.npz` |
| GPU | GPU PQ192 f025 P104 | `nprobe=32` | 61.00% | 64.90% | 66.05% | 68.08% | 70.501% | `PQ192_f025_P104_nprobe32_top500_u32.raw` |
| GPU | GPU PQ192 f025 P104 | `nprobe=64` | 63.00% | 66.30% | 67.55% | 69.72% | 72.633% | `PQ192_f025_P104_nprobe64_top500_u32.raw` |
| GPU | GPU PQ192 f025 P104 | `nprobe=128` | 63.00% | 67.00% | 68.45% | 70.62% | 73.913% | `PQ192_f025_P104_nprobe128_top500_u32.raw` |
| GPU | GPU PQ192 f025 P104 | `nprobe=256` | 63.50% | 67.30% | 68.90% | 71.07% | 74.462% | `PQ192_f025_P104_nprobe256_top500_u32.raw` |
| GPU | GPU PQ192 f025 P104 | `nprobe=512` | 64.00% | 67.60% | 69.20% | 71.33% | 74.735% | `PQ192_f025_P104_nprobe512_top500_u32.raw` |
| GPU | GPU PQ192 f025 P104 | `nprobe=1024` | 65.00% | 67.90% | 69.30% | 71.47% | 74.832% | `PQ192_f025_P104_nprobe1024_top500_u32.raw` |
| GPU | GPU PQ192 f050 P104 | `nprobe=32` | 65.00% | 65.20% | 65.60% | 68.15% | 70.318% | `PQ192_f050_P104_nprobe32_top500_u32.raw` |
| GPU | GPU PQ192 f050 P104 | `nprobe=64` | 66.00% | 66.20% | 67.05% | 69.72% | 72.592% | `PQ192_f050_P104_nprobe64_top500_u32.raw` |
| GPU | GPU PQ192 f050 P104 | `nprobe=128` | 67.00% | 67.70% | 68.30% | 70.55% | 73.723% | `PQ192_f050_P104_nprobe128_top500_u32.raw` |
| GPU | GPU PQ192 f050 P104 | `nprobe=256` | 67.50% | 68.50% | 68.65% | 71.01% | 74.334% | `PQ192_f050_P104_nprobe256_top500_u32.raw` |
| GPU | GPU PQ192 f050 P104 | `nprobe=512` | 67.50% | 68.70% | 68.85% | 71.20% | 74.592% | `PQ192_f050_P104_nprobe512_top500_u32.raw` |
| GPU | GPU PQ192 f050 P104 | `nprobe=1024` | 68.00% | 69.00% | 69.05% | 71.29% | 74.687% | `PQ192_f050_P104_nprobe1024_top500_u32.raw` |
| GPU | GPU PQ384 f025 P104 | `nprobe=32` | 80.50% | 83.20% | 82.70% | 85.31% | 84.553% | `PQ384_f025_P104_nprobe32_top500_u32.raw` |
| GPU | GPU PQ384 f025 P104 | `nprobe=64` | 82.50% | 85.40% | 85.40% | 88.38% | 88.474% | `PQ384_f025_P104_nprobe64_top500_u32.raw` |
| GPU | GPU PQ384 f025 P104 | `nprobe=128` | 84.00% | 86.70% | 86.95% | 89.94% | 90.545% | `PQ384_f025_P104_nprobe128_top500_u32.raw` |
| GPU | GPU PQ384 f025 P104 | `nprobe=256` | 84.50% | 87.60% | 87.85% | 90.73% | 91.594% | `PQ384_f025_P104_nprobe256_top500_u32.raw` |
| GPU | GPU PQ384 f025 P104 | `nprobe=512` | 84.50% | 88.00% | 88.15% | 91.10% | 92.087% | `PQ384_f025_P104_nprobe512_top500_u32.raw` |
| GPU | GPU PQ384 f025 P104 | `nprobe=1024` | 85.00% | 88.20% | 88.30% | 91.19% | 92.295% | `PQ384_f025_P104_nprobe1024_top500_u32.raw` |
| GPU | GPU PQ384 f050 P104 | `nprobe=32` | 80.50% | 83.10% | 83.95% | 85.36% | 84.452% | `PQ384_f050_P104_nprobe32_top500_u32.raw` |
| GPU | GPU PQ384 f050 P104 | `nprobe=64` | 83.00% | 85.50% | 86.60% | 88.52% | 88.490% | `PQ384_f050_P104_nprobe64_top500_u32.raw` |
| GPU | GPU PQ384 f050 P104 | `nprobe=128` | 83.50% | 87.10% | 88.05% | 90.18% | 90.642% | `PQ384_f050_P104_nprobe128_top500_u32.raw` |
| GPU | GPU PQ384 f050 P104 | `nprobe=256` | 84.00% | 87.80% | 88.95% | 91.09% | 91.792% | `PQ384_f050_P104_nprobe256_top500_u32.raw` |
| GPU | GPU PQ384 f050 P104 | `nprobe=512` | 84.00% | 88.40% | 89.25% | 91.42% | 92.290% | `PQ384_f050_P104_nprobe512_top500_u32.raw` |
| GPU | GPU PQ384 f050 P104 | `nprobe=1024` | 84.00% | 88.80% | 89.60% | 91.66% | 92.512% | `PQ384_f050_P104_nprobe1024_top500_u32.raw` |
| GPU | GPU PQ96 f025 P104 | `nprobe=32` | 41.00% | 39.70% | 40.70% | 44.76% | 51.379% | `PQ96_f025_P104_nprobe32_top500_u32.raw` |
| GPU | GPU PQ96 f025 P104 | `nprobe=64` | 41.00% | 40.30% | 41.25% | 45.33% | 52.243% | `PQ96_f025_P104_nprobe64_top500_u32.raw` |
| GPU | GPU PQ96 f025 P104 | `nprobe=128` | 41.50% | 40.50% | 41.50% | 45.63% | 52.689% | `PQ96_f025_P104_nprobe128_top500_u32.raw` |
| GPU | GPU PQ96 f025 P104 | `nprobe=256` | 41.50% | 40.70% | 41.70% | 45.80% | 52.865% | `PQ96_f025_P104_nprobe256_top500_u32.raw` |
| GPU | GPU PQ96 f025 P104 | `nprobe=512` | 41.00% | 40.90% | 41.75% | 45.82% | 52.928% | `PQ96_f025_P104_nprobe512_top500_u32.raw` |
| GPU | GPU PQ96 f025 P104 | `nprobe=1024` | 41.50% | 41.00% | 41.85% | 45.85% | 52.947% | `PQ96_f025_P104_nprobe1024_top500_u32.raw` |
| GPU | GPU PQ96 f050 CMP90 | `nprobe=32` | 34.50% | 39.70% | 39.95% | 44.14% | 51.003% | `PQ96_f050_CMP90_nprobe32_top500_u32.raw` |
| GPU | GPU PQ96 f050 CMP90 | `nprobe=64` | 35.00% | 40.30% | 40.60% | 44.87% | 51.765% | `PQ96_f050_CMP90_nprobe64_top500_u32.raw` |
| GPU | GPU PQ96 f050 CMP90 | `nprobe=128` | 35.00% | 40.90% | 41.15% | 45.20% | 52.168% | `PQ96_f050_CMP90_nprobe128_top500_u32.raw` |
| GPU | GPU PQ96 f050 CMP90 | `nprobe=256` | 36.00% | 40.90% | 41.40% | 45.38% | 52.340% | `PQ96_f050_CMP90_nprobe256_top500_u32.raw` |
| GPU | GPU PQ96 f050 CMP90 | `nprobe=512` | 36.00% | 41.00% | 41.45% | 45.44% | 52.405% | `PQ96_f050_CMP90_nprobe512_top500_u32.raw` |
| GPU | GPU PQ96 f050 CMP90 | `nprobe=1024` | 36.00% | 41.10% | 41.60% | 45.54% | 52.452% | `PQ96_f050_CMP90_nprobe1024_top500_u32.raw` |
| GPU | GPU PQ96 f050 P104 | `nprobe=32` | 38.50% | 42.40% | 43.25% | 46.39% | 51.696% | `PQ96_f050_P104_nprobe32_top500_u32.raw` |
| GPU | GPU PQ96 f050 P104 | `nprobe=64` | 39.00% | 43.00% | 43.70% | 46.85% | 52.489% | `PQ96_f050_P104_nprobe64_top500_u32.raw` |
| GPU | GPU PQ96 f050 P104 | `nprobe=128` | 40.00% | 43.70% | 44.20% | 47.10% | 52.908% | `PQ96_f050_P104_nprobe128_top500_u32.raw` |
| GPU | GPU PQ96 f050 P104 | `nprobe=256` | 40.50% | 43.90% | 44.35% | 47.20% | 53.050% | `PQ96_f050_P104_nprobe256_top500_u32.raw` |
| GPU | GPU PQ96 f050 P104 | `nprobe=512` | 40.50% | 44.00% | 44.45% | 47.30% | 53.118% | `PQ96_f050_P104_nprobe512_top500_u32.raw` |
| GPU | GPU PQ96 f050 P104 | `nprobe=1024` | 40.50% | 44.00% | 44.50% | 47.30% | 53.145% | `PQ96_f050_P104_nprobe1024_top500_u32.raw` |

---

**Scope:** this file changes only the **quality reference** to exact FP32. It does not overwrite older experimental results or infer missing latencies. Cross-platform GPU results should be interpreted with the validation caveat above.
