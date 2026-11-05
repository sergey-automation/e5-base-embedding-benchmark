from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np


# ============================================================
# COMMON BENCHMARK BASE
# ============================================================

NQ = 200
DIM = 768
TOPK = 500
OMP_THREADS = 4

QUERY_EMB = Path(
    r"C:\VAST2_UPLOAD\out\fp32_ef_sweep\20261029_143923\query_embeddings.npy"
)

# Existing F32 HNSW M32 result set for the SAME 200 queries.
# The loader auto-detects the 2D integer labels array and, if present,
# a compatible floating-point scores array.
FP32_REFERENCE_NPZ = Path(
    r"C:\VAST2_UPLOAD\out\compare_fp32_sq8\20261029_103718\fp32_results.npz"
)

OUT_ROOT = Path(r"C:\VAST2_UPLOAD\out\compare_rabitq3_vs_fp32_m32")

NPROBES = [32, 64, 128, 256, 512, 1024]

INDEXES = {
    "RR768_IVF4096_RaBitQ3": Path(
        r"F:\RAG_checkpoints\ivf4096_rabitq3_cpu\ivf4096_rabitq3.index"
    ),
}


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_queries() -> np.ndarray:
    q = np.load(QUERY_EMB)
    if q.shape != (NQ, DIM):
        raise RuntimeError(f"Expected queries {(NQ, DIM)}, got {q.shape}")
    if q.dtype != np.float32:
        q = q.astype(np.float32)
    q = np.array(q, dtype=np.float32, order="C", copy=True)
    # Canonical query embeddings are already normalized. Do not renormalize.
    return q


def choose_reference_arrays(npz_path: Path):
    z = np.load(npz_path)

    int_candidates = []
    float_candidates = []

    for k in z.files:
        a = z[k]
        if a.ndim == 2 and a.shape[0] == NQ and a.shape[1] >= TOPK:
            if np.issubdtype(a.dtype, np.integer):
                int_candidates.append((k, a))
            elif np.issubdtype(a.dtype, np.floating):
                float_candidates.append((k, a))

    if not int_candidates:
        raise RuntimeError(
            "Could not find a 2D integer FP32 labels array "
            f"with shape ({NQ}, >= {TOPK}) in {npz_path}. "
            f"Keys: {[(k, z[k].shape, str(z[k].dtype)) for k in z.files]}"
        )

    # Prefer key names suggesting labels / ids / indices / neighbors.
    def int_score(item):
        k = item[0].lower()
        score = 0
        for token in ("label", "indice", "index", "id", "neighbor"):
            if token in k:
                score += 10
        if "top" in k:
            score += 2
        return score

    int_candidates.sort(key=int_score, reverse=True)
    labels_key, ref_I = int_candidates[0]
    ref_I = np.asarray(ref_I[:, :TOPK], dtype=np.int64)

    # Prefer a score/distance/similarity matrix with same or larger width.
    ref_D = None
    scores_key = None
    if float_candidates:
        def float_score(item):
            k = item[0].lower()
            score = 0
            for token in ("score", "dist", "sim"):
                if token in k:
                    score += 10
            if "top" in k:
                score += 2
            return score

        float_candidates.sort(key=float_score, reverse=True)
        scores_key, ref_D = float_candidates[0]
        ref_D = np.asarray(ref_D[:, :TOPK], dtype=np.float32)

    return ref_I, ref_D, labels_key, scores_key, [
        (k, tuple(z[k].shape), str(z[k].dtype)) for k in z.files
    ]


def overlap_at_k(ref_I: np.ndarray, test_I: np.ndarray, k: int) -> float:
    vals = []
    for q in range(ref_I.shape[0]):
        a = set(map(int, ref_I[q, :k]))
        b = set(map(int, test_I[q, :k]))
        vals.append(len(a & b) / k)
    return 100.0 * float(np.mean(vals))


def top1_exact(ref_I: np.ndarray, test_I: np.ndarray) -> float:
    return 100.0 * float(np.mean(ref_I[:, 0] == test_I[:, 0]))


def common_rank_delta(ref_I: np.ndarray, test_I: np.ndarray, k: int = 10):
    deltas = []
    common_counts = []
    for q in range(ref_I.shape[0]):
        ref_pos = {int(doc): rank for rank, doc in enumerate(ref_I[q, :k])}
        test_pos = {int(doc): rank for rank, doc in enumerate(test_I[q, :k])}
        common = set(ref_pos) & set(test_pos)
        common_counts.append(len(common))
        for doc in common:
            deltas.append(abs(ref_pos[doc] - test_pos[doc]))
    return {
        "rank_delta_common_top10_mean": float(np.mean(deltas)) if deltas else None,
        "rank_delta_common_top10_median": float(np.median(deltas)) if deltas else None,
        "common_docs_top10_mean": float(np.mean(common_counts)),
    }


def matched_similarity_mae(
    ref_I: np.ndarray,
    ref_D: np.ndarray | None,
    test_I: np.ndarray,
    test_D: np.ndarray,
    k: int = 500,
):
    if ref_D is None:
        return None, 0

    abs_err = []
    matched = 0

    for q in range(ref_I.shape[0]):
        ref_score = {
            int(doc): float(score)
            for doc, score in zip(ref_I[q, :k], ref_D[q, :k])
            if int(doc) >= 0
        }
        for doc, score in zip(test_I[q, :k], test_D[q, :k]):
            doc = int(doc)
            if doc in ref_score:
                abs_err.append(abs(float(score) - ref_score[doc]))
                matched += 1

    return (float(np.mean(abs_err)) if abs_err else None), matched


def percentile_ms(x, p):
    return float(np.percentile(np.asarray(x, dtype=np.float64), p))


def benchmark_one(index_name, index_path, q, ref_I, ref_D, run_dir):
    print("\n" + "=" * 100)
    print(f"INDEX: {index_name}")
    print(f"PATH : {index_path}")
    print("=" * 100)

    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index: {index_path}")

    t0 = time.perf_counter()
    index = faiss.read_index(str(index_path))
    load_s = time.perf_counter() - t0
    print(
        f"Loaded in {load_s:.3f}s | d={index.d} | ntotal={index.ntotal:,} "
        f"| trained={index.is_trained}"
    )

    if index.d != DIM:
        raise RuntimeError(f"{index_name}: d={index.d}, expected {DIM}")
    if index.ntotal != 4_304_468:
        raise RuntimeError(
            f"{index_name}: ntotal={index.ntotal:,}, expected 4,304,468"
        )

    ivf = faiss.extract_index_ivf(index)

    rows = []

    for nprobe in NPROBES:
        ivf.nprobe = int(nprobe)
        print(f"\n--- nprobe={nprobe} ---")

        # Warm-up on one query; do not include in latency.
        index.search(q[:1], TOPK)

        lat_ms = []
        I_all = np.empty((NQ, TOPK), dtype=np.int64)
        D_all = np.empty((NQ, TOPK), dtype=np.float32)

        search_t0 = time.perf_counter()
        for i in range(NQ):
            t1 = time.perf_counter()
            D, I = index.search(q[i:i+1], TOPK)
            lat_ms.append((time.perf_counter() - t1) * 1000.0)
            I_all[i] = I[0]
            D_all[i] = D[0]
        search_total_s = time.perf_counter() - search_t0

        metrics = {
            "index": index_name,
            "index_path": str(index_path),
            "nprobe": int(nprobe),
            "queries": NQ,
            "topk": TOPK,
            "top1_exact_pct": top1_exact(ref_I, I_all),
            "top5_overlap_pct": overlap_at_k(ref_I, I_all, 5),
            "top10_overlap_pct": overlap_at_k(ref_I, I_all, 10),
            "top50_overlap_pct": overlap_at_k(ref_I, I_all, 50),
            "top500_overlap_pct": overlap_at_k(ref_I, I_all, 500),
            "latency_mean_ms": float(np.mean(lat_ms)),
            "latency_median_ms": float(np.median(lat_ms)),
            "latency_p95_ms": percentile_ms(lat_ms, 95),
            "latency_p99_ms": percentile_ms(lat_ms, 99),
            "search_total_s": search_total_s,
            "qps": NQ / search_total_s,
            "index_load_s": load_s,
        }
        metrics.update(common_rank_delta(ref_I, I_all, 10))

        mae, matched = matched_similarity_mae(
            ref_I, ref_D, I_all, D_all, TOPK
        )
        metrics["matched_similarity_mae"] = mae
        metrics["matched_similarity_count"] = int(matched)

        npz_out = run_dir / f"{index_name}_nprobe{nprobe}.npz"
        np.savez_compressed(
            npz_out,
            labels=I_all,
            scores=D_all,
            latency_ms=np.asarray(lat_ms, dtype=np.float64),
        )

        json_out = run_dir / f"{index_name}_nprobe{nprobe}.json"
        json_out.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"Top1={metrics['top1_exact_pct']:.2f}% | "
            f"Top10={metrics['top10_overlap_pct']:.2f}% | "
            f"Top50={metrics['top50_overlap_pct']:.2f}% | "
            f"Top500={metrics['top500_overlap_pct']:.2f}% | "
            f"median={metrics['latency_median_ms']:.2f}ms | "
            f"p95={metrics['latency_p95_ms']:.2f}ms"
        )
        if mae is not None:
            print(
                f"matched similarity MAE={mae:.8f} "
                f"over {matched:,} matched results"
            )

        rows.append(metrics)

    return rows


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    faiss.omp_set_num_threads(OMP_THREADS)

    run_dir = OUT_ROOT / now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CPU PQ256 VS OPQ256 BENCHMARK VS F32 HNSW M32")
    print("=" * 100)
    print(f"FAISS threads : {OMP_THREADS}")
    print(f"Queries       : {QUERY_EMB}")
    print(f"F32 reference : {FP32_REFERENCE_NPZ}")
    print(f"Output        : {run_dir}")

    q = load_queries()
    ref_I, ref_D, labels_key, scores_key, ref_keys = choose_reference_arrays(
        FP32_REFERENCE_NPZ
    )

    print(f"\nReference labels key: {labels_key}")
    print(f"Reference scores key: {scores_key}")
    print(f"Reference labels     : {ref_I.shape}")
    print(f"Reference scores     : {None if ref_D is None else ref_D.shape}")

    reference_meta = {
        "query_embeddings": str(QUERY_EMB),
        "fp32_reference_npz": str(FP32_REFERENCE_NPZ),
        "reference_labels_key": labels_key,
        "reference_scores_key": scores_key,
        "reference_npz_keys": ref_keys,
        "queries": NQ,
        "dimension": DIM,
        "topk": TOPK,
        "nprobes": NPROBES,
        "omp_threads": OMP_THREADS,
        "comparison_definition": (
            "All overlap/Top1 metrics use the selected existing F32 HNSW M32 "
            "top-500 result matrix as the ranking reference."
        ),
    }
    (run_dir / "benchmark_meta.json").write_text(
        json.dumps(reference_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_rows = []
    for name, path in INDEXES.items():
        all_rows.extend(
            benchmark_one(name, path, q, ref_I, ref_D, run_dir)
        )

    summary_csv = run_dir / "summary.csv"
    write_csv(summary_csv, all_rows)

    # Compact Top10/latency table sorted by index then nprobe.
    compact_fields = [
        "index", "nprobe",
        "top1_exact_pct", "top10_overlap_pct", "top50_overlap_pct",
        "top500_overlap_pct",
        "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
        "qps", "matched_similarity_mae"
    ]
    compact_rows = [{k: r.get(k) for k in compact_fields} for r in all_rows]
    if compact_rows:
        with (run_dir / "summary_compact.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            w = csv.DictWriter(f, fieldnames=compact_fields)
            w.writeheader()
            w.writerows(compact_rows)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Summary: {summary_csv}")
    print(f"Run dir : {run_dir}")


if __name__ == "__main__":
    main()
