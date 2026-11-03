# build_opq384_ivf4096_pq384_cpu.py
# CPU-only FAISS build for:
# OPQ384_768 -> IVF4096 -> PQ384x8
#
# Dataset: 4,304,468 normalized 768-D E5 vectors in 66 .npy parts.
# Training sample: 199,980 vectors (3030 uniformly sampled from each part),
# matching the earlier SQ8 training-sample method.
#
# Safe/resumable:
# - saves trained empty index after train()
# - checkpoints after every CHECKPOINT_EVERY_PARTS source parts
# - verifies every checkpoint by reopening and checking ntotal
# - resumes from the last verified checkpoint

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from pathlib import Path

import faiss
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DIM = 768
N_EXPECTED = 4_304_468

NLIST = 4096
PQ_M = 384
PQ_BITS = 8

TRAIN_MAX = 200_000
ADD_BATCH = 10_000
CHECKPOINT_EVERY_PARTS = 4
OMP_THREADS = 4

PARTS_DIR = Path(
    r"C:\VAST2_UPLOAD\out\engineering_ru_rag_e5_base_torch\parts"
)

WORK_DIR = Path(
    r"F:\RAG_checkpoints\opq384_ivf4096_pq384_cpu"
)

TRAINED_INDEX = WORK_DIR / "opq384_ivf4096_pq384.trained_empty.index"
CHECKPOINT_INDEX = WORK_DIR / "opq384_ivf4096_pq384.checkpoint.index"
CHECKPOINT_TMP = WORK_DIR / "opq384_ivf4096_pq384.checkpoint.tmp.index"
STATE_JSON = WORK_DIR / "opq384_ivf4096_pq384.state.json"

FINAL_INDEX = WORK_DIR / "opq384_ivf4096_pq384.index"
SUMMARY_JSON = WORK_DIR / "build_summary.json"

FACTORY = "OPQ384_768,IVF4096,PQ384"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def gib(n: int) -> float:
    return n / (1024 ** 3)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32, order="C", copy=True)
    faiss.normalize_L2(x)
    return x


def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def get_parts() -> list[Path]:
    parts = sorted(PARTS_DIR.glob("engineering_ru_rag_emb_part_*.npy"))
    if not parts:
        raise FileNotFoundError(f"No embedding parts found in {PARTS_DIR}")
    return parts


def inspect_parts(parts: list[Path]) -> tuple[list[int], int]:
    rows = []
    total = 0
    print("=" * 86)
    print("SOURCE PARTS")
    print("=" * 86)
    for i, p in enumerate(parts, 1):
        a = np.load(p, mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != DIM:
            raise ValueError(f"{p}: unexpected shape {a.shape}")
        if a.dtype != np.float32:
            raise TypeError(f"{p}: expected float32, got {a.dtype}")
        n = int(a.shape[0])
        rows.append(n)
        total += n
        print(f"[{i:02d}/{len(parts):02d}] {p.name}: {n:,} rows")
        del a
    print(f"\nTotal: {total:,}")
    if total != N_EXPECTED:
        raise ValueError(f"Expected {N_EXPECTED:,}, got {total:,}")
    return rows, total


def build_training_sample(parts: list[Path]) -> np.ndarray:
    target_per_part = max(1, TRAIN_MAX // len(parts))
    # For 66 parts and TRAIN_MAX=200000 this is 3030 -> 199,980 total.
    print("\n" + "=" * 86)
    print("TRAINING SAMPLE")
    print("=" * 86)
    print(f"TRAIN_MAX       : {TRAIN_MAX:,}")
    print(f"parts           : {len(parts)}")
    print(f"target/part     : {target_per_part:,}")
    print(f"expected sample : {target_per_part * len(parts):,}")

    chunks = []
    for i, p in enumerate(parts, 1):
        a = np.load(p, mmap_mode="r")
        n = int(a.shape[0])
        take = min(target_per_part, n)
        idx = np.linspace(0, n - 1, num=take, dtype=np.int64)
        x = np.asarray(a[idx], dtype=np.float32)
        chunks.append(x)
        print(f"[{i:02d}/{len(parts):02d}] {p.name}: sampled {take:,}")
        del a

    train = np.ascontiguousarray(np.vstack(chunks), dtype=np.float32)
    normalize_rows(train)
    print(f"Training matrix : {train.shape}, {train.dtype}")
    return train


def create_index():
    print("\n" + "=" * 86)
    print("CREATE INDEX")
    print("=" * 86)
    print(f"Factory : {FACTORY}")
    print(f"Metric  : InnerProduct")
    index = faiss.index_factory(DIM, FACTORY, faiss.METRIC_INNER_PRODUCT)

    # Make OPQ training progress visible.
    # IndexPreTransform.chain[0] is the OPQMatrix for this factory string.
    try:
        opq = faiss.downcast_VectorTransform(index.chain.at(0))
        opq.verbose = True
        print(f"OPQ niter            : {opq.niter}")
        print(f"OPQ niter_pq         : {opq.niter_pq}")
        print(f"OPQ niter_pq_0       : {opq.niter_pq_0}")
        print(f"OPQ max_train_points : {opq.max_train_points:,}")
    except Exception as e:
        print(f"Warning: could not expose OPQ settings: {e}")

    # Make IVF/PQ training progress visible when possible.
    try:
        ivf = faiss.extract_index_ivf(index)
        ivf.verbose = True
        ivf.cp.verbose = True
        ivf.pq.verbose = True
        print(f"IVF nlist            : {ivf.nlist}")
        print(f"PQ M                 : {ivf.pq.M}")
        print(f"PQ nbits             : {ivf.pq.nbits}")
        print(f"PQ code_size         : {ivf.code_size} B/vector")
    except Exception as e:
        print(f"Warning: could not expose IVF/PQ settings: {e}")

    return index


def save_verified_checkpoint(index, completed_parts: int, expected_ntotal: int) -> None:
    print("\n" + "=" * 86)
    print(f"CHECKPOINT after part {completed_parts}")
    print("=" * 86)

    if CHECKPOINT_TMP.exists():
        CHECKPOINT_TMP.unlink()

    t0 = time.perf_counter()
    faiss.write_index(index, str(CHECKPOINT_TMP))
    write_s = time.perf_counter() - t0
    size_b = CHECKPOINT_TMP.stat().st_size
    print(f"New checkpoint written : {gib(size_b):.3f} GiB in {write_s:.1f} s")

    print("Verifying by reopening...")
    t0 = time.perf_counter()
    verify = faiss.read_index(str(CHECKPOINT_TMP))
    verify_s = time.perf_counter() - t0
    if int(verify.ntotal) != int(expected_ntotal):
        raise RuntimeError(
            f"Checkpoint ntotal mismatch: {verify.ntotal:,} != {expected_ntotal:,}"
        )
    del verify
    gc.collect()
    print(f"Checkpoint VERIFIED: ntotal={expected_ntotal:,}; verify={verify_s:.1f} s")

    os.replace(CHECKPOINT_TMP, CHECKPOINT_INDEX)
    write_json_atomic(
        STATE_JSON,
        {
            "completed_parts": int(completed_parts),
            "expected_ntotal": int(expected_ntotal),
            "factory": FACTORY,
            "metric": "inner_product",
            "dim": DIM,
        },
    )


def load_resume(parts: list[Path]):
    if CHECKPOINT_INDEX.is_file() and STATE_JSON.is_file():
        state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
        completed_parts = int(state["completed_parts"])
        expected_ntotal = int(state["expected_ntotal"])

        print("\nFound checkpoint:")
        print(CHECKPOINT_INDEX)
        print(
            f"Checkpoint state: completed_parts={completed_parts}/{len(parts)}, "
            f"expected_ntotal={expected_ntotal:,}"
        )

        t0 = time.perf_counter()
        index = faiss.read_index(str(CHECKPOINT_INDEX))
        dt = time.perf_counter() - t0
        if int(index.ntotal) != expected_ntotal:
            raise RuntimeError(
                f"Checkpoint ntotal mismatch: {index.ntotal:,} != {expected_ntotal:,}"
            )
        print(f"Checkpoint loaded in {dt:.2f} s; ntotal={index.ntotal:,}")
        return index, completed_parts

    if TRAINED_INDEX.is_file():
        print("\nFound trained empty index:")
        print(TRAINED_INDEX)
        t0 = time.perf_counter()
        index = faiss.read_index(str(TRAINED_INDEX))
        dt = time.perf_counter() - t0
        if not index.is_trained or int(index.ntotal) != 0:
            raise RuntimeError("Saved trained index is invalid")
        print(f"Trained index loaded in {dt:.2f} s")
        return index, 0

    return None, 0


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    faiss.omp_set_num_threads(OMP_THREADS)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print("CPU FAISS BUILD: OPQ384_768,IVF4096,PQ384")
    print("=" * 86)
    print(f"FAISS version : {faiss.__version__}")
    print(f"OMP threads   : {OMP_THREADS}")
    print(f"Parts         : {PARTS_DIR}")
    print(f"Output        : {FINAL_INDEX}")
    print(f"Factory       : {FACTORY}")
    print(f"Expected N    : {N_EXPECTED:,}")
    print(f"Dimension     : {DIM}")

    parts = get_parts()
    rows_per_part, total = inspect_parts(parts)

    # If final index already exists, verify and stop.
    if FINAL_INDEX.is_file():
        print("\nFinal index already exists. Verifying...")
        idx = faiss.read_index(str(FINAL_INDEX))
        print(f"is_trained={idx.is_trained}; ntotal={idx.ntotal:,}")
        if idx.is_trained and int(idx.ntotal) == N_EXPECTED:
            print("FINAL INDEX VERIFIED. Nothing to do.")
            return 0
        raise RuntimeError("Existing final index is incomplete/invalid")

    index, completed_parts = load_resume(parts)

    # Train once, then save trained-empty index.
    train_seconds = None
    if index is None:
        index = create_index()
        train = build_training_sample(parts)

        print("\n" + "=" * 86)
        print("TRAIN")
        print("=" * 86)
        print("Training can be the longest non-resumable stage.")
        t0 = time.perf_counter()
        index.train(train)
        train_seconds = time.perf_counter() - t0
        print(f"TRAIN DONE: {train_seconds:.1f} s = {train_seconds/60:.2f} min")

        if not index.is_trained:
            raise RuntimeError("Index reports is_trained=False after train()")
        if int(index.ntotal) != 0:
            raise RuntimeError(f"Unexpected ntotal after train: {index.ntotal}")

        print("Saving trained empty index...")
        faiss.write_index(index, str(TRAINED_INDEX))
        verify = faiss.read_index(str(TRAINED_INDEX))
        if not verify.is_trained or int(verify.ntotal) != 0:
            raise RuntimeError("Trained empty index verification failed")
        del verify
        del train
        gc.collect()
        print(f"Trained empty index saved: {TRAINED_INDEX}")

    if completed_parts:
        print(
            f"\nRESUMING from part {completed_parts + 1}/{len(parts)} "
            f"(already completed {completed_parts})"
        )

    add_start = time.perf_counter()
    added_this_run = 0

    for part_idx in range(completed_parts, len(parts)):
        p = parts[part_idx]
        a = np.load(p, mmap_mode="r")
        n = int(a.shape[0])

        print("\n" + "=" * 86)
        print(f"PART {part_idx + 1}/{len(parts)}: {p.name} ({n:,} vectors)")
        print("=" * 86)

        for start in range(0, n, ADD_BATCH):
            end = min(start + ADD_BATCH, n)
            x = np.asarray(a[start:end], dtype=np.float32)
            x = normalize_rows(x)

            t0 = time.perf_counter()
            index.add(x)
            dt = time.perf_counter() - t0

            added_this_run += end - start
            print(
                f"{start:9d}:{end:9d}  "
                f"added={index.ntotal:,}/{N_EXPECTED:,}  batch={dt:.2f}s"
            )

            del x

        del a
        gc.collect()

        completed = part_idx + 1
        if (
            completed % CHECKPOINT_EVERY_PARTS == 0
            or completed == len(parts)
        ):
            expected_ntotal = sum(rows_per_part[:completed])
            if int(index.ntotal) != expected_ntotal:
                raise RuntimeError(
                    f"Before checkpoint ntotal mismatch: "
                    f"{index.ntotal:,} != {expected_ntotal:,}"
                )
            save_verified_checkpoint(index, completed, expected_ntotal)

    add_seconds_this_run = time.perf_counter() - add_start

    if int(index.ntotal) != N_EXPECTED:
        raise RuntimeError(f"Final ntotal mismatch: {index.ntotal:,}")

    print("\n" + "=" * 86)
    print("FINAL SERIALIZATION")
    print("=" * 86)

    final_tmp = FINAL_INDEX.with_suffix(".tmp.index")
    if final_tmp.exists():
        final_tmp.unlink()

    t0 = time.perf_counter()
    faiss.write_index(index, str(final_tmp))
    final_write_seconds = time.perf_counter() - t0

    print("Reopening final index for verification...")
    t0 = time.perf_counter()
    verify = faiss.read_index(str(final_tmp))
    final_verify_seconds = time.perf_counter() - t0
    if not verify.is_trained or int(verify.ntotal) != N_EXPECTED:
        raise RuntimeError(
            f"Final verification failed: trained={verify.is_trained}, "
            f"ntotal={verify.ntotal:,}"
        )
    del verify
    gc.collect()

    os.replace(final_tmp, FINAL_INDEX)
    size_b = FINAL_INDEX.stat().st_size

    summary = {
        "factory": FACTORY,
        "metric": "inner_product",
        "vectors": N_EXPECTED,
        "dimension": DIM,
        "nlist": NLIST,
        "pq_m": PQ_M,
        "pq_bits": PQ_BITS,
        "train_max": TRAIN_MAX,
        "actual_training_vectors": (TRAIN_MAX // len(parts)) * len(parts),
        "add_batch": ADD_BATCH,
        "omp_threads": OMP_THREADS,
        "train_seconds_this_run": train_seconds,
        "add_seconds_this_run": add_seconds_this_run,
        "added_this_run": added_this_run,
        "index_size_bytes": size_b,
        "index_size_gib": gib(size_b),
        "final_write_seconds": final_write_seconds,
        "final_verify_seconds": final_verify_seconds,
        "final_index": str(FINAL_INDEX),
    }
    write_json_atomic(SUMMARY_JSON, summary)

    print("\n" + "=" * 86)
    print("DONE")
    print("=" * 86)
    print(f"Vectors : {N_EXPECTED:,}")
    print(f"Index   : {FINAL_INDEX}")
    print(f"Size    : {gib(size_b):.3f} GiB")
    print(f"Summary : {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
