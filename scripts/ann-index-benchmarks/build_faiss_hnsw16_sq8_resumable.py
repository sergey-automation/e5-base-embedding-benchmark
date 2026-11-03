# -*- coding: utf-8 -*-
"""
Resumable FAISS HNSW16 + SQ8 builder.

- Existing FP32 embeddings are read-only.
- Existing FP32/SQ8 M32 indexes are never touched.
- Verified checkpoint is written every N completed source parts.
- Rerun the same command after interruption/reboot to continue.
- Final index is obtained from the last verified checkpoint without rebuilding.
"""

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import faiss
import numpy as np

PARTS_DIR = Path(r"C:\VAST2_UPLOAD\out\engineering_ru_rag_e5_base_torch\parts")
DEFAULT_WORK_DIR = Path(r"F:\RAG_checkpoints\chunk_index_sq8_m16")

DIM = 768
M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 64
TRAIN_MAX = 200_000
ADD_BATCH = 10_000
DEFAULT_CHECKPOINT_EVERY_PARTS = 2

FINAL_NAME = "faiss_hnsw16_sq8.index"
CHECKPOINT_NAME = "faiss_hnsw16_sq8.checkpoint.index"
CHECKPOINT_NEW_NAME = "faiss_hnsw16_sq8.checkpoint.new.index"
STATE_NAME = "faiss_hnsw16_sq8.state.json"
SUMMARY_NAME = "build_index_summary.json"


def normalize_rows(x):
    # Exact normalization used by the original SQ8 M32 builder.
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def get_parts():
    parts = sorted(PARTS_DIR.glob("engineering_ru_rag_emb_part_*.npy"))
    if not parts:
        raise RuntimeError(f"No embedding parts found in:\n{PARTS_DIR}")
    return parts


def scan_parts(parts):
    rows_per_part = []
    total_rows = 0
    print("Scanning source parts...")
    for i, p in enumerate(parts, 1):
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 2:
            raise RuntimeError(f"{p.name}: expected 2D, got {arr.shape}")
        if int(arr.shape[1]) != DIM:
            raise RuntimeError(f"{p.name}: dim={arr.shape[1]}, expected {DIM}")
        if arr.dtype != np.float32:
            raise RuntimeError(f"{p.name}: dtype={arr.dtype}, expected float32")
        n = int(arr.shape[0])
        rows_per_part.append(n)
        total_rows += n
        print(f"[{i:02d}/{len(parts):02d}] {p.name}: {n:,} rows")
        del arr
    return rows_per_part, total_rows


def atomic_write_json(path, obj):
    tmp = Path(str(path) + ".new")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_training_sample(parts, total_rows):
    # EXACTLY preserves the original SQ8 M32 training-sample method:
    # target_per_part = TRAIN_MAX // number_of_parts, evenly spread within each part.
    # With the current 66 parts: 200000 // 66 = 3030, total = 199980 vectors.
    target_per_part = max(1, TRAIN_MAX // len(parts))
    train_chunks = []

    print(
        f"\nBuilding SQ8 training sample using original M32 method: "
        f"{target_per_part:,} vectors/part × {len(parts)} parts"
    )

    for p in parts:
        arr = np.load(p, mmap_mode="r")
        n = int(arr.shape[0])
        take = min(target_per_part, n)

        if take == n:
            sample = np.asarray(arr, dtype=np.float32)
        else:
            idx = np.linspace(0, n - 1, num=take, dtype=np.int64)
            sample = np.asarray(arr[idx], dtype=np.float32)

        sample = normalize_rows(sample)
        train_chunks.append(sample)
        del arr

    train = np.concatenate(train_chunks, axis=0)
    if train.shape[0] > TRAIN_MAX:
        train = train[:TRAIN_MAX]

    actual_train = int(train.shape[0])
    print(f"Training vectors: {actual_train:,}")
    print("Training shape  :", train.shape)
    print("Training dtype  :", train.dtype)

    return train, actual_train


def create_new_index(parts, total_rows):
    sample, training_vectors = build_training_sample(parts, total_rows)
    print("\nCreating new FAISS IndexHNSWSQ(QT_8bit)...")
    index = faiss.IndexHNSWSQ(
        DIM,
        faiss.ScalarQuantizer.QT_8bit,
        M,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efConstruction = int(EF_CONSTRUCTION)
    index.hnsw.efSearch = int(EF_SEARCH)
    print("Training SQ8...")
    t0 = time.perf_counter()
    index.train(sample)
    train_sec = time.perf_counter() - t0
    del sample
    gc.collect()
    if not index.is_trained:
        raise RuntimeError("SQ8 index training failed")
    print(f"Training completed in {train_sec:.3f} s")
    return index, train_sec, training_vectors


def expected_ntotal(rows_per_part, completed_parts):
    return int(sum(rows_per_part[:completed_parts]))


def validate_state(state, parts, rows_per_part, total_rows):
    if state.get("part_names") != [p.name for p in parts]:
        raise RuntimeError("Source part list differs from saved checkpoint state")
    if state.get("rows_per_part") != rows_per_part:
        raise RuntimeError("Source part row counts differ from saved checkpoint state")
    checks = {
        "dimension": DIM,
        "hnsw_M": M,
        "efConstruction": EF_CONSTRUCTION,
        "quantizer": "ScalarQuantizer.QT_8bit",
        "total_vectors": total_rows,
    }
    for key, expected in checks.items():
        if state.get(key) != expected:
            raise RuntimeError(
                f"Checkpoint parameter mismatch: {key}={state.get(key)!r}, expected {expected!r}"
            )


def load_checkpoint(checkpoint_path, state_path, parts, rows_per_part, total_rows):
    if not checkpoint_path.exists() and not state_path.exists():
        return None
    if checkpoint_path.exists() != state_path.exists():
        raise RuntimeError(
            "Checkpoint/state pair is incomplete.\n"
            f"Checkpoint: {checkpoint_path.exists()}\n"
            f"State: {state_path.exists()}"
        )
    print("\nFound verified checkpoint:")
    print(checkpoint_path)
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    validate_state(state, parts, rows_per_part, total_rows)
    completed_parts = int(state["completed_parts"])
    expected_count = expected_ntotal(rows_per_part, completed_parts)
    print(
        f"Checkpoint state: completed_parts={completed_parts}/{len(parts)}, "
        f"expected_ntotal={expected_count:,}"
    )
    t0 = time.perf_counter()
    index = faiss.read_index(str(checkpoint_path))
    load_sec = time.perf_counter() - t0
    if int(index.d) != DIM:
        raise RuntimeError(f"Checkpoint dim={index.d}, expected {DIM}")
    if int(index.ntotal) != expected_count:
        raise RuntimeError(
            f"Checkpoint ntotal={index.ntotal:,}, state expects {expected_count:,}"
        )
    if not index.is_trained:
        raise RuntimeError("Checkpoint index is not trained")
    index.hnsw.efConstruction = int(EF_CONSTRUCTION)
    index.hnsw.efSearch = int(EF_SEARCH)
    print(f"Checkpoint loaded in {load_sec:.2f} s; ntotal={index.ntotal:,}")
    return index, state, load_sec


def free_space_bytes(path):
    return shutil.disk_usage(path).free


def checkpoint_index(index, work_dir, checkpoint_path, checkpoint_new_path, state_path, state):
    print("\n" + "=" * 78)
    print(f"CHECKPOINT after part {state['completed_parts']}/{state['part_count']}")
    print("=" * 78)
    free_before = free_space_bytes(work_dir)
    old_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
    print(f"Free before checkpoint : {free_before / 2**30:.3f} GiB")
    if old_size:
        print(f"Previous checkpoint    : {old_size / 2**30:.3f} GiB")
    if checkpoint_new_path.exists():
        print("Removing stale unverified .new checkpoint...")
        checkpoint_new_path.unlink()

    # Rough safety check. During atomic checkpointing the old and new files coexist.
    # If there is an old checkpoint, the new one will be of similar or larger size.
    if old_size and free_before < int(old_size * 1.25):
        raise RuntimeError(
            "Not enough free space for a safe atomic checkpoint.\n"
            f"Free: {free_before / 2**30:.3f} GiB\n"
            f"Old checkpoint: {old_size / 2**30:.3f} GiB\n"
            "Free more space or use another --work-dir. The existing verified checkpoint remains intact."
        )

    t0 = time.perf_counter()
    faiss.write_index(index, str(checkpoint_new_path))
    write_sec = time.perf_counter() - t0
    new_size = checkpoint_new_path.stat().st_size
    print(f"New checkpoint written : {new_size / 2**30:.3f} GiB in {write_sec:.1f} s")

    print("Verifying new checkpoint by reopening it...")
    tv = time.perf_counter()
    verify = faiss.read_index(str(checkpoint_new_path))
    verify_sec = time.perf_counter() - tv
    if int(verify.d) != DIM:
        raise RuntimeError(f"New checkpoint verification failed: d={verify.d}")
    if int(verify.ntotal) != int(index.ntotal):
        raise RuntimeError(
            f"New checkpoint verification failed: ntotal={verify.ntotal:,}, expected {index.ntotal:,}"
        )
    if not verify.is_trained:
        raise RuntimeError("New checkpoint verification failed: not trained")
    del verify
    gc.collect()

    # Old verified checkpoint remains untouched until this atomic promotion.
    os.replace(checkpoint_new_path, checkpoint_path)

    state["checkpoint_size_bytes"] = int(new_size)
    state["checkpoint_write_seconds_last"] = float(write_sec)
    state["checkpoint_verify_seconds_last"] = float(verify_sec)
    state["checkpoint_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["ntotal"] = int(index.ntotal)
    atomic_write_json(state_path, state)
    print(f"Checkpoint VERIFIED: ntotal={index.ntotal:,}; verify={verify_sec:.1f} s")
    print("=" * 78)


def add_part(index, part_path, total_rows, already_added):
    arr = np.load(part_path, mmap_mode="r")
    n = int(arr.shape[0])
    part_t0 = time.perf_counter()
    added = already_added
    for start in range(0, n, ADD_BATCH):
        end = min(start + ADD_BATCH, n)
        # Exact vector preparation used by the original SQ8 M32 builder.
        batch = np.asarray(arr[start:end], dtype=np.float32)
        batch = normalize_rows(batch)
        tb = time.perf_counter()
        index.add(batch)
        batch_sec = time.perf_counter() - tb
        added += len(batch)
        print(
            f"  {start:7d}:{end:7d}  added={added:,}/{total_rows:,}  batch={batch_sec:.2f}s"
        )
        del batch
    del arr
    gc.collect()
    return added, time.perf_counter() - part_t0


def finalize(index, checkpoint_path, final_path, summary_path, state_path, state, current_run_sec):
    if int(index.ntotal) != int(state["total_vectors"]):
        raise RuntimeError(
            f"Cannot finalize: ntotal={index.ntotal:,}, expected {state['total_vectors']:,}"
        )
    if not checkpoint_path.exists():
        raise RuntimeError("Final verified checkpoint does not exist")
    if final_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing final index:\n{final_path}")

    # Rename, do not rewrite multi-GiB data.
    os.replace(checkpoint_path, final_path)
    final_size = final_path.stat().st_size

    summary = {
        "mode": "faiss_hnsw16_sq8",
        "type": "FAISS IndexHNSWSQ",
        "quantizer": "ScalarQuantizer.QT_8bit",
        "dimension": DIM,
        "hnsw_M": M,
        "efConstruction": EF_CONSTRUCTION,
        "default_efSearch": EF_SEARCH,
        "metric": "METRIC_INNER_PRODUCT",
        "input_vectors_normalized": True,
        "source_parts_dir": str(PARTS_DIR),
        "part_count": state["part_count"],
        "rows_per_part": state["rows_per_part"],
        "total_vectors": state["total_vectors"],
        "train_max": TRAIN_MAX,
        "training_vectors": int(state.get("training_vectors", 0)),
        "add_batch": ADD_BATCH,
        "checkpoint_every_parts": state["checkpoint_every_parts"],
        "index_path": str(final_path),
        "index_size_bytes": int(final_size),
        "index_size_gib": final_size / (1024 ** 3),
        "accumulated_add_seconds": state.get("accumulated_add_seconds", 0.0),
        "current_run_seconds": current_run_sec,
        "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(summary_path, summary)
    state["finalized"] = True
    state["final_index_path"] = str(final_path)
    state["final_index_size_bytes"] = int(final_size)
    atomic_write_json(state_path, state)

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("Vectors :", f"{index.ntotal:,}")
    print("Index   :", final_path)
    print("Size    :", f"{final_size / 2**30:.3f} GiB")
    print("Summary :", summary_path)
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="Directory for checkpoints and final SQ8 M16 index",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_PARTS,
        help="Save a verified checkpoint every N completed source parts",
    )
    args = parser.parse_args()

    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = work_dir / CHECKPOINT_NAME
    checkpoint_new_path = work_dir / CHECKPOINT_NEW_NAME
    state_path = work_dir / STATE_NAME
    final_path = work_dir / FINAL_NAME
    summary_path = work_dir / SUMMARY_NAME

    if not PARTS_DIR.exists():
        raise SystemExit(f"Source directory not found:\n{PARTS_DIR}")
    if final_path.exists():
        raise SystemExit(f"Final SQ8 M16 index already exists; nothing to resume:\n{final_path}")

    print("=" * 78)
    print("RESUMABLE FAISS HNSW16 + SQ8 BUILDER")
    print("=" * 78)
    print("Source          :", PARTS_DIR)
    print("Work dir        :", work_dir)
    print("Checkpoint every:", args.checkpoint_every, "part(s)")
    print("DIM             :", DIM)
    print("M               :", M)
    print("efConstruction  :", EF_CONSTRUCTION)
    print("SQ              : ScalarQuantizer.QT_8bit")
    print("Metric          : inner product on normalized vectors")
    print("=" * 78)
    print(f"Free space in work dir: {free_space_bytes(work_dir) / 2**30:.3f} GiB")

    parts = get_parts()
    rows_per_part, total_rows = scan_parts(parts)

    resumed = load_checkpoint(
        checkpoint_path, state_path, parts, rows_per_part, total_rows
    )
    run_t0 = time.perf_counter()

    if resumed is None:
        index, train_sec, training_vectors = create_new_index(parts, total_rows)
        state = {
            "version": 1,
            "quantizer": "ScalarQuantizer.QT_8bit",
            "dimension": DIM,
            "hnsw_M": M,
            "efConstruction": EF_CONSTRUCTION,
            "default_efSearch": EF_SEARCH,
            "metric": "METRIC_INNER_PRODUCT",
            "input_vectors_normalized": True,
            "part_count": len(parts),
            "part_names": [p.name for p in parts],
            "rows_per_part": rows_per_part,
            "total_vectors": total_rows,
            "completed_parts": 0,
            "ntotal": 0,
            "checkpoint_every_parts": args.checkpoint_every,
            "training_vectors": int(training_vectors),
            "training_seconds": float(train_sec),
            "accumulated_add_seconds": 0.0,
            "finalized": False,
        }
        start_part = 0
        print("\nStarting a NEW build.")
    else:
        index, state, _load_sec = resumed
        start_part = int(state["completed_parts"])
        state["checkpoint_every_parts"] = args.checkpoint_every
        print(
            f"\nRESUMING from part {start_part + 1}/{len(parts)} "
            f"(already completed {start_part})"
        )

    added = int(index.ntotal)

    for i in range(start_part, len(parts)):
        p = parts[i]
        print(f"\nPART {i + 1}/{len(parts)}: {p.name} ({rows_per_part[i]:,} vectors)")
        added, part_sec = add_part(index, p, total_rows, added)
        expected_after = expected_ntotal(rows_per_part, i + 1)
        if int(index.ntotal) != expected_after:
            raise RuntimeError(
                f"After {p.name}: index.ntotal={index.ntotal:,}, expected {expected_after:,}"
            )
        if added != expected_after:
            raise RuntimeError(f"Added counter={added:,}, expected {expected_after:,}")

        state["completed_parts"] = i + 1
        state["ntotal"] = int(index.ntotal)
        state["accumulated_add_seconds"] = float(
            state.get("accumulated_add_seconds", 0.0) + part_sec
        )

        if ((i + 1) % args.checkpoint_every == 0) or ((i + 1) == len(parts)):
            checkpoint_index(
                index,
                work_dir,
                checkpoint_path,
                checkpoint_new_path,
                state_path,
                state,
            )

    if int(index.ntotal) == total_rows and not checkpoint_path.exists():
        checkpoint_index(
            index,
            work_dir,
            checkpoint_path,
            checkpoint_new_path,
            state_path,
            state,
        )

    finalize(
        index,
        checkpoint_path,
        final_path,
        summary_path,
        state_path,
        state,
        time.perf_counter() - run_t0,
    )


if __name__ == "__main__":
    main()
