import glob
import json
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    raise SystemExit(
        "FAISS is not installed.\n"
        "Install with:\n"
        "py -m pip install faiss-cpu"
    )


# ---------------------------------------------------------------------
# INPUT: READ ONLY
# ---------------------------------------------------------------------

PARTS_DIR = r"C:\VAST2_UPLOAD\out\engineering_ru_rag_e5_base_torch\parts"

PART_GLOB = os.path.join(
    PARTS_DIR,
    "engineering_ru_rag_emb_part_*.npy",
)

# ---------------------------------------------------------------------
# OUTPUT: NEW DIRECTORY ONLY
# ---------------------------------------------------------------------

OUT_DIR = r"C:\VAST2_UPLOAD\out\chunk_index_sq8"

INDEX_PATH = os.path.join(
    OUT_DIR,
    "faiss_hnsw32_sq8.index",
)

SUMMARY_PATH = os.path.join(
    OUT_DIR,
    "build_index_summary.json",
)

TMP_INDEX_PATH = INDEX_PATH + ".tmp"

# ---------------------------------------------------------------------

DIM = 768
M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64

# Number of vectors used to train the SQ8 quantizer.
TRAIN_MAX = 200_000

# Add vectors to FAISS in chunks to keep temporary RAM bounded.
ADD_BATCH = 10_000


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def main():
    print("=" * 72)
    print("FAISS HNSW32 + SQ8 BUILD")
    print("=" * 72)
    print("SOURCE PARTS :", PARTS_DIR)
    print("OUTPUT INDEX :", INDEX_PATH)
    print("DIM          :", DIM)
    print("M            :", M)
    print("efConstruction:", EF_CONSTRUCTION)
    print("efSearch     :", EF_SEARCH)
    print("TRAIN_MAX    :", TRAIN_MAX)
    print("=" * 72)

    # --------------------------------------------------------------
    # Safety checks: NEVER overwrite anything
    # --------------------------------------------------------------

    if not os.path.isdir(PARTS_DIR):
        raise SystemExit(f"Parts directory not found:\n{PARTS_DIR}")

    parts = sorted(glob.glob(PART_GLOB))

    if not parts:
        raise SystemExit(
            "No embedding .npy parts found:\n"
            + PART_GLOB
        )

    print(f"Embedding parts found: {len(parts)}")
    print("First:", parts[0])
    print("Last :", parts[-1])

    os.makedirs(OUT_DIR, exist_ok=True)

    for p in (INDEX_PATH, SUMMARY_PATH, TMP_INDEX_PATH):
        if os.path.exists(p):
            raise SystemExit(
                "\nSTOP: output already exists; nothing was overwritten:\n"
                + p
            )

    # --------------------------------------------------------------
    # Inspect all parts
    # --------------------------------------------------------------

    total_vectors = 0
    part_info = []

    print("\nInspecting .npy parts...")

    for i, path in enumerate(parts):
        arr = np.load(path, mmap_mode="r")

        if arr.ndim != 2:
            raise RuntimeError(
                f"Unexpected shape in {path}: {arr.shape}"
            )

        if arr.shape[1] != DIM:
            raise RuntimeError(
                f"Wrong embedding dimension in {path}: "
                f"{arr.shape[1]} != {DIM}"
            )

        n = int(arr.shape[0])
        total_vectors += n

        part_info.append({
            "path": path,
            "rows": n,
            "dtype": str(arr.dtype),
        })

        print(
            f"[{i + 1:03d}/{len(parts):03d}] "
            f"rows={n:8d} dtype={arr.dtype} "
            f"total={total_vectors}"
        )

    print("\nTotal vectors:", total_vectors)

    # --------------------------------------------------------------
    # Build representative training sample for SQ8
    # --------------------------------------------------------------

    print("\nPreparing SQ8 training sample...")

    target_per_part = max(
        1,
        TRAIN_MAX // len(parts)
    )

    train_chunks = []

    for path in parts:
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]

        take = min(target_per_part, n)

        if take == n:
            sample = np.asarray(arr, dtype=np.float32)
        else:
            # evenly spread sample over the whole part
            idx = np.linspace(
                0,
                n - 1,
                num=take,
                dtype=np.int64,
            )
            sample = np.asarray(arr[idx], dtype=np.float32)

        sample = normalize_rows(sample)
        train_chunks.append(sample)

    train = np.concatenate(train_chunks, axis=0)

    if train.shape[0] > TRAIN_MAX:
        train = train[:TRAIN_MAX]

    print("Training vectors:", train.shape[0])
    print("Training shape  :", train.shape)
    print("Training dtype  :", train.dtype)

    # --------------------------------------------------------------
    # Create HNSW32 + SQ8
    #
    # Existing embeddings are normalized for cosine search.
    # Cosine similarity == inner product for normalized vectors.
    # --------------------------------------------------------------

    print("\nCreating FAISS IndexHNSWSQ...")

    index = faiss.IndexHNSWSQ(
        DIM,
        faiss.ScalarQuantizer.QT_8bit,
        M,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH

    # --------------------------------------------------------------
    # Train SQ8
    # --------------------------------------------------------------

    print("Training SQ8 quantizer...")
    t_train = time.time()

    index.train(train)

    train_sec = time.time() - t_train

    print(f"SQ8 training done in {train_sec:.1f} s")
    print("is_trained:", index.is_trained)

    del train
    del train_chunks

    # --------------------------------------------------------------
    # Add all vectors, preserving original sequential order
    # --------------------------------------------------------------

    print("\nAdding vectors...")

    t0 = time.time()
    added = 0

    for part_no, path in enumerate(parts):
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]

        print(
            f"\nPART {part_no + 1}/{len(parts)} "
            f"{os.path.basename(path)} "
            f"rows={n}"
        )

        for start in range(0, n, ADD_BATCH):
            end = min(start + ADD_BATCH, n)

            x = np.asarray(
                arr[start:end],
                dtype=np.float32,
            )

            x = normalize_rows(x)

            index.add(x)

            added += x.shape[0]

            elapsed = time.time() - t0
            speed = added / max(elapsed, 1e-9)

            eta = (
                (total_vectors - added) / speed
                if speed > 0
                else 0
            )

            print(
                f"added={added}/{total_vectors} "
                f"speed={speed:.1f} vec/s "
                f"ETA={eta/60:.1f} min",
                flush=True,
            )

    if added != total_vectors:
        raise RuntimeError(
            f"Added count mismatch: {added} != {total_vectors}"
        )

    if index.ntotal != total_vectors:
        raise RuntimeError(
            f"FAISS ntotal mismatch: "
            f"{index.ntotal} != {total_vectors}"
        )

    build_sec = time.time() - t0

    # --------------------------------------------------------------
    # Save to TEMP first
    # --------------------------------------------------------------

    print("\nWriting temporary index:")
    print(TMP_INDEX_PATH)

    faiss.write_index(index, TMP_INDEX_PATH)

    if not os.path.exists(TMP_INDEX_PATH):
        raise RuntimeError("Temporary index was not created")

    tmp_size = os.path.getsize(TMP_INDEX_PATH)

    print(
        f"Temporary index size: "
        f"{tmp_size / (1024 ** 3):.3f} GiB"
    )

    # Atomic move to final name.
    # Destination was checked above and does not exist.
    os.replace(TMP_INDEX_PATH, INDEX_PATH)

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary = {
        "mode": "faiss_hnsw32_sq8",
        "source_parts_dir": PARTS_DIR,
        "source_parts_count": len(parts),
        "indexed_vectors": total_vectors,
        "embedding_dim": DIM,
        "metric": "inner_product_on_normalized_vectors",
        "encoding": "SQ8",
        "faiss_class": "IndexHNSWSQ",
        "hnsw_M": M,
        "hnsw_ef_construction": EF_CONSTRUCTION,
        "hnsw_ef_search": EF_SEARCH,
        "sq_type": "QT_8bit",
        "training_vectors": min(
            TRAIN_MAX,
            target_per_part * len(parts),
        ),
        "index_path": INDEX_PATH,
        "index_size_bytes": os.path.getsize(INDEX_PATH),
        "train_time_sec": train_sec,
        "build_time_sec": build_sec,
        "parts": part_info,
    }

    with open(
        SUMMARY_PATH,
        "x",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print("Vectors :", index.ntotal)
    print("Index   :", INDEX_PATH)
    print(
        "Size    :",
        f"{os.path.getsize(INDEX_PATH)/(1024**3):.3f} GiB",
    )
    print("Summary :", SUMMARY_PATH)
    print(
        "Build   :",
        f"{build_sec/60:.1f} min",
    )
    print("=" * 72)


if __name__ == "__main__":
    main()