import argparse
import json
import os
import time
from glob import glob
from typing import Dict, List, Tuple

import numpy as np

try:
    import hnswlib
except ImportError as e:
    raise SystemExit(
        "hnswlib is not installed. Install it with:\n"
        "py -m pip install hnswlib"
    ) from e


def load_chunks_meta(chunks_path: str) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row["chunk_id"])
            meta[cid] = {
                "chunk_id": cid,
                "doc_id": row.get("doc_id"),
                "page_start": row.get("page_start"),
                "year": row.get("year"),
                "txt_rel": row.get("txt_rel"),
                "text": row.get("text", ""),
            }
    return meta


def discover_part_pairs(parts_dir: str) -> List[Tuple[str, str]]:
    npy_files = sorted(glob(os.path.join(parts_dir, "*_emb_part_*.npy")))
    pairs: List[Tuple[str, str]] = []
    for npy_file in npy_files:
        meta_file = npy_file.replace("_emb_", "_meta_").replace(".npy", ".npz")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Missing meta file for part: {npy_file}\nExpected: {meta_file}")
        pairs.append((npy_file, meta_file))
    if not pairs:
        raise FileNotFoundError(f"No *_emb_part_*.npy files found in: {parts_dir}")
    return pairs


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def build_index_from_parts(
    chunks_path: str,
    parts_dir: str,
    out_dir: str,
    space: str = "cosine",
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    normalize: bool = True,
    save_text_preview: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Loading chunks meta...")
    chunk_meta = load_chunks_meta(chunks_path)
    print(f"Chunks meta loaded: {len(chunk_meta)}")

    print("Discovering parts...")
    part_pairs = discover_part_pairs(parts_dir)
    print(f"Parts found: {len(part_pairs)}")

    total_vectors = 0
    dim = None

    print("Counting vectors...")
    for npy_file, _ in part_pairs:
        x = np.load(npy_file, mmap_mode="r")
        if dim is None:
            dim = int(x.shape[1])
        total_vectors += int(x.shape[0])

    if dim is None:
        raise RuntimeError("Could not determine embedding dimension")

    print(f"Total vectors: {total_vectors}")
    print(f"Embedding dim: {dim}")

    print("Initializing HNSW index...")
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=total_vectors, ef_construction=ef_construction, M=m)
    index.set_ef(ef_search)

    chunk_ids: List[str] = []
    doc_ids: List[str] = []
    page_starts: List[int] = []
    years: List[str] = []
    txt_rels: List[str] = []
    texts: List[str] = []

    next_label = 0
    t0 = time.time()

    for part_no, (npy_file, meta_file) in enumerate(part_pairs, 1):
        emb = np.load(npy_file).astype(np.float32, copy=False)
        part_meta = np.load(meta_file, allow_pickle=True)

        if "chunk_ids" not in part_meta:
            raise RuntimeError(f"chunk_ids not found in {meta_file}")

        ids = part_meta["chunk_ids"]
        ids = [x.decode() if isinstance(x, bytes) else str(x) for x in ids]

        if len(ids) != emb.shape[0]:
            raise RuntimeError(
                f"Mismatch in {npy_file}: embeddings rows={emb.shape[0]} vs chunk_ids={len(ids)}"
            )

        valid_rows = []
        valid_labels = []

        for i, cid in enumerate(ids):
            row = chunk_meta.get(cid)
            if row is None:
                continue

            chunk_ids.append(cid)
            doc_ids.append("" if row.get("doc_id") is None else str(row.get("doc_id")))
            page_starts.append(int(row.get("page_start") or 0))
            years.append("" if row.get("year") is None else str(row.get("year")))
            txt_rels.append("" if row.get("txt_rel") is None else str(row.get("txt_rel")))
            if save_text_preview:
                texts.append("" if row.get("text") is None else str(row.get("text")))
            valid_rows.append(i)
            valid_labels.append(next_label)
            next_label += 1

        if not valid_rows:
            continue

        x = emb[np.array(valid_rows)]
        if normalize:
            x = normalize_rows(x)

        index.add_items(x, np.array(valid_labels, dtype=np.int64))

        elapsed = time.time() - t0
        speed = next_label / max(elapsed, 1e-6)
        eta = ((elapsed / part_no) * (len(part_pairs) - part_no)) if part_no > 0 else 0.0

        print(
            f"[{part_no}/{len(part_pairs)}] "
            f"indexed={next_label} "
            f"speed={speed:.1f} vec/s "
            f"eta={eta/60:.1f} min"
        )

    if next_label == 0:
        raise RuntimeError("No vectors were indexed")

    index_path = os.path.join(out_dir, "hnsw_index.bin")
    meta_path = os.path.join(out_dir, "meta.npz")
    summary_path = os.path.join(out_dir, "build_index_summary.json")

    print("Saving index...")
    index.save_index(index_path)

    print("Saving meta...")
    save_kwargs = {
        "chunk_ids": np.array(chunk_ids, dtype=object),
        "doc_ids": np.array(doc_ids, dtype=object),
        "page_starts": np.array(page_starts, dtype=np.int32),
        "years": np.array(years, dtype=object),
        "txt_rels": np.array(txt_rels, dtype=object),
    }
    if save_text_preview:
        save_kwargs["texts"] = np.array(texts, dtype=object)

    np.savez_compressed(meta_path, **save_kwargs)

    summary = {
        "mode": "raw_chunk_embeddings_from_parts",
        "chunks_path": chunks_path,
        "parts_dir": parts_dir,
        "out_dir": out_dir,
        "index_path": index_path,
        "meta_path": meta_path,
        "space": space,
        "embedding_dim": dim,
        "indexed_vectors": next_label,
        "normalize": normalize,
        "hnsw_M": m,
        "hnsw_ef_construction": ef_construction,
        "hnsw_ef_search": ef_search,
        "save_text_preview": save_text_preview,
        "time_sec": time.time() - t0,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("DONE")
    print(f"Index : {index_path}")
    print(f"Meta  : {meta_path}")
    print(f"Sum   : {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build HNSW search index directly from raw chunk embeddings parts"
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks JSONL")
    parser.add_argument("--parts-dir", required=True, help="Directory with *_emb_part_*.npy and *_meta_*.npz")
    parser.add_argument("--out-dir", required=True, help="Output directory for index")
    parser.add_argument("--space", default="cosine", choices=["cosine", "l2", "ip"])
    parser.add_argument("--m", type=int, default=32, help="HNSW M")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction")
    parser.add_argument("--ef-search", type=int, default=64, help="Default efSearch saved in the index object")
    parser.add_argument("--no-normalize", action="store_true", help="Disable row normalization before indexing")
    parser.add_argument("--save-text-preview", action="store_true", help="Also save chunk text into meta.npz")
    args = parser.parse_args()

    build_index_from_parts(
        chunks_path=args.chunks,
        parts_dir=args.parts_dir,
        out_dir=args.out_dir,
        space=args.space,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        normalize=not args.no_normalize,
        save_text_preview=args.save_text_preview,
    )


if __name__ == "__main__":
    main()
