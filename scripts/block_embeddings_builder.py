# -*- coding: utf-8 -*-
r"""
build_embeddings_universal_v1.py

Назначение
==========
Потоковый расчёт эмбеддингов по JSONL-файлу чанков с сохранением результатов
по независимым блокам. Программа не строит HNSW-индекс. Она готовит части,
которые потом можно:
1) объединить с результатами работы этой же программы на других компьютерах;
2) использовать для отдельной сборки общего HNSW-индекса.

Главная идея этой версии
========================
Это универсальная версия исходной программы build_embeddings_openvino_v1.py.
Логика обработки, формат частей, resume, блоки, tqdm, ETA и отчёты сохранены
максимально близко к рабочей OpenVINO-версии. Управляемое отличие только одно:
backend задаётся параметром командной строки.

Поддерживаемые backend
======================
- torch
- onnx
- openvino

Архитектура
===========
- вход: любой JSONL-файл чанков (путь задаётся явно через --chunks)
- выход: OUT_ROOT\\<base_name>_e5_base_<backend>\\...
- расчёт: Sentence Transformers + выбранный backend
- хранение: float32
- размер блока: по умолчанию 5000 чанков
- checkpoint/resume: по тройке файлов части (.npy + .npz + .json)

Важно по совместимости
======================
- Имена файлов частей сохранены в том же стиле, что и в исходной версии.
- Состав meta сохраняется без поля text.
- Resume использует фактическую рабочую проверку: если существуют .npy, .npz
  и .json для блока, блок считается готовым и пропускается.
- Отчёты можно расширять, но старые поля удалять нельзя.
- В этой версии дополнительно фиксируется фактическая вычислительная среда:
  CPU/GPU, доступность CUDA, версия torch/CUDA и имя первой GPU, если она видна.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

try:
    import orjson  # type: ignore

    def json_loads_bytes(data: bytes) -> Dict[str, Any]:
        return orjson.loads(data)

except Exception:
    def json_loads_bytes(data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8"))

import numpy as np
from tqdm import tqdm

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

PROGRAM = "BUILD_EMBEDDINGS_UNIVERSAL_V1"
VERSION = "2026-04-07 v1.0"
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BACKEND = "onnx"
DEFAULT_BATCH = 256
DEFAULT_BLOCK_SIZE = 5000
VALID_BACKENDS = ("torch", "onnx", "openvino")


class RunLogger:
    """Логгер, который пишет и в файл, и в консоль через tqdm.write.

    Это позволяет не ломать отображение progress bar и одновременно хранить
    нормальный текстовый лог без мусора от tqdm.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8", newline="\n")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        tqdm.write(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    sec = int(round(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def count_non_empty_lines_fast(path: Path) -> int:
    """Считает только непустые строки JSONL.

    Вся логика глобальных индексов в программе привязана именно к непустым
    строкам, чтобы прогресс и блоки совпадали с реальным рабочим форматом.
    """
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_meta_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Сохраняем все поля чанка кроме text.

    Это повторяет поведение рабочей исходной версии и нужно для совместимости
    с downstream merge/search/loaders.
    """
    meta: Dict[str, Any] = {}
    for k, v in rec.items():
        if k == "text":
            continue
        meta[k] = v
    return meta


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.save(f, array)
    os.replace(tmp_path, path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def write_report_txt(path: Path, summary: Dict[str, Any]) -> None:
    """Пишем и старые поля, и расширенный человекочитаемый блок.

    Старые поля оставляем для обратной совместимости. Ниже добавляем понятные
    строки сравнения режимов torch / onnx / openvino.
    """
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("Summary\n")
        f.write("=======\n")
        f.write(f"Backend requested: {summary.get('backend_requested')}\n")
        f.write(f"Backend reported : {summary.get('backend_reported')}\n")
        f.write(f"Model            : {summary.get('model')}\n")
        f.write(f"Batch            : {summary.get('batch')}\n")
        f.write(f"Block size       : {summary.get('block_size')}\n")
        f.write(f"Total chunks     : {summary.get('total_chunks')}\n")
        f.write(f"Processed blocks : {summary.get('processed_blocks')}\n")
        f.write(f"Skipped blocks   : {summary.get('skipped_blocks')}\n")
        f.write(f"Elapsed sec      : {summary.get('elapsed_sec')}\n")
        f.write(f"Avg speed        : {summary.get('avg_speed_chunks_per_sec')} chunks/s\n")
        f.write(f"Last block speed : {summary.get('last_block_speed_chunks_per_sec')} chunks/s\n")
        f.write(f"Compute device   : {summary.get('compute_device')}\n")
        f.write(f"CUDA available   : {summary.get('cuda_available')}\n")
        f.write(f"Torch version    : {summary.get('torch_version')}\n")
        f.write(f"CUDA version     : {summary.get('torch_cuda_version')}\n")
        f.write(f"GPU name         : {summary.get('gpu_name')}\n")
        f.write(f"GPU count        : {summary.get('gpu_count')}\n")
        f.write("\n")
        f.write("Raw summary fields\n")
        f.write("==================\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")


def file_triplet_exists(emb_path: Path, meta_path: Path, info_path: Path) -> bool:
    return emb_path.exists() and meta_path.exists() and info_path.exists()


class JsonlNonEmptyReader:
    """Последовательный reader только по непустым строкам.

    Глобальный индекс определяется по непустым строкам. Это совпадает с тем,
    как считается total_chunks и как формируются блоки.
    """

    def __init__(self, path: Path):
        self.path = path
        self.fh: BinaryIO = path.open("rb")
        self.index_nonempty = 0
        self.eof = False

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass

    def read_next_non_empty(self) -> Optional[Tuple[int, int, bytes]]:
        while True:
            off = self.fh.tell()
            line = self.fh.readline()
            if not line:
                self.eof = True
                return None
            if not line.strip():
                continue
            idx = self.index_nonempty
            self.index_nonempty += 1
            return idx, off, line

    def fast_forward_to(self, target_index: int) -> int:
        """Перемотка до нужного глобального индекса без JSON-парсинга."""
        while self.index_nonempty < target_index:
            item = self.read_next_non_empty()
            if item is None:
                break
        return self.index_nonempty


def detect_runtime_device() -> Dict[str, Any]:
    """Определяет фактическую вычислительную среду без изменения логики расчёта.

    Ничего не переключает принудительно. Только фиксирует, доступна ли CUDA,
    какая версия torch загружена и какая GPU видна из текущего процесса.
    Это нужно для честной записи в лог и отчёты, чтобы было видно, шёл расчёт
    на CPU или на GPU.
    """
    info: Dict[str, Any] = {
        "torch_imported": torch is not None,
        "torch_version": None,
        "cuda_available": False,
        "torch_cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
        "compute_device": "cpu",
    }

    if torch is None:
        return info

    try:
        info["torch_version"] = str(torch.__version__)
    except Exception:
        pass

    try:
        info["torch_cuda_version"] = str(torch.version.cuda) if torch.version.cuda is not None else None
    except Exception:
        pass

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["cuda_available"] = False

    if info["cuda_available"]:
        info["compute_device"] = "cuda"
        try:
            info["gpu_count"] = int(torch.cuda.device_count())
        except Exception:
            info["gpu_count"] = 0
        try:
            if info["gpu_count"] > 0:
                info["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:
            info["gpu_name"] = None

    return info


class Embedder:
    """Обёртка над SentenceTransformer с унификацией backend.

    Для openvino разрешено передавать ov_config. Для torch и onnx логика остаётся
    обычной и не меняет внешний формат программы.
    """

    def __init__(self, model_name: str, backend: str, ov_cache_dir: Optional[Path], logger: RunLogger):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence_transformers is not installed in this environment. "
                f"Original import error: {_IMPORT_ERROR!r}"
            )

        self.model_name = model_name
        self.backend = backend
        self.ov_cache_dir = ov_cache_dir
        self.logger = logger
        self.model = self._load_model()
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.backend_reported = self._get_backend_reported()

    def _load_model(self):
        model_kwargs: Dict[str, Any] = {}
        if self.backend == "openvino":
            ov_config: Dict[str, Any] = {
                "PERFORMANCE_HINT": "THROUGHPUT",
            }
            if self.ov_cache_dir is not None:
                self.ov_cache_dir.mkdir(parents=True, exist_ok=True)
                ov_config["CACHE_DIR"] = str(self.ov_cache_dir)
            model_kwargs = {
                "ov_config": ov_config,
            }
            self.logger.log(
                f"Loading model with backend=openvino ov_config={json.dumps(ov_config, ensure_ascii=False)}"
            )
        else:
            self.logger.log(f"Loading model with backend={self.backend}")

        kwargs: Dict[str, Any] = {
            "model_name_or_path": self.model_name,
            "backend": self.backend,
        }
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        return SentenceTransformer(**kwargs)

    def _get_backend_reported(self) -> str:
        """Пытаемся получить backend из объекта модели.

        В разных версиях sentence-transformers реализация может отличаться,
        поэтому делаем несколько безопасных попыток. Если значение не удалось
        получить, возвращаем 'unknown', а программа продолжает работу.
        """
        try:
            if hasattr(self.model, "get_backend"):
                val = self.model.get_backend()
                if val is not None:
                    return str(val)
        except Exception:
            pass

        for attr_name in ("backend", "_backend"):
            try:
                if hasattr(self.model, attr_name):
                    val = getattr(self.model, attr_name)
                    if val is not None:
                        return str(val)
            except Exception:
                pass

        return "unknown"

    def encode_passages(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Кодирование passages в том же стиле, как в исходной версии.

        Префикс passage: оставлен без изменений, чтобы не менять логику расчёта
        эмбеддингов относительно текущего рабочего пайплайна.
        """
        prefixed = ["passage: " + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
            precision="float32",
        )
        return np.asarray(vecs, dtype=np.float32)


class BlockWriter:
    def __init__(
        self,
        parts_dir: Path,
        base_name: str,
        chunks_path: Path,
        model_name: str,
        backend_requested: str,
        backend_reported: str,
        dim: int,
        dtype_name: str,
        block_size: int,
    ):
        self.parts_dir = parts_dir
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.base_name = base_name
        self.chunks_path = chunks_path
        self.model_name = model_name
        self.backend_requested = backend_requested
        self.backend_reported = backend_reported
        self.dim = dim
        self.dtype_name = dtype_name
        self.block_size = block_size
        self.host_name = socket.gethostname()

    def paths(self, block_id: int) -> Tuple[Path, Path, Path]:
        bid = f"{block_id:06d}"
        emb_path = self.parts_dir / f"{self.base_name}_emb_part_{bid}.npy"
        meta_path = self.parts_dir / f"{self.base_name}_meta_part_{bid}.npz"
        info_path = self.parts_dir / f"{self.base_name}_part_info_{bid}.json"
        return emb_path, meta_path, info_path

    def save_block(
        self,
        block_id: int,
        global_start_index: int,
        global_end_index_exclusive: int,
        embeddings: np.ndarray,
        chunk_ids: List[Any],
        metas: List[Dict[str, Any]],
        offsets: List[int],
    ) -> Tuple[Path, Path, Path]:
        emb_path, meta_path, info_path = self.paths(block_id)

        chunk_ids_arr = np.asarray(chunk_ids, dtype=object)
        metas_arr = np.asarray(metas, dtype=object)
        offsets_arr = np.asarray(offsets, dtype=np.int64)

        save_npy_atomic(emb_path, embeddings)
        save_npz_atomic(
            meta_path,
            chunk_ids=chunk_ids_arr,
            metas=metas_arr,
            offsets=offsets_arr,
        )

        info = {
            "program": PROGRAM,
            "version": VERSION,
            "created_at": iso_now(),
            "host_name": self.host_name,
            "base_name": self.base_name,
            "block_id": int(block_id),
            "block_size": int(self.block_size),
            "items": int(embeddings.shape[0]),
            "global_start_index": int(global_start_index),
            "global_end_index_exclusive": int(global_end_index_exclusive),
            "model": self.model_name,
            "backend": self.backend_requested,
            "backend_requested": self.backend_requested,
            "backend_reported": self.backend_reported,
            "dim": int(self.dim),
            "dtype": self.dtype_name,
            "source_chunks_path": str(self.chunks_path),
            "source_chunks_file": self.chunks_path.name,
            "emb_file": emb_path.name,
            "meta_file": meta_path.name,
            "status": "done",
        }
        write_json_atomic(info_path, info)
        return emb_path, meta_path, info_path


class BlockAccumulator:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunk_ids: List[Any] = []
        self.metas: List[Dict[str, Any]] = []
        self.offsets: List[int] = []
        self._vec_parts: List[np.ndarray] = []

    def add_batch(self, vecs: np.ndarray, chunk_ids: List[Any], metas: List[Dict[str, Any]], offsets: List[int]) -> None:
        if vecs.size:
            self._vec_parts.append(np.asarray(vecs, dtype=np.float32))
        self.chunk_ids.extend(chunk_ids)
        self.metas.extend(metas)
        self.offsets.extend(offsets)

    def finalize(self) -> Tuple[np.ndarray, List[Any], List[Dict[str, Any]], List[int]]:
        if self._vec_parts:
            embeddings = np.vstack(self._vec_parts).astype(np.float32, copy=False)
        else:
            embeddings = np.empty((0, self.dim), dtype=np.float32)
        return embeddings, self.chunk_ids, self.metas, self.offsets


class BatchBuffers:
    def __init__(self):
        self.texts: List[str] = []
        self.chunk_ids: List[Any] = []
        self.metas: List[Dict[str, Any]] = []
        self.offsets: List[int] = []

    def add(self, text: str, chunk_id: Any, meta: Dict[str, Any], offset: int) -> None:
        self.texts.append(text)
        self.chunk_ids.append(chunk_id)
        self.metas.append(meta)
        self.offsets.append(offset)

    def clear(self) -> None:
        self.texts.clear()
        self.chunk_ids.clear()
        self.metas.clear()
        self.offsets.clear()

    def __len__(self) -> int:
        return len(self.texts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Full path to input JSONL chunks file")
    ap.add_argument("--out-root", required=True, help="Root output directory")
    ap.add_argument("--base-name", required=True, help="Logical base name used for all output file names")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, help="Inference backend: torch | onnx | openvino")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Encode batch size")
    ap.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="Chunks per output block")
    ap.add_argument("--block-start", type=int, default=0, help="Start block id (inclusive)")
    ap.add_argument("--block-end", type=int, default=-1, help="End block id (exclusive); -1 means until EOF")
    ap.add_argument("--ov-cache-dir", default=None, help="Optional OpenVINO cache dir")
    args = ap.parse_args()

    if args.backend not in VALID_BACKENDS:
        raise ValueError(f"--backend must be one of: {', '.join(VALID_BACKENDS)}")
    if args.batch <= 0:
        raise ValueError("--batch must be > 0")
    if args.block_size <= 0:
        raise ValueError("--block-size must be > 0")
    if args.block_start < 0:
        raise ValueError("--block-start must be >= 0")
    if args.block_end != -1 and args.block_end < args.block_start:
        raise ValueError("--block-end must be -1 or >= --block-start")

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks not found: {chunks_path}")

    base_name = str(args.base_name).strip()
    if not base_name:
        raise ValueError("--base-name must not be empty")

    out_root = Path(args.out_root)
    out_dir = out_root / f"{base_name}_e5_base_{args.backend}"
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"{base_name}_build_embed_resume.log"
    report_txt = out_dir / f"{base_name}_build_embed_report.txt"
    report_json = out_dir / f"{base_name}_build_embed_report.json"
    ov_cache_dir = Path(args.ov_cache_dir) if args.ov_cache_dir else (out_dir / f"{base_name}_openvino_cache")

    logger = RunLogger(log_path)
    t0 = time.perf_counter()

    try:
        logger.log(f"Program: {PROGRAM}")
        logger.log(f"Version: {VERSION}")
        logger.log(f"Chunks file: {chunks_path}")
        logger.log(f"Out root: {out_root}")
        logger.log(f"Out dir: {out_dir}")
        logger.log(f"Base name: {base_name}")
        logger.log(f"Model: {args.model}")
        logger.log(f"Backend: {args.backend}")
        logger.log(f"Batch: {args.batch}")
        logger.log(f"Block size: {args.block_size}")
        logger.log(f"Block start: {args.block_start}")
        logger.log(f"Block end: {args.block_end}")
        if args.backend == "openvino":
            logger.log(f"OpenVINO cache dir: {ov_cache_dir}")

        logger.log("Counting total non-empty chunks in source file...")
        total_chunks_all = count_non_empty_lines_fast(chunks_path)
        logger.log(f"Total chunks (non-empty JSONL lines): {total_chunks_all}")

        runtime_device = detect_runtime_device()
        logger.log(f"Torch imported: {runtime_device.get('torch_imported')}")
        logger.log(f"Torch version: {runtime_device.get('torch_version')}")
        logger.log(f"CUDA available: {runtime_device.get('cuda_available')}")
        logger.log(f"Torch CUDA version: {runtime_device.get('torch_cuda_version')}")
        logger.log(f"GPU count: {runtime_device.get('gpu_count')}")
        logger.log(f"GPU name: {runtime_device.get('gpu_name')}")
        logger.log(f"Compute device detected: {runtime_device.get('compute_device')}")

        embedder = Embedder(args.model, args.backend, ov_cache_dir if args.backend == "openvino" else None, logger)
        logger.log(f"Embedding dim: {embedder.dim}")
        logger.log(f"Model backend reported by SentenceTransformer: {embedder.backend_reported}")

        writer = BlockWriter(
            parts_dir=parts_dir,
            base_name=base_name,
            chunks_path=chunks_path,
            model_name=args.model,
            backend_requested=args.backend,
            backend_reported=embedder.backend_reported,
            dim=embedder.dim,
            dtype_name="float32",
            block_size=args.block_size,
        )

        total_blocks_written = 0
        total_items_written = 0
        skipped_ready_blocks = 0
        skipped_empty_text = 0
        bad_json = 0
        current_position_chunks = min(args.block_start * args.block_size, total_chunks_all)
        last_block_speed: Optional[float] = None

        reader = JsonlNonEmptyReader(chunks_path)
        try:
            block_id = args.block_start
            while True:
                if args.block_end != -1 and block_id >= args.block_end:
                    break

                global_start = block_id * args.block_size
                if global_start >= total_chunks_all:
                    break
                global_end = global_start + args.block_size

                # Держим reader синхронизированным с началом текущего блока.
                if reader.index_nonempty < global_start:
                    reader.fast_forward_to(global_start)

                emb_path, meta_path, info_path = writer.paths(block_id)
                if file_triplet_exists(emb_path, meta_path, info_path):
                    skipped_ready_blocks += 1
                    reader.fast_forward_to(min(global_end, total_chunks_all))
                    current_position_chunks = min(reader.index_nonempty, total_chunks_all)
                    eta = None if last_block_speed is None or last_block_speed <= 0 else max(0, total_chunks_all - current_position_chunks) / last_block_speed
                    logger.log(
                        f"skip block {block_id:06d} (already done) "
                        f"progress={current_position_chunks}/{total_chunks_all} "
                        f"speed_last_block={'n/a' if last_block_speed is None else f'{last_block_speed:.2f}'} chunks/s "
                        f"eta={format_eta(eta)}"
                    )
                    block_id += 1
                    continue

                logger.log(f"block start id={block_id:06d} global_range=[{global_start}, {global_end})")
                block_t0 = time.perf_counter()
                accumulator = BlockAccumulator(embedder.dim)
                batch_buf = BatchBuffers()
                raw_seen_in_block = 0

                pbar = tqdm(total=args.block_size, desc=f"block {block_id:06d}", unit="chunk", leave=True)
                try:
                    while reader.index_nonempty < global_end:
                        item = reader.read_next_non_empty()
                        if item is None:
                            break
                        idx, offset, raw_line = item
                        raw_seen_in_block += 1
                        pbar.update(1)

                        try:
                            rec = json_loads_bytes(raw_line)
                        except Exception as exc:
                            bad_json += 1
                            raise RuntimeError(
                                f"Bad JSON at non-empty chunk index {idx}, offset {offset}, file {chunks_path}"
                            ) from exc

                        text = rec.get("text", "")
                        if not text:
                            skipped_empty_text += 1
                            continue

                        batch_buf.add(
                            text=text,
                            chunk_id=rec.get("chunk_id"),
                            meta=build_meta_record(rec),
                            offset=offset,
                        )

                        if len(batch_buf) >= args.batch:
                            vecs = embedder.encode_passages(batch_buf.texts, args.batch)
                            accumulator.add_batch(vecs, batch_buf.chunk_ids, batch_buf.metas, batch_buf.offsets)
                            batch_buf.clear()

                    if len(batch_buf) > 0:
                        vecs = embedder.encode_passages(batch_buf.texts, args.batch)
                        accumulator.add_batch(vecs, batch_buf.chunk_ids, batch_buf.metas, batch_buf.offsets)
                        batch_buf.clear()
                finally:
                    pbar.close()

                if raw_seen_in_block == 0:
                    break

                embeddings, chunk_ids, metas, offsets = accumulator.finalize()
                actual_global_end = min(global_start + raw_seen_in_block, total_chunks_all)
                writer.save_block(
                    block_id=block_id,
                    global_start_index=global_start,
                    global_end_index_exclusive=actual_global_end,
                    embeddings=embeddings,
                    chunk_ids=chunk_ids,
                    metas=metas,
                    offsets=offsets,
                )

                block_elapsed = time.perf_counter() - block_t0
                total_blocks_written += 1
                total_items_written += int(embeddings.shape[0])
                current_position_chunks = actual_global_end

                if embeddings.shape[0] > 0 and block_elapsed > 0:
                    last_block_speed = embeddings.shape[0] / block_elapsed
                else:
                    last_block_speed = None

                eta = None
                if last_block_speed is not None and last_block_speed > 0:
                    eta = max(0, total_chunks_all - current_position_chunks) / last_block_speed

                logger.log(
                    f"block done id={block_id:06d} items={embeddings.shape[0]} raw_seen={raw_seen_in_block} "
                    f"progress={current_position_chunks}/{total_chunks_all} "
                    f"speed_last_block={'n/a' if last_block_speed is None else f'{last_block_speed:.2f}'} chunks/s "
                    f"eta={format_eta(eta)}"
                )

                if reader.eof or current_position_chunks >= total_chunks_all:
                    break
                block_id += 1
        finally:
            reader.close()

        elapsed = time.perf_counter() - t0
        overall_speed = (total_items_written / elapsed) if elapsed > 0 else 0.0

        summary = {
            # Старые поля из рабочей версии сохраняем.
            "program": PROGRAM,
            "version": VERSION,
            "chunks_path": str(chunks_path),
            "out_root": str(out_root),
            "out_dir": str(out_dir),
            "parts_dir": str(parts_dir),
            "base_name": base_name,
            "model": args.model,
            "backend": args.backend,
            "batch": args.batch,
            "block_size": args.block_size,
            "block_start": args.block_start,
            "block_end": args.block_end,
            "dim": embedder.dim,
            "dtype": "float32",
            "total_chunks_all": total_chunks_all,
            "total_blocks_written": total_blocks_written,
            "total_items_written": total_items_written,
            "skipped_ready_blocks": skipped_ready_blocks,
            "skipped_empty_text": skipped_empty_text,
            "bad_json": bad_json,
            "time_sec": round(elapsed, 6),
            "chunks_per_sec": round(overall_speed, 6),
            "last_block_speed_chunks_per_sec": None if last_block_speed is None else round(last_block_speed, 6),
            "openvino_cache_dir": str(ov_cache_dir) if args.backend == "openvino" else None,
            # Новые поля для честного сравнения режимов.
            "backend_requested": args.backend,
            "backend_reported": embedder.backend_reported,
            "total_chunks": total_chunks_all,
            "processed_blocks": total_blocks_written,
            "skipped_blocks": skipped_ready_blocks,
            "elapsed_sec": round(elapsed, 6),
            "avg_speed_chunks_per_sec": round(overall_speed, 6),
            "input_file_name": chunks_path.name,
            "torch_imported": runtime_device.get("torch_imported"),
            "torch_version": runtime_device.get("torch_version"),
            "cuda_available": runtime_device.get("cuda_available"),
            "torch_cuda_version": runtime_device.get("torch_cuda_version"),
            "gpu_count": runtime_device.get("gpu_count"),
            "gpu_name": runtime_device.get("gpu_name"),
            "compute_device": runtime_device.get("compute_device"),
        }
        write_report_txt(report_txt, summary)
        write_json_atomic(report_json, summary)

        logger.log("OK")
        logger.log(f"Report TXT: {report_txt}")
        logger.log(f"Report JSON: {report_json}")
        logger.log(f"Total blocks written: {total_blocks_written}")
        logger.log(f"Total items written: {total_items_written}")
        logger.log(f"Skipped ready blocks: {skipped_ready_blocks}")
        logger.log(f"Skipped empty text: {skipped_empty_text}")
        logger.log(f"Bad JSON: {bad_json}")
        logger.log(f"Overall speed: {overall_speed:.2f} chunks/s")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
