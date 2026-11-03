@echo off
setlocal

set "PY=py"
set "SCRIPT=C:\VAST2_UPLOAD\_scripts\build_index\build_chunk_index_from_parts_raw.py"

set "CHUNKS=C:\VAST2_UPLOAD\chunks_engineering_ru_rag.jsonl"
set "PARTS_DIR=C:\VAST2_UPLOAD\out\engineering_ru_rag_e5_base_torch\parts"
set "OUT_DIR=C:\VAST2_UPLOAD\out\chunk_index_raw"

set "M=32"
set "EF_CONSTRUCTION=200"
set "EF_SEARCH=64"

%PY% "%SCRIPT%" ^
  --chunks "%CHUNKS%" ^
  --parts-dir "%PARTS_DIR%" ^
  --out-dir "%OUT_DIR%" ^
  --space cosine ^
  --m %M% ^
  --ef-construction %EF_CONSTRUCTION% ^
  --ef-search %EF_SEARCH%

pause
