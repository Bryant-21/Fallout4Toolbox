from __future__ import annotations
import io
import os
from typing import Iterable

from json_handler import load_json, save_json, update_textures_json
from bgsm_bin import read_bgsm, BGSMData
from bgem_bin import read_bgem, BGEMData

BGSM_SIGNATURE = 0x4D534742
BGEM_SIGNATURE = 0x4D454742


def is_json_file(path: str) -> bool:
    try:
        with open(path, 'rb') as f:
            head = f.read(1)
            if not head:
                return False
            return head in (b'{', b'[')
    except OSError:
        return False


def detect_binary_type(path: str) -> str | None:
    try:
        with open(path, 'rb') as f:
            data = f.read(4)
            if len(data) < 4:
                return None
            sig = int.from_bytes(data, 'little')
            if sig == BGSM_SIGNATURE:
                return 'BGSM'
            if sig == BGEM_SIGNATURE:
                return 'BGEM'
            return None
    except OSError:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def each_material_file(root: str, skip_dirnames: set[str] | None = None, skip_roots: list[str] | None = None) -> Iterable[str]:
    skip_dirnames = skip_dirnames or set()
    skip_roots = [os.path.abspath(p) for p in (skip_roots or [])]
    for dirpath, dirnames, filenames in os.walk(root):
        # prevent descending into specified output folders or roots to avoid infinite copying
        if skip_dirnames:
            dirnames[:] = [d for d in dirnames if d not in skip_dirnames]
        if skip_roots:
            # prune any child dir that is under a skipped root
            keep = []
            parent = os.path.abspath(dirpath)
            for d in dirnames:
                full = os.path.abspath(os.path.join(parent, d))
                if any(full.startswith(rp) for rp in skip_roots):
                    continue
                keep.append(d)
            dirnames[:] = keep
        for name in filenames:
            lower = name.lower()
            if lower.endswith('.bgsm') or lower.endswith('.bgem') or lower.endswith('.json'):
                yield os.path.join(dirpath, name)


def prefix_texture(s: str | None, folder: str) -> str | None:
    """Insert the folder immediately before the filename component of s.
    Keeps existing parent directories and normalizes to forward slashes.
    Avoid double insertion if the folder is already directly before the filename.
    """
    if not s:
        return s
    q = s.replace('\\', '/')
    if '/' in q:
        dirpart, fname = q.rsplit('/', 1)
    else:
        dirpart, fname = '', q
    if dirpart and dirpart.endswith('/' + folder):
        return q
    if not dirpart and q.startswith(folder + '/'):
        return q
    if dirpart:
        return f"{dirpart}/{folder}/{fname}"
    else:
        return f"{folder}/{fname}"


def process_binary_bgsm(src_path: str, folders: list[str], out_root: str | None, logger=None) -> None:
    if logger:
        logger(f"Reading BGSM: {src_path}")
    with open(src_path, 'rb') as f:
        br = io.BufferedReader(f)
        bgsm = read_bgsm(br)
    # Update and write per folder
    for folder in folders:
        new_bgsm = BGSMData(**{**bgsm.__dict__})
        new_bgsm.DiffuseTexture = prefix_texture(new_bgsm.DiffuseTexture, folder) or ""
        new_bgsm.NormalTexture = prefix_texture(new_bgsm.NormalTexture, folder) or ""
        new_bgsm.SmoothSpecTexture = prefix_texture(new_bgsm.SmoothSpecTexture, folder) or ""
        # Build output path
        base_dir = out_root or os.path.dirname(src_path)
        target_dir = os.path.join(base_dir, folder)
        ensure_dir(target_dir)
        out_path = os.path.join(target_dir, os.path.basename(src_path))
        with open(out_path, 'wb') as out_f:
            bw = io.BufferedWriter(out_f)
            new_bgsm.write(bw)
            bw.flush()
        if logger:
            logger(f"Wrote BGSM: {out_path} (folder={folder})")


def process_binary_bgem(src_path: str, folders: list[str], out_root: str | None, logger=None) -> None:
    if logger:
        logger(f"Reading BGEM: {src_path}")
    with open(src_path, 'rb') as f:
        br = io.BufferedReader(f)
        bgem = read_bgem(br)
    for folder in folders:
        new_bgem = BGEMData(**{**bgem.__dict__})
        new_bgem.BaseTexture = prefix_texture(new_bgem.BaseTexture, folder) or ""
        new_bgem.NormalTexture = prefix_texture(new_bgem.NormalTexture, folder) or ""
        if new_bgem.SpecularTexture is not None:
            new_bgem.SpecularTexture = prefix_texture(new_bgem.SpecularTexture, folder) or ""
        base_dir = out_root or os.path.dirname(src_path)
        target_dir = os.path.join(base_dir, folder)
        ensure_dir(target_dir)
        out_path = os.path.join(target_dir, os.path.basename(src_path))
        with open(out_path, 'wb') as out_f:
            bw = io.BufferedWriter(out_f)
            new_bgem.write(bw)
            bw.flush()
        if logger:
            logger(f"Wrote BGEM: {out_path} (folder={folder})")


def process_json(src_path: str, folders: list[str], out_root: str | None, include_bgsm: bool = True, include_bgem: bool = True, logger=None) -> None:
    mat_type, obj = load_json(src_path)
    # Filter by type if requested
    if (mat_type == 'BGSM' and not include_bgsm) or (mat_type == 'BGEM' and not include_bgem):
        if logger:
            logger(f"Skipping JSON ({mat_type}) due to filter: {src_path}")
        return
    for folder in folders:
        new_obj = update_textures_json(obj, mat_type, folder)
        base_dir = out_root or os.path.dirname(src_path)
        target_dir = os.path.join(base_dir, folder)
        ensure_dir(target_dir)
        out_path = os.path.join(target_dir, os.path.basename(src_path))
        save_json(out_path, new_obj)
        if logger:
            logger(f"Wrote JSON {mat_type}: {out_path} (folder={folder})")


def run(input_dir: str, folders: list[str], out_root: str | None = None, include_bgsm: bool = True, include_bgem: bool = True, logger=None) -> None:
    def emit(msg: str) -> None:
        if logger:
            try:
                logger(msg)
            except Exception:
                # Never let logging break the run
                pass
    input_dir = os.path.abspath(input_dir)
    out_root_abs = os.path.abspath(out_root) if out_root else None

    emit(f"Run params:\n  input_dir={input_dir}\n  out_root={out_root_abs or '(same as input)'}\n  folders={folders}\n  include_bgsm={include_bgsm} include_bgem={include_bgem}")

    # Determine directories to skip during scan to avoid rescanning outputs
    skip_names: set[str] = set()
    skip_roots: list[str] = []
    if out_root_abs is None:
        # Writing under input_dir/<folder>; skip those folder names during the same walk
        skip_names = set(folders)
    else:
        # If output root resides inside input_dir, skip those concrete output roots
        out_root_abs_norm = os.path.normcase(out_root_abs)
        input_dir_norm = os.path.normcase(input_dir)
        if out_root_abs_norm.startswith(input_dir_norm):
            skip_roots = [os.path.join(out_root_abs, f) for f in folders]

    if skip_names:
        emit(f"Skipping subdirectories by name: {sorted(skip_names)}")
    if skip_roots:
        emit(f"Skipping concrete output roots: {skip_roots}")

    scanned = 0
    processed_bgsm = 0
    processed_bgem = 0
    processed_json_bgsm = 0
    processed_json_bgem = 0
    skipped_filtered = 0
    skipped_unknown = 0
    errors = 0

    for path in each_material_file(input_dir, skip_dirnames=skip_names, skip_roots=skip_roots):
        scanned += 1
        try:
            if is_json_file(path):
                # Determine type from JSON for logging
                from json_handler import detect_material_type_from_json
                try:
                    with open(path, 'r', encoding='utf-8') as jf:
                        import json as _json
                        obj = _json.load(jf)
                    jtype = detect_material_type_from_json(obj)
                except Exception:
                    jtype = 'JSON'
                emit(f"Found JSON ({jtype}): {path}")
                if (jtype == 'BGSM' and not include_bgsm) or (jtype == 'BGEM' and not include_bgem):
                    skipped_filtered += 1
                    emit(f"  -> Skipped by type filter")
                else:
                    process_json(path, folders, out_root_abs, include_bgsm=include_bgsm, include_bgem=include_bgem, logger=logger)
                    if jtype == 'BGSM':
                        processed_json_bgsm += 1
                    elif jtype == 'BGEM':
                        processed_json_bgem += 1
                continue
            mtype = detect_binary_type(path)
            if mtype == 'BGSM':
                if not include_bgsm:
                    skipped_filtered += 1
                    emit(f"Found BGSM (binary), skipped by filter: {path}")
                else:
                    emit(f"Processing BGSM (binary): {path}")
                    process_binary_bgsm(path, folders, out_root_abs, logger=logger)
                    processed_bgsm += 1
            elif mtype == 'BGEM':
                if not include_bgem:
                    skipped_filtered += 1
                    emit(f"Found BGEM (binary), skipped by filter: {path}")
                else:
                    emit(f"Processing BGEM (binary): {path}")
                    process_binary_bgem(path, folders, out_root_abs, logger=logger)
                    processed_bgem += 1
            else:
                skipped_unknown += 1
                emit(f"Unknown or unsupported file signature, skipped: {path}")
        except Exception as ex:
            errors += 1
            emit(f"Error processing {path}: {ex}")

    emit(
        "Summary:\n"
        f"  scanned={scanned}\n"
        f"  processed_bgsm_binary={processed_bgsm}\n"
        f"  processed_bgem_binary={processed_bgem}\n"
        f"  processed_json_bgsm={processed_json_bgsm}\n"
        f"  processed_json_bgem={processed_json_bgem}\n"
        f"  skipped_filtered={skipped_filtered}\n"
        f"  skipped_unknown={skipped_unknown}\n"
        f"  errors={errors}"
    )

# Note: This module exposes run() for use by the PySide6 UI.
# The old CLI has been removed to focus on UI integration.
