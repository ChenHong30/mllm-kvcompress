# SPDX-License-Identifier: Apache-2.0
"""Generic data loading utilities for registered benchmarks."""

from __future__ import annotations

import ast
import io
import json
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image


@dataclass
class BenchmarkSample:
    """One benchmark sample normalized enough for generation and evaluation."""

    benchmark: str
    sample_id: str
    prompt: str
    media: list[Any]
    media_sources: list[str]
    references: list[str] = field(default_factory=list)
    choices: list[str] | None = None
    answer: str | None = None
    answer_index: int | None = None
    task_type: str = "open"
    config: str | None = None
    split: str | None = None
    native: dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 64


RowConverter = Callable[[dict[str, Any], int, str, str | None], list[BenchmarkSample]]


def load_samples_from_source(
    source: str | Path,
    split: str,
    configs: Iterable[str | None] | None,
    row_converter: RowConverter,
    limit: int | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
    cap_expanded_samples: bool = False,
) -> list[BenchmarkSample]:
    """Load rows from a local path or Hugging Face id and convert them to samples."""

    samples: list[BenchmarkSample] = []
    for config in list(configs or [None]):
        config_samples: list[BenchmarkSample] = []
        rows = iter_rows(str(source), split, config, streaming=streaming, cache_dir=cache_dir)
        if limit is not None:
            rows = islice(rows, limit)
        for index, row in enumerate(rows):
            config_samples.extend(row_converter(row, index, split, config))
            if cap_expanded_samples and limit is not None and len(config_samples) >= limit:
                config_samples = config_samples[:limit]
                break
        samples.extend(config_samples)
    return samples


def iter_rows(
    source: str,
    split: str,
    config: str | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
) -> Iterable[dict[str, Any]]:
    path = Path(source).expanduser()
    if path.exists():
        yield from iter_local_rows(path, split)
        return

    try:
        from datasets import load_dataset
    except Exception as exception:  # pragma: no cover - dependency guard
        raise ImportError("Install the benchmark extra first: pip install -e '.[bench]'") from exception

    kwargs = {"split": split, "streaming": streaming, "trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(source, config, **kwargs) if config else load_dataset(source, **kwargs)
    yield from dataset


def iter_local_rows(path: Path, split: str) -> Iterable[dict[str, Any]]:
    if path.is_file():
        yield from read_file_rows(path)
        return

    if (path / "dataset_info.json").exists() or any(path.glob("*.arrow")):
        try:
            from datasets import load_from_disk
        except Exception as exception:  # pragma: no cover - dependency guard
            raise ImportError("Install the benchmark extra first: pip install -e '.[bench]'") from exception
        dataset = load_from_disk(str(path))
        if hasattr(dataset, "keys") and split in dataset:
            dataset = dataset[split]
        yield from dataset
        return

    parquet_files = sorted(path.glob("**/*.parquet"))
    if parquet_files:
        for parquet in parquet_files:
            yield from read_file_rows(parquet)
        return

    json_files = sorted(path.glob("*.json"))
    if json_files:
        preferred = [file for file in json_files if split in file.stem.lower()] or json_files
        yield from read_file_rows(preferred[0])
        return

    raise FileNotFoundError(f"Could not find a supported native dataset file under {path}")


def read_file_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except Exception as exception:  # pragma: no cover - dependency guard
            raise ImportError("Reading parquet benchmark files requires pandas") from exception
        for row in pd.read_parquet(path).to_dict("records"):
            yield resolve_local_media_paths(row, path.parent)
        return
    if suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            if suffix == ".jsonl":
                for line in handle:
                    if line.strip():
                        yield resolve_local_media_paths(json.loads(line), path.parent)
                return
            payload = json.load(handle)
        if isinstance(payload, list):
            for row in payload:
                yield resolve_local_media_paths(row, path.parent)
            return
        if "images" in payload and "annotations" in payload:
            yield from iter_coco_caption_rows(payload, path.parent)
            return
        if "data" in payload and isinstance(payload["data"], list):
            for row in payload["data"]:
                yield resolve_local_media_paths(row, path.parent)
            return
        raise ValueError(f"Unsupported JSON dataset structure: {path}")
    raise ValueError(f"Unsupported dataset file type: {path}")


def iter_coco_caption_rows(payload: dict[str, Any], root: Path):
    captions_by_image: dict[Any, list[str]] = {}
    for ann in payload.get("annotations", []):
        captions_by_image.setdefault(ann.get("image_id"), []).append(str(ann.get("caption", "")))
    for image in payload.get("images", []):
        image_id = image.get("id")
        row = dict(image)
        row["image"] = str(root / image.get("file_name", ""))
        row["captions"] = captions_by_image.get(image_id, [])
        row["image_id"] = image_id
        yield row


def resolve_local_media_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_local_media_path_value(key, item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_local_media_paths(item, root) for item in value]
    return value


def load_sample_media(sample: BenchmarkSample) -> list[Image.Image]:
    return [load_image(item).convert("RGB") for item in sample.media]


def load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (str, Path)):
        return Image.open(value)
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    raise ValueError(f"Unsupported image value: {type(value).__name__}")


def extract_media(row: dict[str, Any]) -> list[Any]:
    media = []
    for key in ("image", "img"):
        value = row.get(key)
        if value is not None:
            media.append(value)
    for key in ("images", "image_list"):
        media.extend([value for value in as_list(row.get(key)) if value is not None])
    for index in range(1, 16):
        value = row.get(f"image_{index}")
        if value is not None:
            media.append(value)
    return media


def replace_numbered_media_tags(text: str, num_media: int, placeholder: str) -> str:
    for index in range(1, max(num_media, 8) + 1):
        for tag in (f"<image {index}>", f"<image_{index}>", f"{{image#{index}}}"):
            text = text.replace(tag, placeholder)
    return text


def media_source(value: Any, row: dict[str, Any]) -> str:
    if isinstance(value, (str, Path)):
        return str(value)
    for key in ("image_file_name", "filename", "file_name", "image_url", "image_coco_url", "image_id"):
        if row.get(key) is not None:
            return str(row[key])
    if isinstance(value, Image.Image) and getattr(value, "filename", ""):
        return str(value.filename)
    return ""


def format_choices(choices: list[str]) -> str:
    return " ".join(f"({letter(index)}) {choice}" for index, choice in enumerate(choices))


def letter(index: int | None) -> str:
    if index is None or index < 0:
        return ""
    if index < 26:
        return chr(ord("A") + index)
    return f"A{chr(ord('A') + index - 26)}"


def letter_to_index(value: str) -> int | None:
    value = value.strip().upper()
    if len(value) == 1 and "A" <= value <= "Z":
        return ord(value) - ord("A")
    return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def nonnegative_int(value: Any) -> int | None:
    parsed = safe_int(value)
    return parsed if parsed is not None and parsed >= 0 else None


def first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except Exception:
                pass
        return [value]
    return [value]


def as_string_list(value: Any) -> list[str]:
    return [str(item) for item in as_list(value) if item is not None and str(item) != ""]


def sanitize_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_native(item) for item in value]
    if isinstance(value, Image.Image):
        return {"type": "PIL.Image", "size": list(value.size), "mode": value.mode}
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value)}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _resolve_local_media_path_value(key: Any, value: Any, root: Path) -> Any:
    key_text = str(key)
    if isinstance(value, list):
        return [_resolve_local_media_path_value(key, item, root) for item in value]
    if isinstance(value, dict):
        return resolve_local_media_paths(value, root)
    if not _looks_like_media_key(key_text) or not isinstance(value, str):
        return value
    text = value.strip()
    if not text or "://" in text or Path(text).is_absolute():
        return value
    for candidate in (root / text, root / "images" / text):
        if candidate.exists():
            return str(candidate)
    return value


def _looks_like_media_key(key: str) -> bool:
    return key in {"image", "img", "images", "image_list", "image_path", "file_name", "filename"} or key.startswith("image_")
