# SPDX-License-Identifier: Apache-2.0
"""Download and prepare the official MileBench data archives."""

from __future__ import annotations

import tarfile
from pathlib import Path

from mllm_kvcompress.benchmarks.milebench.data import normalize_dataset_name

MILEBENCH_REPO_ID = "FreedomIntelligence/MileBench"
MILEBENCH_ARCHIVES = [f"MileBench_part{index}.tar.gz" for index in range(6)]


def ensure_milebench_data(
    data_dir: str | Path,
    datasets: list[str],
    split: str = "test",
    combine_image: int | None = None,
    download: bool = True,
    repo_id: str = MILEBENCH_REPO_ID,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    Ensure selected MileBench datasets exist locally.

    `data_dir` is the extracted official folder, usually `data/MileBench`. If any
    selected dataset annotation is missing and `download=True`, the six official
    archives are downloaded from Hugging Face and extracted.
    """
    data_dir = Path(data_dir)
    missing = missing_datasets(data_dir, datasets, split=split, combine_image=combine_image)
    if not missing:
        return data_dir

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_paths = local_milebench_archives(data_dir)
    if archive_paths:
        print(
            f"[milebench] Missing datasets under {data_dir}: {missing}. "
            f"Using local official archives in {data_dir}.",
            flush=True,
        )
    else:
        if not download:
            raise FileNotFoundError(
                f"MileBench data is missing under {data_dir}: {missing}, and no local "
                f"{MILEBENCH_ARCHIVES[0]} ... {MILEBENCH_ARCHIVES[-1]} archives were found. "
                "Run again without --no-download-data to download them automatically."
            )
        print(
            f"[milebench] Missing datasets under {data_dir}: {missing}. "
            f"Downloading official archives from {repo_id}.",
            flush=True,
        )
        archive_paths = download_milebench_archives(repo_id=repo_id, cache_dir=cache_dir, force_download=force_download)
    extract_archives(archive_paths, data_dir)

    missing = missing_datasets(data_dir, datasets, split=split, combine_image=combine_image)
    if missing:
        raise FileNotFoundError(
            f"MileBench download/extraction finished, but these datasets are still missing under {data_dir}: {missing}"
        )
    return data_dir


def missing_datasets(
    data_dir: str | Path,
    datasets: list[str],
    split: str = "test",
    combine_image: int | None = None,
) -> list[str]:
    data_dir = Path(data_dir)
    missing = []
    for dataset in datasets:
        dataset = normalize_dataset_name(dataset)
        dataset_dir = data_dir / dataset
        if not (dataset_dir / annotation_filename(dataset, split=split, combine_image=combine_image)).exists():
            missing.append(dataset)
    return missing


def local_milebench_archives(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    archive_paths = [data_dir / archive for archive in MILEBENCH_ARCHIVES]
    if all(path.exists() for path in archive_paths):
        return archive_paths
    parent_archive_paths = [data_dir.parent / archive for archive in MILEBENCH_ARCHIVES]
    if all(path.exists() for path in parent_archive_paths):
        return parent_archive_paths
    return []


def annotation_filename(dataset: str, split: str = "test", combine_image: int | None = None) -> str:
    if split == "adv":
        return f"{dataset}-adv.json"
    if combine_image and combine_image != 1:
        return f"{dataset}_combined_{combine_image}.json"
    return f"{dataset}.json"


def download_milebench_archives(
    repo_id: str = MILEBENCH_REPO_ID,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exception:
        raise ImportError("Install the benchmark extra first: pip install -e '.[bench]'") from exception

    archive_paths = []
    for archive in MILEBENCH_ARCHIVES:
        print(f"[milebench] downloading {archive}", flush=True)
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=archive,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            force_download=force_download,
        )
        archive_paths.append(Path(path))
    return archive_paths


def extract_archives(archive_paths: list[Path], data_dir: Path):
    for archive_path in archive_paths:
        extract_root = _extract_root_for_archive(archive_path, data_dir)
        print(f"[milebench] extracting {archive_path.name} -> {extract_root}", flush=True)
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, extract_root)


def _extract_root_for_archive(archive_path: Path, data_dir: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as tar:
        names = []
        for _ in range(200):
            member = tar.next()
            if member is None:
                break
            if member.name:
                names.append(member.name)
    top_levels = {Path(name).parts[0] for name in names if Path(name).parts}
    if "MileBench" in top_levels:
        return data_dir.parent
    return data_dir


def _safe_extract(tar: tarfile.TarFile, destination: Path):
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if destination != member_path and destination not in member_path.parents:
            raise RuntimeError(f"Unsafe path in MileBench archive: {member.name}")
    tar.extractall(destination)
