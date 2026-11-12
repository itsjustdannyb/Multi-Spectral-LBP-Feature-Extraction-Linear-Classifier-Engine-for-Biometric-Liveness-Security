from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

REAL_LABEL = 1
SPOOF_LABEL = 0

# Clockwise 8-neighborhood offsets and bit weights matching lbp_calculated_pixel.
_LBP_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (-1, -1, 1),    # top-left
    (-1, 0, 2),     # top
    (-1, 1, 4),     # top-right
    (0, 1, 8),      # right
    (1, 1, 16),     # bottom-right
    (1, 0, 32),     # bottom
    (1, -1, 64),    # bottom-left
    (0, -1, 128),   # left
)


def compute_lbp_map(gray: np.ndarray) -> np.ndarray:
    """Compute the 8-neighbor LBP map using zero-padded image boundaries."""
    height, width = gray.shape
    padded = np.pad(gray, 1, mode="constant", constant_values=0)
    center = padded[1:-1, 1:-1]
    img_lbp = np.zeros((height, width), dtype=np.uint8)

    for dx, dy, weight in _LBP_OFFSETS:
        neighbor = padded[1 + dx : 1 + dx + height, 1 + dy : 1 + dy + width]
        img_lbp |= (neighbor >= center).astype(np.uint8) * weight

    return img_lbp


def compute_spatial_histogram(
    img_lbp: np.ndarray, grid: int = 4, n_bins: int = 256
) -> np.ndarray:
    """Split an LBP map into a grid x grid of blocks and concatenate normalized histograms."""
    hists = []
    for rows in np.array_split(img_lbp, grid, axis=0):
        for block in np.array_split(rows, grid, axis=1):
            if n_bins == 256:
                counts = np.bincount(block.ravel(), minlength=256)
            else:
                counts, _ = np.histogram(block.ravel(), bins=n_bins, range=(0, 256))
            hists.append(counts / block.size)
    return np.concatenate(hists)


def get_lbp_and_histogram(
    image_path: str | Path,
    target_size: tuple[int, int] = (256, 256),
    n_bins: int = 256,
    grid: int = 4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load an image, convert to grayscale, resize, and extract LBP map + histogram."""
    path = Path(image_path)
    if not path.is_file():
        return None

    img = cv2.imread(str(path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width, height = target_size
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    img_lbp = compute_lbp_map(resized)
    hist_feats = compute_spatial_histogram(img_lbp, grid=grid, n_bins=n_bins)
    return img_lbp, hist_feats


def _parse_relative_path(raw_line: str) -> Path:
    """Normalize Windows/POSIX separators in an index line into a Path."""
    parts = [part for part in raw_line.strip().replace("\\", "/").split("/") if part]
    return Path(*parts)


def read_indexed_samples(
    index_path: str | Path,
    images_dir: str | Path,
    label: int,
) -> list[tuple[str, Path, int]]:
    """Read image paths from an index file and pair each with its class label."""
    idx_file = Path(index_path)
    root_dir = Path(images_dir)
    if not idx_file.is_file():
        raise FileNotFoundError(f"Index file not found: {idx_file}")
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root_dir}")

    samples: list[tuple[str, Path, int]] = []
    with idx_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rel_path = _parse_relative_path(stripped)
            # Prefix with root folder name so identical relative paths in ClientRaw/ImposterRaw stay unique.
            image_id = f"{root_dir.name}/{rel_path.as_posix()}"
            samples.append((image_id, root_dir / rel_path, label))
    return samples


def build_lbp_dataframe(
    sources: Sequence[tuple[str | Path, str | Path, int]],
    target_size: tuple[int, int] = (256, 256),
    n_bins: int = 256,
    grid: int = 4,
    progress_every: int = 500,
) -> tuple[pd.DataFrame, list[Path]]:
    """Preprocess all indexed images through get_lbp_and_histogram into a DataFrame."""
    all_samples: list[tuple[str, Path, int]] = []
    for index_path, images_dir, label in sources:
        all_samples.extend(read_indexed_samples(index_path, images_dir, label))

    data: dict[str, list] = {
        "image": [],
        "hists": [],
        "type": [],
    }
    skipped: list[Path] = []
    total = len(all_samples)

    for idx, (image_id, full_path, label) in enumerate(all_samples, start=1):
        result = get_lbp_and_histogram(
            full_path,
            target_size=target_size,
            n_bins=n_bins,
            grid=grid,
        )
        if result is None:
            skipped.append(full_path)
            continue

        _, hists = result
        data["image"].append(image_id)
        data["hists"].append(hists)
        data["type"].append(label)

        if progress_every and idx % progress_every == 0:
            print(f"Processed {idx}/{total} images...")

    df = pd.DataFrame(data)
    return df, skipped


def prepare_nuaa_datasets(
    real_train_path: str | Path,
    real_test_path: str | Path,
    spoof_train_path: str | Path,
    spoof_test_path: str | Path,
    real_images_dir: str | Path,
    spoof_images_dir: str | Path,
    target_size: tuple[int, int] = (256, 256),
    n_bins: int = 256,
    grid: int = 4,
    progress_every: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[Path]]]:
    """Build train and test DataFrames for the NUAA raw client/imposter splits."""
    train_sources = [
        (real_train_path, real_images_dir, REAL_LABEL),
        (spoof_train_path, spoof_images_dir, SPOOF_LABEL),
    ]
    test_sources = [
        (real_test_path, real_images_dir, REAL_LABEL),
        (spoof_test_path, spoof_images_dir, SPOOF_LABEL),
    ]

    print("Building training split...")
    train_df, skipped_train = build_lbp_dataframe(
        train_sources,
        target_size=target_size,
        n_bins=n_bins,
        grid=grid,
        progress_every=progress_every,
    )
    print("Building test split...")
    test_df, skipped_test = build_lbp_dataframe(
        test_sources,
        target_size=target_size,
        n_bins=n_bins,
        grid=grid,
        progress_every=progress_every,
    )

    skipped = {"train": skipped_train, "test": skipped_test}
    return train_df, test_df, skipped
