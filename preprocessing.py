"""Rule-based optic-disc crop used to create ``preprocessed_data``.

The detector uses only deterministic OpenCV/NumPy operations:

    score = brightness + 0.8 * local_contrast + 0.6 * vessel_density

Each component is robustly normalized over valid optic-disc candidates. The
winning point becomes the center of a square whose side is 35% of the detected
fundus height. The square is zero-padded at image boundaries and resized to
512 x 512 without distorting the square crop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CropResult:
    center_x: float
    center_y: float
    crop_side_original_pixels: float
    fundus_height: int
    score: float


def robust_unit(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map the valid 5th—99.5th percentile range to [0, 1]."""
    selected = values[valid]
    if selected.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(selected, (5.0, 99.5))
    scale = max(float(high - low), 1e-6)
    return np.clip((values.astype(np.float32) - low) / scale, 0.0, 1.0)


def fundus_mask(rgb: np.ndarray) -> np.ndarray:
    """Return the largest connected non-black field-of-view region."""
    foreground = (rgb.max(axis=2) > 12).astype(np.uint8)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if count <= 1:
        return np.ones(rgb.shape[:2], dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def detect_fundus_bounds(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest non-black fundus bounds as (x, y, width, height)."""
    height, width = rgb.shape[:2]
    scale = min(1.0, 512.0 / max(height, width))
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    preview = cv2.resize(
        rgb, (preview_width, preview_height), interpolation=cv2.INTER_AREA
    )
    foreground = (preview.max(axis=2) > 12).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0, 0, width, height
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < preview_width * preview_height * 0.1:
        return 0, 0, width, height
    x, y, box_width, box_height = cv2.boundingRect(contour)
    scale_x, scale_y = width / preview_width, height / preview_height
    x0 = max(0, int(np.floor(x * scale_x)))
    y0 = max(0, int(np.floor(y * scale_y)))
    x1 = min(width, int(np.ceil((x + box_width) * scale_x)))
    y1 = min(height, int(np.ceil((y + box_height) * scale_y)))
    return x0, y0, x1 - x0, y1 - y0


def candidate_mask(fov: np.ndarray, window: int = 64) -> np.ndarray:
    """Exclude borders and the top/bottom 15% of the fundus."""
    radius = max(1, window // 2)
    distance = cv2.distanceTransform(fov.astype(np.uint8), cv2.DIST_L2, 5)
    valid = distance >= radius * 0.75
    ys, _ = np.where(fov)
    if ys.size:
        y0, y1 = int(ys.min()), int(ys.max())
        vertical = np.zeros_like(valid)
        start = int(y0 + 0.15 * (y1 - y0))
        stop = int(y0 + 0.85 * (y1 - y0)) + 1
        vertical[start:stop] = True
        valid &= vertical
    return valid if np.any(valid) else fov


def brightness_contrast_vessel_center(
    preview_rgb: np.ndarray,
) -> tuple[float, float, float]:
    """Locate the maximum brightness + contrast + vessel-density score."""
    fov = fundus_mask(preview_rgb)
    green = preview_rgb[:, :, 1].astype(np.float32)
    green_unit = robust_unit(green, fov)

    # At preview height 512 these are the 32, 64, 128 and 256 pixel
    # windows from the original height-1024 design, scaled by one half.
    mean16 = cv2.boxFilter(green_unit, -1, (16, 16), normalize=True)
    mean32 = cv2.boxFilter(green_unit, -1, (32, 32), normalize=True)
    mean64 = cv2.boxFilter(green_unit, -1, (64, 64), normalize=True)
    mean128 = cv2.boxFilter(green_unit, -1, (128, 128), normalize=True)
    multiscale_brightness = (mean16 + mean32 + mean64) / 3.0
    local_contrast = np.maximum(mean32 - mean128, 0.0)

    # Dark elongated vessels become bright after black-hat morphology.
    blackhat = cv2.morphologyEx(
        (green_unit * 255).astype(np.uint8),
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(np.float32)
    vessel_density = cv2.boxFilter(blackhat, -1, (96, 96), normalize=True)

    valid = candidate_mask(fov, window=64)
    score = (
        robust_unit(multiscale_brightness, valid)
        + 0.8 * robust_unit(local_contrast, valid)
        + 0.6 * robust_unit(vessel_density, valid)
    )
    restricted = np.where(valid, score, -np.inf)
    y, x = np.unravel_index(int(np.argmax(restricted)), restricted.shape)
    return float(x), float(y), float(restricted[y, x])


def crop_square_with_padding(
    rgb: np.ndarray, center_x: float, center_y: float, side: float
) -> np.ndarray:
    side_px = max(1, int(round(side)))
    left = int(round(center_x - side_px / 2))
    top = int(round(center_y - side_px / 2))
    right, bottom = left + side_px, top + side_px
    crop = np.zeros((side_px, side_px, 3), dtype=rgb.dtype)
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(rgb.shape[1], right), min(rgb.shape[0], bottom)
    if src_left < src_right and src_top < src_bottom:
        dst_left, dst_top = src_left - left, src_top - top
        crop[
            dst_top : dst_top + src_bottom - src_top,
            dst_left : dst_left + src_right - src_left,
        ] = rgb[src_top:src_bottom, src_left:src_right]
    return crop


def crop_optic_disc(
    rgb: np.ndarray,
    preview_height: int = 512,
    crop_ratio: float = 0.35,
    output_size: int = 512,
) -> tuple[np.ndarray, CropResult]:
    """Crop one RGB fundus image using the preprocessing-data rule."""
    original_height, original_width = rgb.shape[:2]
    preview_width = max(1, int(round(original_width * preview_height / original_height)))
    preview = cv2.resize(
        rgb, (preview_width, preview_height), interpolation=cv2.INTER_AREA
    )
    preview_x, preview_y, score = brightness_contrast_vessel_center(preview)
    center_x = preview_x * original_width / preview_width
    center_y = preview_y * original_height / preview_height
    fundus_height = detect_fundus_bounds(rgb)[3]
    crop_side = crop_ratio * fundus_height
    crop = crop_square_with_padding(rgb, center_x, center_y, crop_side)
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    result = CropResult(
        center_x=center_x,
        center_y=center_y,
        crop_side_original_pixels=crop_side,
        fundus_height=fundus_height,
        score=score,
    )
    return crop, result


def collect_images(inputs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in inputs:
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            images.append(path)
        elif path.is_dir():
            images.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            )
        else:
            raise FileNotFoundError(path)
    return sorted(set(item.resolve() for item in images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-height", type=int, default=512)
    parser.add_argument("--crop-ratio", type=float, default=0.35)
    parser.add_argument("--output-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args.inputs)
    if not image_paths:
        raise ValueError("No supported images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    for image_path in image_paths:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop, result = crop_optic_disc(
            rgb,
            preview_height=args.preview_height,
            crop_ratio=args.crop_ratio,
            output_size=args.output_size,
        )
        output_path = args.output_dir / f"{image_path.stem}_crop.jpg"
        ok = cv2.imwrite(
            str(output_path),
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not ok:
            raise OSError(f"Could not write {output_path}")
        metadata.append(
            {
                "source_path": str(image_path),
                "output_path": str(output_path),
                **asdict(result),
            }
        )
        print(f"saved {output_path}")
    (args.output_dir / "crop_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()