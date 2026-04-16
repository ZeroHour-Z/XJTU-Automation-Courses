from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
OUT_DIR = ROOT / "outputs"
HIST_DIR = OUT_DIR / "histograms"
EQUALIZE_DIR = OUT_DIR / "equalized"
MATCH_DIR = OUT_DIR / "matched"
LOCAL_DIR = OUT_DIR / "local_equalized"
SEG_DIR = OUT_DIR / "segmentation"

for directory in [OUT_DIR, HIST_DIR, EQUALIZE_DIR, MATCH_DIR, LOCAL_DIR, SEG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}
MATCH_REFERENCES = {
    "citywall1.bmp": "citywall.bmp",
    "citywall2.bmp": "citywall.bmp",
    "elain1.bmp": "elain.bmp",
    "elain2.bmp": "elain.bmp",
    "elain3.bmp": "elain.bmp",
    "lena1.bmp": "lena.bmp",
    "lena2.bmp": "lena.bmp",
    "lena4.bmp": "lena.bmp",
    "woman1.bmp": "woman.BMP",
    "woman2.bmp": "woman.BMP",
}
LOCAL_ENHANCE_TARGETS = ["elain.bmp", "lena.bmp"]
SEGMENT_TARGETS = ["elain.bmp", "woman.BMP"]


def iter_images():
    return sorted(
        path
        for path in ASSET_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.lower().startswith("readme")
    )


def imread_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def imwrite_gray(path: Path, image: np.ndarray):
    Image.fromarray(image.astype(np.uint8), mode="L").save(path)


def compute_histogram(image: np.ndarray) -> np.ndarray:
    return np.bincount(image.ravel(), minlength=256)


def image_stats(image: np.ndarray) -> dict:
    return {
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
    }


def save_histogram_figure(hist: np.ndarray, out_path: Path, title: str, threshold: int | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(256), hist, width=1.0, color="steelblue")
    if threshold is not None:
        ax.axvline(threshold, color="crimson", linestyle="--", linewidth=2, label=f"threshold={threshold}")
        ax.legend()
    ax.set_xlim(0, 255)
    ax.set_xlabel("Gray Level")
    ax.set_ylabel("Pixel Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def cdf_lut_from_hist(hist: np.ndarray) -> np.ndarray:
    cdf = hist.cumsum()
    nz = np.flatnonzero(hist)
    if nz.size == 0:
        return np.arange(256, dtype=np.uint8)

    cdf_min = cdf[nz[0]]
    denom = cdf[-1] - cdf_min
    if denom <= 0:
        return np.arange(256, dtype=np.uint8)

    lut = np.round((cdf - cdf_min) * 255.0 / denom)
    return np.clip(lut, 0, 255).astype(np.uint8)


def equalize_histogram(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist = compute_histogram(image)
    lut = cdf_lut_from_hist(hist)
    return lut[image], lut


def match_histogram(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_hist = compute_histogram(source).astype(np.float64)
    ref_hist = compute_histogram(reference).astype(np.float64)
    src_cdf = np.cumsum(src_hist) / source.size
    ref_cdf = np.cumsum(ref_hist) / reference.size

    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for gray_level in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[gray_level]:
            ref_idx += 1

        if ref_idx == 0:
            lut[gray_level] = 0
        else:
            prev_idx = ref_idx - 1
            if abs(ref_cdf[prev_idx] - src_cdf[gray_level]) <= abs(ref_cdf[ref_idx] - src_cdf[gray_level]):
                lut[gray_level] = prev_idx
            else:
                lut[gray_level] = ref_idx

    return lut[source], lut


def local_histogram_equalize(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    pad = window_size // 2
    padded = np.pad(image, pad, mode="edge")
    out = np.empty_like(image)
    window_pixels = window_size * window_size

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = padded[row : row + window_size, col : col + window_size]
            hist = compute_histogram(window)
            cdf = hist.cumsum()
            nz = np.flatnonzero(hist)
            if nz.size == 0:
                out[row, col] = image[row, col]
                continue

            cdf_min = cdf[nz[0]]
            denom = window_pixels - cdf_min
            if denom <= 0:
                out[row, col] = image[row, col]
            else:
                center = int(image[row, col])
                value = round((cdf[center] - cdf_min) * 255.0 / denom)
                out[row, col] = np.uint8(np.clip(value, 0, 255))

    return out


def otsu_threshold(image: np.ndarray) -> int:
    hist = compute_histogram(image).astype(np.float64)
    probability = hist / hist.sum()
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * np.arange(256))
    mu_total = mu[-1]

    sigma_between = (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return int(np.nanargmax(sigma_between))


def segment_image(image: np.ndarray, threshold: int) -> np.ndarray:
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def save_equalization_panel(name: str, src: np.ndarray, dst: np.ndarray):
    src_hist = compute_histogram(src)
    dst_hist = compute_histogram(dst)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(src, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"{name} Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(dst, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"{name} Equalized")
    axes[0, 1].axis("off")

    axes[1, 0].bar(np.arange(256), src_hist, width=1.0, color="gray")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_title("Original Histogram")

    axes[1, 1].bar(np.arange(256), dst_hist, width=1.0, color="gray")
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_title("Equalized Histogram")

    fig.tight_layout()
    fig.savefig(EQUALIZE_DIR / f"{Path(name).stem}_equalization_panel.png", dpi=180)
    plt.close(fig)


def save_matching_panel(name: str, src: np.ndarray, ref: np.ndarray, matched: np.ndarray, ref_name: str):
    src_hist = compute_histogram(src)
    ref_hist = compute_histogram(ref)
    matched_hist = compute_histogram(matched)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(src, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"{name} Source")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(matched, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Matched to {ref_name}")
    axes[0, 1].axis("off")

    axes[1, 0].plot(src_hist, label="source", color="tab:blue")
    axes[1, 0].plot(ref_hist, label="reference", color="tab:orange")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_title("Source vs Reference Histogram")
    axes[1, 0].legend()

    axes[1, 1].plot(matched_hist, label="matched", color="tab:green")
    axes[1, 1].plot(ref_hist, label="reference", color="tab:orange")
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_title("Matched vs Reference Histogram")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(MATCH_DIR / f"{Path(name).stem}_matched_panel.png", dpi=180)
    plt.close(fig)


def save_local_panel(name: str, src: np.ndarray, dst: np.ndarray, window_size: int):
    src_hist = compute_histogram(src)
    dst_hist = compute_histogram(dst)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(src, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"{name} Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(dst, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"{name} Local Equalized ({window_size}x{window_size})")
    axes[0, 1].axis("off")

    axes[1, 0].bar(np.arange(256), src_hist, width=1.0, color="gray")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_title("Original Histogram")

    axes[1, 1].bar(np.arange(256), dst_hist, width=1.0, color="gray")
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_title("Local Equalized Histogram")

    fig.tight_layout()
    fig.savefig(LOCAL_DIR / f"{Path(name).stem}_local_equalized_panel.png", dpi=180)
    plt.close(fig)


def save_segmentation_panel(name: str, src: np.ndarray, mask: np.ndarray, threshold: int):
    hist = compute_histogram(src)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(src, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"{name} Original")
    axes[0].axis("off")

    axes[1].bar(np.arange(256), hist, width=1.0, color="steelblue")
    axes[1].axvline(threshold, color="crimson", linestyle="--", linewidth=2)
    axes[1].set_xlim(0, 255)
    axes[1].set_title(f"Histogram (t={threshold})")

    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Segmentation Mask")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(SEG_DIR / f"{Path(name).stem}_segmentation_panel.png", dpi=180)
    plt.close(fig)


def main():
    images = {path.name: imread_gray(path) for path in iter_images()}
    summary_lines = []

    summary_lines.append("[Images]")
    for name, image in images.items():
        stats = image_stats(image)
        summary_lines.append(
            f"{name}: shape={image.shape}, min={stats['min']}, max={stats['max']}, "
            f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
        )

    summary_lines.append("")
    summary_lines.append("[Task 1] Histogram Figures")
    for name, image in images.items():
        hist = compute_histogram(image)
        save_histogram_figure(hist, HIST_DIR / f"{Path(name).stem}_hist.png", f"{name} Histogram")
        summary_lines.append(f"{name} -> histograms/{Path(name).stem}_hist.png")

    summary_lines.append("")
    summary_lines.append("[Task 2] Histogram Equalization")
    for name, image in images.items():
        equalized, _ = equalize_histogram(image)
        out_name = f"{Path(name).stem}_equalized.bmp"
        imwrite_gray(EQUALIZE_DIR / out_name, equalized)
        save_equalization_panel(name, image, equalized)
        src_stats = image_stats(image)
        dst_stats = image_stats(equalized)
        summary_lines.append(
            f"{name}: std {src_stats['std']:.4f} -> {dst_stats['std']:.4f}, "
            f"range {src_stats['min']}-{src_stats['max']} -> {dst_stats['min']}-{dst_stats['max']}"
        )

    summary_lines.append("")
    summary_lines.append("[Task 3] Histogram Matching")
    for source_name, reference_name in MATCH_REFERENCES.items():
        source_image = images[source_name]
        reference_image = images[reference_name]
        matched, _ = match_histogram(source_image, reference_image)
        out_name = f"{Path(source_name).stem}_matched_to_{Path(reference_name).stem}.bmp"
        imwrite_gray(MATCH_DIR / out_name, matched)
        save_matching_panel(source_name, source_image, reference_image, matched, reference_name)
        src_stats = image_stats(source_image)
        dst_stats = image_stats(matched)
        ref_stats = image_stats(reference_image)
        summary_lines.append(
            f"{source_name} -> {reference_name}: mean {src_stats['mean']:.4f} -> {dst_stats['mean']:.4f}, "
            f"reference mean {ref_stats['mean']:.4f}"
        )

    summary_lines.append("")
    summary_lines.append("[Task 4] Local Histogram Equalization (7x7)")
    for name in LOCAL_ENHANCE_TARGETS:
        image = images[name]
        local_equalized = local_histogram_equalize(image, window_size=7)
        out_name = f"{Path(name).stem}_local_equalized_7x7.bmp"
        imwrite_gray(LOCAL_DIR / out_name, local_equalized)
        save_local_panel(name, image, local_equalized, window_size=7)
        src_stats = image_stats(image)
        dst_stats = image_stats(local_equalized)
        summary_lines.append(
            f"{name}: std {src_stats['std']:.4f} -> {dst_stats['std']:.4f}, "
            f"mean {src_stats['mean']:.4f} -> {dst_stats['mean']:.4f}"
        )

    summary_lines.append("")
    summary_lines.append("[Task 5] Histogram-Based Segmentation")
    for name in SEGMENT_TARGETS:
        image = images[name]
        threshold = otsu_threshold(image)
        mask = segment_image(image, threshold)
        out_name = f"{Path(name).stem}_segmented_otsu.bmp"
        imwrite_gray(SEG_DIR / out_name, mask)
        save_histogram_figure(
            compute_histogram(image),
            SEG_DIR / f"{Path(name).stem}_hist_with_threshold.png",
            f"{name} Histogram",
            threshold=threshold,
        )
        save_segmentation_panel(name, image, mask, threshold)
        white_ratio = float(mask.mean() / 255.0)
        summary_lines.append(f"{name}: threshold={threshold}, foreground_ratio={white_ratio:.4f}")

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("HW3 processing complete.")
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    main()
