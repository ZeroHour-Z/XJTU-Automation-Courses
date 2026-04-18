from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
OUT_DIR = ROOT / "outputs"
LOWPASS_DIR = OUT_DIR / "lowpass"
HIGHPASS_DIR = OUT_DIR / "highpass"

for directory in [OUT_DIR, LOWPASS_DIR, HIGHPASS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SMOOTH_TARGETS = ["test1.pgm", "test2.tif"]
EDGE_TARGETS = ["test3_corrupt.pgm", "test4.tif"]

KERNEL_SIZES = [3, 5, 7]
GAUSSIAN_SIGMA = 1.5


def imread_gray(path: Path) -> np.ndarray:
    try:
        return np.array(Image.open(path).convert("L"), dtype=np.float64)
    except Exception:
        if path.suffix.lower() == ".tif":
            bmp_path = path.with_name(path.stem + " copy.bmp")
            if bmp_path.exists():
                return np.array(Image.open(bmp_path).convert("L"), dtype=np.float64)
        raise


def imwrite_gray(path: Path, image: np.ndarray):
    arr = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def image_stats(image: np.ndarray) -> dict:
    return {
        "min": float(image.min()),
        "max": float(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
    }


# --------------- Gaussian kernel ---------------

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    k = size // 2
    ax = np.arange(-k, k + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# --------------- 2-D convolution (zero-padding) ---------------

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i : i + kh, j : j + kw] * kernel)
    return out


# --------------- Median filter ---------------

def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    k = size // 2
    padded = np.pad(image, k, mode="edge")
    out = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.median(padded[i : i + size, j : j + size])
    return out


# --------------- High-pass filters ---------------

def unsharp_masking(image: np.ndarray, sigma: float = 1.5, amount: float = 1.0) -> np.ndarray:
    blurred = convolve2d(image, gaussian_kernel(7, sigma))
    mask = image - blurred
    return image + amount * mask


def sobel_edge(image: np.ndarray) -> np.ndarray:
    gx_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)
    gy_kernel = np.array([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]], dtype=np.float64)
    gx = convolve2d(image, gx_kernel)
    gy = convolve2d(image, gy_kernel)
    return np.sqrt(gx ** 2 + gy ** 2)


def laplace_edge(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float64)
    return np.abs(convolve2d(image, kernel))


def canny_edge(image: np.ndarray, sigma: float = 1.5,
               low_ratio: float = 0.05, high_ratio: float = 0.15) -> np.ndarray:
    smoothed = convolve2d(image, gaussian_kernel(5, sigma))

    gx_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)
    gy_kernel = np.array([[-1, -2, -1],
                           [0,  0,  0],
                           [1,  2,  1]], dtype=np.float64)
    gx = convolve2d(smoothed, gx_kernel)
    gy = convolve2d(smoothed, gy_kernel)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx) * 180.0 / np.pi
    angle[angle < 0] += 180.0

    h, w = magnitude.shape
    nms = np.zeros_like(magnitude)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j + 1], magnitude[i, j - 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i + 1, j], magnitude[i - 1, j]
            else:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                nms[i, j] = magnitude[i, j]

    high_thresh = magnitude.max() * high_ratio
    low_thresh = magnitude.max() * low_ratio
    strong = 255.0
    weak = 75.0
    result = np.zeros_like(nms)
    result[nms >= high_thresh] = strong
    result[(nms >= low_thresh) & (nms < high_thresh)] = weak

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if result[i, j] == weak:
                if np.any(result[i - 1 : i + 2, j - 1 : j + 2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result


# --------------- Visualisation helpers ---------------

def save_lowpass_comparison(name: str, original: np.ndarray,
                            gauss_results: dict, median_results: dict):
    n_sizes = len(KERNEL_SIZES)
    fig, axes = plt.subplots(3, n_sizes + 1, figsize=(4 * (n_sizes + 1), 12))

    axes[0, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    for idx, ks in enumerate(KERNEL_SIZES):
        col = idx + 1
        g_img = gauss_results[ks]
        m_img = median_results[ks]

        axes[0, col].imshow(g_img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(f"Gaussian {ks}x{ks}")
        axes[0, col].axis("off")

        axes[1, col].imshow(m_img, cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title(f"Median {ks}x{ks}")
        axes[1, col].axis("off")

        diff = np.abs(g_img - m_img)
        axes[2, col].imshow(diff, cmap="hot")
        axes[2, col].set_title(f"Diff {ks}x{ks}")
        axes[2, col].axis("off")

    fig.suptitle(f"{name} - Low-pass Filtering (σ={GAUSSIAN_SIGMA})", fontsize=14)
    fig.tight_layout()
    fig.savefig(LOWPASS_DIR / f"{Path(name).stem}_lowpass_panel.png", dpi=150)
    plt.close(fig)


def save_gaussian_kernels_figure():
    fig, axes = plt.subplots(1, len(KERNEL_SIZES), figsize=(4 * len(KERNEL_SIZES), 4))
    for idx, ks in enumerate(KERNEL_SIZES):
        kernel = gaussian_kernel(ks, GAUSSIAN_SIGMA)
        im = axes[idx].imshow(kernel, cmap="hot", interpolation="nearest")
        axes[idx].set_title(f"Gaussian {ks}x{ks}, σ={GAUSSIAN_SIGMA}")
        for i in range(ks):
            for j in range(ks):
                axes[idx].text(j, i, f"{kernel[i, j]:.3f}",
                               ha="center", va="center", fontsize=7, color="white")
        plt.colorbar(im, ax=axes[idx], fraction=0.046)
    fig.suptitle(f"Gaussian Kernels (σ={GAUSSIAN_SIGMA})", fontsize=14)
    fig.tight_layout()
    fig.savefig(LOWPASS_DIR / "gaussian_kernels.png", dpi=150)
    plt.close(fig)


def save_highpass_comparison(name: str, original: np.ndarray,
                              unsharp: np.ndarray, sobel: np.ndarray,
                              laplace: np.ndarray, canny: np.ndarray):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.clip(unsharp, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Unsharp Masking")
    axes[0, 1].axis("off")

    sobel_norm = sobel / sobel.max() * 255 if sobel.max() > 0 else sobel
    axes[0, 2].imshow(sobel_norm, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Sobel Edge")
    axes[0, 2].axis("off")

    laplace_norm = laplace / laplace.max() * 255 if laplace.max() > 0 else laplace
    axes[1, 0].imshow(laplace_norm, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Laplace Edge")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(canny, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Canny Edge")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off")

    fig.suptitle(f"{name} - High-pass Filtering", fontsize=14)
    fig.tight_layout()
    fig.savefig(HIGHPASS_DIR / f"{Path(name).stem}_highpass_panel.png", dpi=150)
    plt.close(fig)


# --------------- Main ---------------

def main():
    images = {}
    for name in SMOOTH_TARGETS + EDGE_TARGETS:
        if name not in images:
            images[name] = imread_gray(ASSET_DIR / name)

    summary = []

    summary.append("[Images]")
    for name, img in images.items():
        s = image_stats(img)
        summary.append(
            f"{name}: shape={img.shape}, min={s['min']:.0f}, max={s['max']:.0f}, "
            f"mean={s['mean']:.4f}, std={s['std']:.4f}"
        )

    # ---- Task 1 & 2: Low-pass filtering ----
    summary.append("")
    summary.append(f"[Task 1 & 2] Low-pass Filtering (Gaussian sigma={GAUSSIAN_SIGMA}, Median)")

    save_gaussian_kernels_figure()
    summary.append("Gaussian kernel visualisation -> lowpass/gaussian_kernels.png")

    for ks in KERNEL_SIZES:
        k = gaussian_kernel(ks, GAUSSIAN_SIGMA)
        summary.append(f"\nGaussian kernel {ks}x{ks} (sigma={GAUSSIAN_SIGMA}):")
        for row in k:
            summary.append("  " + "  ".join(f"{v:.6f}" for v in row))

    for name in SMOOTH_TARGETS:
        img = images[name]
        gauss_results = {}
        median_results = {}
        summary.append(f"\n{name}:")
        for ks in KERNEL_SIZES:
            k = gaussian_kernel(ks, GAUSSIAN_SIGMA)
            g_img = convolve2d(img, k)
            m_img = median_filter(img, ks)
            gauss_results[ks] = g_img
            median_results[ks] = m_img

            imwrite_gray(LOWPASS_DIR / f"{Path(name).stem}_gaussian_{ks}x{ks}.pgm", g_img)
            imwrite_gray(LOWPASS_DIR / f"{Path(name).stem}_median_{ks}x{ks}.pgm", m_img)

            gs = image_stats(g_img)
            ms = image_stats(m_img)
            summary.append(
                f"  Gaussian {ks}x{ks}: mean={gs['mean']:.4f}, std={gs['std']:.4f}"
            )
            summary.append(
                f"  Median   {ks}x{ks}: mean={ms['mean']:.4f}, std={ms['std']:.4f}"
            )

        save_lowpass_comparison(name, img, gauss_results, median_results)
        summary.append(f"  Panel -> lowpass/{Path(name).stem}_lowpass_panel.png")

    # ---- Task 3: High-pass filtering ----
    summary.append("")
    summary.append("[Task 3] High-pass Filtering (Unsharp, Sobel, Laplace, Canny)")

    for name in EDGE_TARGETS:
        img = images[name]
        summary.append(f"\n{name}:")

        unsharp = unsharp_masking(img, sigma=GAUSSIAN_SIGMA, amount=1.5)
        sobel = sobel_edge(img)
        laplace = laplace_edge(img)
        canny = canny_edge(img, sigma=GAUSSIAN_SIGMA)

        stem = Path(name).stem
        imwrite_gray(HIGHPASS_DIR / f"{stem}_unsharp.pgm", unsharp)
        imwrite_gray(HIGHPASS_DIR / f"{stem}_sobel.pgm",
                     sobel / sobel.max() * 255 if sobel.max() > 0 else sobel)
        imwrite_gray(HIGHPASS_DIR / f"{stem}_laplace.pgm",
                     laplace / laplace.max() * 255 if laplace.max() > 0 else laplace)
        imwrite_gray(HIGHPASS_DIR / f"{stem}_canny.pgm", canny)

        save_highpass_comparison(name, img, unsharp, sobel, laplace, canny)

        summary.append(f"  Unsharp: mean={image_stats(unsharp)['mean']:.4f}, std={image_stats(unsharp)['std']:.4f}")
        summary.append(f"  Sobel:   mean={image_stats(sobel)['mean']:.4f}, std={image_stats(sobel)['std']:.4f}")
        summary.append(f"  Laplace: mean={image_stats(laplace)['mean']:.4f}, std={image_stats(laplace)['std']:.4f}")
        summary.append(f"  Canny:   edges={int(np.sum(canny == 255))} pixels")
        summary.append(f"  Panel -> highpass/{stem}_highpass_panel.png")

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(summary), encoding="utf-8")

    print("HW4 processing complete.")
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    main()
