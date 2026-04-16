from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
OUT_DIR = ROOT / "outputs"
LOWPASS_DIR = OUT_DIR / "lowpass_freq"
HIGHPASS_DIR = OUT_DIR / "highpass_freq"
SPATIAL_HIGHPASS_DIR = OUT_DIR / "highpass_spatial"
SPECTRUM_DIR = OUT_DIR / "spectrums"
COMPARE_DIR = OUT_DIR / "comparison"

for directory in [OUT_DIR, LOWPASS_DIR, HIGHPASS_DIR, SPATIAL_HIGHPASS_DIR, SPECTRUM_DIR, COMPARE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LOWPASS_TARGETS = ["test1.pgm", "test2.tif"]
HIGHPASS_TARGETS = ["test3_corrupt.pgm", "test4.tif"]
BUTTERWORTH_ORDER = 2


def imread_gray(path: Path) -> np.ndarray:
    try:
        return np.array(Image.open(path).convert("L"), dtype=np.float64)
    except Exception:
        if path.suffix.lower() == ".tif":
            fallback = path.with_name(path.stem + " copy.bmp")
            if fallback.exists():
                return np.array(Image.open(fallback).convert("L"), dtype=np.float64)
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


def centered_fft(image: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(image))


def centered_ifft(freq: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.ifftshift(freq)))


def distance_grid(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    vv, uu = np.meshgrid(v, u)
    return np.sqrt(uu * uu + vv * vv)


def gaussian_lowpass(shape: tuple[int, int], radius: float) -> np.ndarray:
    d = distance_grid(shape)
    h = np.exp(-(d * d) / (2.0 * radius * radius))
    return np.clip(h, 0.0, 1.0)


def butterworth_lowpass(shape: tuple[int, int], radius: float, order: int = 2) -> np.ndarray:
    d = distance_grid(shape)
    d = np.maximum(d, 1e-12)
    h = 1.0 / (1.0 + (d / radius) ** (2 * order))
    return np.clip(h, 0.0, 1.0)


def gaussian_highpass(shape: tuple[int, int], radius: float) -> np.ndarray:
    return 1.0 - gaussian_lowpass(shape, radius)


def butterworth_highpass(shape: tuple[int, int], radius: float, order: int = 2) -> np.ndarray:
    return 1.0 - butterworth_lowpass(shape, radius, order)


def apply_frequency_filter(image: np.ndarray, h_filter: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freq = centered_fft(image)
    filtered_freq = freq * h_filter
    filtered = centered_ifft(filtered_freq)
    return filtered, freq, filtered_freq


def power_spectrum_ratio(freq: np.ndarray, filtered_freq: np.ndarray) -> float:
    all_power = float(np.sum(np.abs(freq) ** 2))
    kept_power = float(np.sum(np.abs(filtered_freq) ** 2))
    if all_power <= 1e-12:
        return 0.0
    return kept_power / all_power


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    dmin = float(data.min())
    dmax = float(data.max())
    if dmax - dmin <= 1e-12:
        return np.zeros_like(data, dtype=np.uint8)
    out = (data - dmin) * 255.0 / (dmax - dmin)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_spectrum(path: Path, freq: np.ndarray):
    magnitude = np.log1p(np.abs(freq))
    imwrite_gray(path, normalize_to_uint8(magnitude))


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    k = size // 2
    axis = np.arange(-k, k + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def conv2d_edge(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def laplace_filter(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    response = conv2d_edge(image, kernel)
    return np.abs(response)


def unsharp_mask(image: np.ndarray, sigma: float = 1.2, amount: float = 1.5) -> np.ndarray:
    blur = conv2d_edge(image, gaussian_kernel(7, sigma))
    return image + amount * (image - blur)


def save_filter_panel(
    name: str,
    original: np.ndarray,
    g_img: np.ndarray,
    b_img: np.ndarray,
    g_ratio: float,
    b_ratio: float,
    out_dir: Path,
    title_prefix: str,
):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(g_img, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Gaussian ({g_ratio:.4f})")
    axes[1].axis("off")

    axes[2].imshow(np.clip(b_img, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Butterworth ({b_ratio:.4f})")
    axes[2].axis("off")

    fig.suptitle(f"{name} - {title_prefix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(name).stem}_panel.png", dpi=160)
    plt.close(fig)


def save_spatial_highpass_panel(name: str, original: np.ndarray, laplace: np.ndarray, unsharp: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    laplace_view = normalize_to_uint8(laplace)
    axes[1].imshow(laplace_view, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Laplace")
    axes[1].axis("off")

    axes[2].imshow(np.clip(unsharp, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Unsharp")
    axes[2].axis("off")

    fig.suptitle(f"{name} - Spatial High-pass", fontsize=13)
    fig.tight_layout()
    fig.savefig(SPATIAL_HIGHPASS_DIR / f"{Path(name).stem}_panel.png", dpi=160)
    plt.close(fig)


def save_equivalence_panel(name: str, src: np.ndarray, low_freq: np.ndarray, high_freq: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(src, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(low_freq, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Freq Low-pass")
    axes[1].axis("off")

    axes[2].imshow(np.clip(high_freq, 0, 255), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Freq High-pass")
    axes[2].axis("off")

    fig.suptitle(f"{name} - Frequency Domain Results", fontsize=13)
    fig.tight_layout()
    fig.savefig(COMPARE_DIR / f"{Path(name).stem}_freq_compare_panel.png", dpi=160)
    plt.close(fig)


def choose_radius(shape: tuple[int, int], mode: str) -> float:
    base = min(shape)
    if mode == "lowpass":
        return max(15.0, base / 7.0)
    return max(12.0, base / 9.0)


def main():
    report = []
    report.append("[Inputs]")

    images = {}
    for name in LOWPASS_TARGETS + HIGHPASS_TARGETS:
        image = imread_gray(ASSET_DIR / name)
        images[name] = image
        stat = image_stats(image)
        report.append(
            f"{name}: shape={image.shape}, min={stat['min']:.0f}, max={stat['max']:.0f}, "
            f"mean={stat['mean']:.4f}, std={stat['std']:.4f}"
        )

    report.append("")
    report.append("[Task 1] Frequency Low-pass (Gaussian / Butterworth)")

    for name in LOWPASS_TARGETS:
        src = images[name]
        radius = choose_radius(src.shape, "lowpass")

        g_h = gaussian_lowpass(src.shape, radius)
        b_h = butterworth_lowpass(src.shape, radius, BUTTERWORTH_ORDER)

        g_img, freq, g_freq = apply_frequency_filter(src, g_h)
        b_img, _, b_freq = apply_frequency_filter(src, b_h)

        g_ratio = power_spectrum_ratio(freq, g_freq)
        b_ratio = power_spectrum_ratio(freq, b_freq)

        stem = Path(name).stem
        imwrite_gray(LOWPASS_DIR / f"{stem}_gaussian_freq.pgm", g_img)
        imwrite_gray(LOWPASS_DIR / f"{stem}_butterworth_freq.pgm", b_img)

        save_spectrum(SPECTRUM_DIR / f"{stem}_origin_spectrum.pgm", freq)
        save_spectrum(SPECTRUM_DIR / f"{stem}_gaussian_lowpass_spectrum.pgm", g_freq)
        save_spectrum(SPECTRUM_DIR / f"{stem}_butterworth_lowpass_spectrum.pgm", b_freq)
        save_filter_panel(name, src, g_img, b_img, g_ratio, b_ratio, LOWPASS_DIR, f"Freq Low-pass r={radius:.1f}")
        save_equivalence_panel(name, src, g_img, src - g_img)

        report.append(
            f"{name}: radius={radius:.2f}, gaussian_psr={g_ratio:.6f}, butterworth_psr={b_ratio:.6f}"
        )

    report.append("")
    report.append("[Task 2] Frequency High-pass (Gaussian / Butterworth)")

    for name in HIGHPASS_TARGETS:
        src = images[name]
        radius = choose_radius(src.shape, "highpass")

        g_h = gaussian_highpass(src.shape, radius)
        b_h = butterworth_highpass(src.shape, radius, BUTTERWORTH_ORDER)

        g_img, freq, g_freq = apply_frequency_filter(src, g_h)
        b_img, _, b_freq = apply_frequency_filter(src, b_h)

        g_ratio = power_spectrum_ratio(freq, g_freq)
        b_ratio = power_spectrum_ratio(freq, b_freq)

        stem = Path(name).stem
        imwrite_gray(HIGHPASS_DIR / f"{stem}_gaussian_freq.pgm", normalize_to_uint8(g_img))
        imwrite_gray(HIGHPASS_DIR / f"{stem}_butterworth_freq.pgm", normalize_to_uint8(b_img))

        save_spectrum(SPECTRUM_DIR / f"{stem}_gaussian_highpass_spectrum.pgm", g_freq)
        save_spectrum(SPECTRUM_DIR / f"{stem}_butterworth_highpass_spectrum.pgm", b_freq)
        save_filter_panel(
            name,
            src,
            normalize_to_uint8(g_img),
            normalize_to_uint8(b_img),
            g_ratio,
            b_ratio,
            HIGHPASS_DIR,
            f"Freq High-pass r={radius:.1f}",
        )

        report.append(
            f"{name}: radius={radius:.2f}, gaussian_psr={g_ratio:.6f}, butterworth_psr={b_ratio:.6f}"
        )

    report.append("")
    report.append("[Task 3] Spatial High-pass (Laplace / Unsharp)")

    for name in HIGHPASS_TARGETS:
        src = images[name]
        laplace = laplace_filter(src)
        unsharp = unsharp_mask(src, sigma=1.2, amount=1.5)

        stem = Path(name).stem
        imwrite_gray(SPATIAL_HIGHPASS_DIR / f"{stem}_laplace.pgm", normalize_to_uint8(laplace))
        imwrite_gray(SPATIAL_HIGHPASS_DIR / f"{stem}_unsharp.pgm", unsharp)
        save_spatial_highpass_panel(name, src, laplace, unsharp)

        l_stat = image_stats(laplace)
        u_stat = image_stats(unsharp)
        report.append(
            f"{name}: laplace_std={l_stat['std']:.4f}, unsharp_std={u_stat['std']:.4f}, "
            f"unsharp_mean={u_stat['mean']:.4f}"
        )

    report.append("")
    report.append("[Task 4] Relation Discussion Summary")
    report.append("1) 频域低通抑制高频，对应空域卷积平滑；频域高通抑制低频，对应空域微分或细节增强。")
    report.append("2) 在理想线性平移不变条件下，频域乘法与空域卷积等效；有限尺寸、边界处理和离散采样会导致非完全一致。")
    report.append("3) Gaussian 在频域与空域同为 Gaussian，等效性最好；Butterworth 在空域等效核无限支撑且可能振铃。")

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("HW5 processing complete.")
    print(f"Summary saved to: {report_path}")


if __name__ == "__main__":
    main()
