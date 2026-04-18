from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ASSET_DIR_CANDIDATES = ROOT / "assets"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# imdecode/tofile
def cv_imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def cv_imwrite_unicode(path: Path, image: np.ndarray):
    ext = path.suffix if path.suffix else ".bmp"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def resolve_asset_dir() -> Path:
    required_files = {"7.bmp", "lena.bmp", "elain1.bmp"}
    if ASSET_DIR_CANDIDATES.exists():
        if required_files.issubset({item.name for item in ASSET_DIR_CANDIDATES.iterdir()}):
            return ASSET_DIR_CANDIDATES
    raise FileNotFoundError(
        "Cannot find 7.bmp, lena.bmp, and elain1.bmp in assets/, assets/, or the script directory"
    )

def parse_bmp_header(path: Path):
    with open(path, "rb") as f:
        data = f.read(54)

    file_header = {
        "bfType": data[0:2].decode("ascii", errors="ignore"),
        "bfSize": int.from_bytes(data[2:6], "little"),
        "bfReserved1": int.from_bytes(data[6:8], "little"),
        "bfReserved2": int.from_bytes(data[8:10], "little"),
        "bfOffBits": int.from_bytes(data[10:14], "little"),
    }

    info_header = {
        "biSize": int.from_bytes(data[14:18], "little"),
        "biWidth": int.from_bytes(data[18:22], "little", signed=True),
        "biHeight": int.from_bytes(data[22:26], "little", signed=True),
        "biPlanes": int.from_bytes(data[26:28], "little"),
        "biBitCount": int.from_bytes(data[28:30], "little"),
        "biCompression": int.from_bytes(data[30:34], "little"),
        "biSizeImage": int.from_bytes(data[34:38], "little"),
        "biXPelsPerMeter": int.from_bytes(data[38:42], "little", signed=True),
        "biYPelsPerMeter": int.from_bytes(data[42:46], "little", signed=True),
        "biClrUsed": int.from_bytes(data[46:50], "little"),
        "biClrImportant": int.from_bytes(data[50:54], "little"),
    }
    return file_header, info_header

def describe_bmp_layout(info_header: dict):
    width = info_header["biWidth"]
    height = abs(info_header["biHeight"])
    bit_count = info_header["biBitCount"]
    bytes_per_pixel = max(bit_count // 8, 1)
    # bmp 每一行数据都按 4 字节边界对齐存储
    row_stride = ((width * bytes_per_pixel) + 3) // 4 * 4
    pixel_array_size = row_stride * height
    return {
        "width": width,
        "height": height,
        "bit_count": bit_count,
        "row_stride": row_stride,
        "pixel_array_size": pixel_array_size,
    }

def quantize_levels(img_gray: np.ndarray, levels: int) -> np.ndarray:
    if levels <= 1:
        return np.zeros_like(img_gray)

    # 将 [0, 255] 的灰度范围均匀量化到指定的灰度级数
    step = 256 / levels
    q = np.floor(img_gray / step)
    q = np.clip(q, 0, levels - 1)
    out = (q * (255 / (levels - 1))).astype(np.uint8)
    return out


def save_gray_reduction_panel(img_gray: np.ndarray):
    levels_list = list(range(8, 0, -1))
    panels = [quantize_levels(img_gray, lv) for lv in levels_list]

    # 将 8 到 1 级灰度的量化结果拼成一张图
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, lv in enumerate(levels_list):
        ax = axes[i // 4, i % 4]
        ax.imshow(panels[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{lv} levels")
        ax.axis("off")

    fig.suptitle("Lena Gray Level Reduction: 8 -> 1", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lena_gray_levels_8_to_1.png", dpi=200)
    plt.close(fig)

INTERPOLATION_METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}

def resize_and_save(img: np.ndarray, prefix: str):
    for name, interp in INTERPOLATION_METHODS.items():
        out = cv2.resize(img, (2048, 2048), interpolation=interp)
        cv_imwrite_unicode(OUT_DIR / f"{prefix}_{name}_2048.bmp", out)


def shear_horizontal(img: np.ndarray, shear_k: float, interpolation: int):
    h, w = img.shape[:2]
    new_w = int(w + abs(shear_k) * h)
    m = np.float32([[1, shear_k, 0], [0, 1, 0]])
    return cv2.warpAffine(img, m, (new_w, h), flags=interpolation, borderValue=0)


def rotate_image(img: np.ndarray, angle_deg: float, interpolation: int):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, m, (new_w, new_h), flags=interpolation, borderValue=0)


def transform_and_resize(img: np.ndarray, prefix: str, transform_fn):
    for name, interp in INTERPOLATION_METHODS.items():
        # 几何变换和最终缩放使用同一种插值方法
        transformed = transform_fn(img, interp)
        resized = cv2.resize(transformed, (2048, 2048), interpolation=interp)
        cv_imwrite_unicode(OUT_DIR / f"{prefix}_{name}_2048.bmp", resized)


def main():
    asset_dir = resolve_asset_dir()
    bmp7 = asset_dir / "7.bmp"
    lena_path = asset_dir / "lena.bmp"
    elain_path = asset_dir / "elain1.bmp"

    fh, ih = parse_bmp_header(bmp7)
    bmp_layout = describe_bmp_layout(ih)

    lena_gray = cv_imread_unicode(lena_path, cv2.IMREAD_GRAYSCALE)
    if lena_gray is None:
        raise FileNotFoundError("Cannot read lena.bmp")

    lena_color = cv_imread_unicode(lena_path, cv2.IMREAD_COLOR)
    elain_color = cv_imread_unicode(elain_path, cv2.IMREAD_COLOR)
    if lena_color is None or elain_color is None:
        raise FileNotFoundError("Cannot read lena.bmp or elain1.bmp")

    save_gray_reduction_panel(lena_gray)

    mean_val = float(np.mean(lena_gray))
    var_val = float(np.var(lena_gray))

    resize_and_save(lena_gray, "lena_resize")

    shear_k = 1.5
    angle = 30.0

    # 分别为 lena 和 elain 生成 shear 与旋转后的全部结果图
    transform_and_resize(lena_color, "lena_shear", lambda image, interp: shear_horizontal(image, shear_k, interp))
    transform_and_resize(lena_color, "lena_rotate30", lambda image, interp: rotate_image(image, angle, interp))
    transform_and_resize(elain_color, "elain_shear", lambda image, interp: shear_horizontal(image, shear_k, interp))
    transform_and_resize(elain_color, "elain_rotate30", lambda image, interp: rotate_image(image, angle, interp))

    # 额外保存一份纯文本报告，便于在 Markdown 作业中直接引用结果
    report = []
    report.append(f"[Asset Directory]\n{asset_dir}")
    report.append("")
    report.append("[7.bmp Header Info]")
    for k, v in fh.items():
        report.append(f"{k}: {v}")
    for k, v in ih.items():
        report.append(f"{k}: {v}")
    report.append("")
    report.append("[7.bmp Layout Analysis]")
    report.append(f"width: {bmp_layout['width']}")
    report.append(f"height: {bmp_layout['height']}")
    report.append(f"bit_count: {bmp_layout['bit_count']}")
    report.append(f"row_stride_bytes: {bmp_layout['row_stride']}")
    report.append(f"pixel_array_size: {bmp_layout['pixel_array_size']}")

    report.append("")
    report.append("[Lena Statistics]")
    report.append(f"mean: {mean_val:.6f}")
    report.append(f"variance: {var_val:.6f}")
    report.append("")
    report.append("[Task Parameters]")
    report.append(f"shear_k: {shear_k}")
    report.append(f"rotation_angle_deg: {angle}")
    report.append("zoom_size: 2048x2048")

    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("Processing complete.")
    print(f"mean={mean_val:.6f}, variance={var_val:.6f}")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
