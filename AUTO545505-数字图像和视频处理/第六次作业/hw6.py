from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RNG_SEED = 2026

TASK1_GAUSSIAN_MEAN = 0.0
TASK1_GAUSSIAN_VAR = 400.0

TASK2_SALT_PROB = 0.1
TASK2_PEPPER_PROB = 0.1

TASK3_MOTION_ANGLE = 45.0
TASK3_MOTION_T = 1.0
TASK3_NOISE_MEAN = 0.0
TASK3_NOISE_VAR = 10.0

TASK1_DIR = OUT_DIR / "task1_gaussian_noise"
TASK2_DIR = OUT_DIR / "task2_salt_pepper"
TASK3_DIR = OUT_DIR / "task3_wiener_filter"

for d in [TASK1_DIR, TASK2_DIR, TASK3_DIR]:
    d.mkdir(exist_ok=True)


def cv_imread_unicode(path: Path, flags=cv2.IMREAD_GRAYSCALE):
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


def add_gaussian_noise(image: np.ndarray, mean: float, var: float, rng: np.random.Generator) -> np.ndarray:
    """添加高斯噪声"""
    sigma = np.sqrt(var)
    noise = rng.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray,
    salt_prob: float,
    pepper_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """添加椒盐噪声"""
    noisy = image.copy().astype(np.float64)
    total_pixels = image.size

    # Salt (white) noise
    num_salt = int(total_pixels * salt_prob)
    salt_coords = [rng.integers(0, i, num_salt) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255

    # Pepper (black) noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [rng.integers(0, i, num_pepper) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    return noisy.astype(np.uint8)


def arithmetic_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """算术均值滤波器"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    result = cv2.filter2D(image.astype(np.float64), -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(result, 0, 255).astype(np.uint8)


def geometric_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """几何均值滤波器"""
    img = image.astype(np.float64) + 1e-10  # 避免log(0)
    log_img = np.log(img)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    log_result = cv2.filter2D(log_img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    result = np.exp(log_result)
    return np.clip(result, 0, 255).astype(np.uint8)


def harmonic_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """谐波均值滤波器"""
    img = image.astype(np.float64)
    reciprocal = 1.0 / (img + 1e-10)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    recip_result = cv2.filter2D(reciprocal, -1, kernel, borderType=cv2.BORDER_REFLECT)
    result = 1.0 / (recip_result + 1e-10)
    return np.clip(result, 0, 255).astype(np.uint8)


def contraharmonic_mean_filter(image: np.ndarray, Q: float, kernel_size: int = 3) -> np.ndarray:
    """反谐波均值滤波器"""
    img = image.astype(np.float64)
    # 避免零值问题
    img = np.clip(img, 1e-10, 255)

    if abs(Q) < 0.01:
        # Q接近0时退化为算术均值
        return arithmetic_mean_filter(image, kernel_size)

    numerator = img ** (Q + 1)
    denominator = img ** Q

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    num_sum = cv2.filter2D(numerator, -1, kernel, borderType=cv2.BORDER_REFLECT)
    den_sum = cv2.filter2D(denominator, -1, kernel, borderType=cv2.BORDER_REFLECT)

    result = num_sum / (den_sum + 1e-10)
    return np.clip(result, 0, 255).astype(np.uint8)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """中值滤波器"""
    return cv2.medianBlur(image, kernel_size)


def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """计算PSNR"""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def save_comparison_panel(title: str, images: dict, save_path: Path, psnrs: dict = None):
    """保存对比图"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (name, img) in enumerate(images.items()):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        if psnrs and name in psnrs:
            axes[i].set_title(f"{name}\nPSNR: {psnrs[name]:.2f} dB")
        else:
            axes[i].set_title(name)
        axes[i].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ========================== Task 3: Wiener Filter ==========================

def motion_blur_psf(size: tuple, angle: float, length: int = 15) -> np.ndarray:
    """
    创建运动模糊的点扩散函数(PSF) - 空域
    根据 Gonzalez 描述：在指定角度方向的直线运动
    """
    psf = np.zeros(size, dtype=np.float64)
    center = (size[0] // 2, size[1] // 2)

    theta = np.deg2rad(angle)

    # 在运动方向上创建线段
    for i in range(-length//2, length//2 + 1):
        x = int(center[0] + i * np.cos(theta))
        y = int(center[1] + i * np.sin(theta))
        if 0 <= x < size[0] and 0 <= y < size[1]:
            psf[x, y] = 1.0

    # 归一化
    psf_sum = np.sum(psf)
    if psf_sum > 0:
        psf /= psf_sum

    return psf


def motion_blur_kernel_freq(shape: tuple, angle: float, T: float) -> np.ndarray:
    """
    运动模糊滤波器 H(u,v) - 频域表示
    根据 Gonzalez 4th Ed. Eq. (5.6-11)
    H(u,v) = sin(pi*D) / (pi*D) * exp(-j*pi*D)
    其中 D = u*a + v*b, a = T*cos(theta), b = T*sin(theta)
    """
    M, N = shape
    u = np.fft.fftfreq(M).reshape(-1, 1) * M
    v = np.fft.fftfreq(N).reshape(1, -1) * N

    theta = np.deg2rad(angle)
    a = T * np.cos(theta)
    b = T * np.sin(theta)

    # D = a*u + b*v
    D = a * u + b * v

    # H = sinc(D) * exp(-j*pi*D) = sin(pi*D)/(pi*D) * exp(-j*pi*D)
    # 当 D=0 时，H=1
    H = np.ones_like(D, dtype=np.complex128)
    nonzero = np.abs(D) > 1e-10
    pi_D = np.pi * D[nonzero]
    H[nonzero] = np.sin(pi_D) / pi_D * np.exp(-1j * pi_D)

    return H


def apply_motion_blur(image: np.ndarray, angle: float, T: float) -> np.ndarray:
    """应用运动模糊 - 使用频域方法"""
    H = motion_blur_kernel_freq(image.shape, angle, T)
    F = fft2(image)
    G = F * H
    blurred = np.real(ifft2(G))
    return np.clip(blurred, 0, 255).astype(np.uint8)


def wiener_filter_simple(degraded: np.ndarray, H: np.ndarray, K: float = 0.01) -> np.ndarray:
    """
    简化的维纳滤波器 (Eq. 5.9-4)
    F_hat(u,v) = [1/H(u,v)] * [|H(u,v)|^2 / (|H(u,v)|^2 + K)] * G(u,v)
    """
    G = fft2(degraded)

    H_mag_sq = np.abs(H) ** 2
    # 维纳滤波器
    W = np.zeros_like(H, dtype=np.complex128)
    nonzero = np.abs(H) > 1e-10
    W[nonzero] = (H_mag_sq[nonzero] / (H_mag_sq[nonzero] + K)) / H[nonzero]

    F_hat = W * G
    restored = np.real(ifft2(F_hat))
    return np.clip(restored, 0, 255)


def wiener_filter_full(
    degraded: np.ndarray,
    H: np.ndarray,
    noise_psd: np.ndarray,
    signal_psd: np.ndarray,
) -> np.ndarray:
    """
    完整维纳滤波器 (Eq. 5.8-6)
    F_hat(u,v) = [H*(u,v) / (|H(u,v)|^2 + Sn(u,v)/Sf(u,v))] * G(u,v)
    这里按频率点使用噪声与信号功率谱比，而不是退化为常数 K。
    """
    G = fft2(degraded)

    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2
    noise_signal_ratio = noise_psd / (signal_psd + 1e-10)

    W = H_conj / (H_mag_sq + noise_signal_ratio)

    F_hat = W * G
    restored = np.real(ifft2(F_hat))
    return np.clip(restored, 0, 255)


def inverse_filter(degraded: np.ndarray, H: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    逆滤波器 (Eq. 5.8-6 的简化形式)
    F_hat(u,v) = G(u,v) / H(u,v)
    带阈值限制以避免高频噪声放大
    """
    G = fft2(degraded)

    # 限制逆滤波器响应
    H_inv = np.where(np.abs(H) > threshold, 1.0 / (H + 1e-10), 0)

    F_hat = H_inv * G
    restored = np.real(ifft2(F_hat))
    return np.clip(restored, 0, 255)


def compute_psd(image: np.ndarray) -> np.ndarray:
    """计算功率谱密度"""
    F = fftshift(fft2(image))
    return np.abs(F) ** 2


def format_psnr(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return f"{value:.2f}"


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def top_psnr_items(items: dict, count: int = 3) -> list[tuple[str, float]]:
    ranked = sorted(items.items(), key=lambda item: item[1], reverse=True)
    return ranked[:count]


def build_reports(image: np.ndarray, task1: dict, task2: dict, task3: dict) -> tuple[str, str]:
    image_mean = float(np.mean(image))
    image_std = float(np.std(image))
    best_task1 = max(task1["filter_psnrs"], key=task1["filter_psnrs"].get)
    best_task2 = max(task2["filter_psnrs"], key=task2["filter_psnrs"].get)
    best_task3 = max(task3["restoration_psnrs"], key=task3["restoration_psnrs"].get)

    report_lines = [
        "数字图像和视频处理-第六次作业实验报告",
        "班级：自动化2305",
        "姓名：周湛昊",
        "学号：2233712088",
        "",
        "一、实验内容",
        "1. 在测试图像上产生高斯噪声 lena 图，指定均值和方差，并用多种滤波器恢复图像，分析各自优缺点。",
        "2. 在测试图像 lena 图加入椒盐噪声，椒和盐噪声密度均为 0.1，用学过的滤波器恢复图像，并分析反谐波滤波中 Q 大于 0 和小于 0 的作用。",
        "3. 推导维纳滤波器并完成以下内容：",
        "   (a) 实现模糊滤波器，如方程 Eq. (5.6-11)。",
        "   (b) 对 lena 图像施加 45 度方向、T=1 的运动模糊。",
        "   (c) 在模糊图像中加入均值为 0、方差为 10 的高斯噪声。",
        "   (d) 分别利用方程 Eq. (5.8-6) 和 Eq. (5.9-4) 恢复图像，并分析算法优缺点。",
        "主程序文件为 hw6.py，输入图像位于 assets 文件夹，结果输出到 outputs 文件夹。",
        "",
        "二、实验环境",
        "- Python 3.11",
        "- NumPy",
        "- OpenCV",
        "- Matplotlib",
        "- SciPy",
        "- Pillow",
        "",
        "三、实验过程与结果分析",
        "",
        "1. 高斯噪声图像恢复",
        f"输入图像：lena.bmp，尺寸={image.shape}，均值={image_mean:.2f}，标准差={image_std:.2f}",
        f"噪声参数：均值={format_number(task1['noise_mean'])}，方差={format_number(task1['noise_var'])}，加噪后 PSNR={task1['psnr_noisy']:.2f} dB",
        "各滤波器 PSNR：",
    ]

    for name, value in sorted(task1["filter_psnrs"].items(), key=lambda item: item[1], reverse=True):
        report_lines.append(f"- {name}: {value:.2f} dB")

    report_lines.extend(
        [
            f"分析：本次高斯噪声恢复中，{best_task1} 的 PSNR 最高，为 {task1['filter_psnrs'][best_task1]:.2f} dB。"
            "算术均值和几何均值都能平滑随机噪声，但会损失部分边缘细节；中值滤波更擅长抑制脉冲噪声，因此在该任务中通常不占优。",
            "结果图：outputs/task1_gaussian_noise/comparison_panel.png",
            "",
            "2. 椒盐噪声图像恢复",
            f"噪声参数：salt={format_number(task2['salt_prob'])}，pepper={format_number(task2['pepper_prob'])}，加噪后 PSNR={task2['psnr_noisy']:.2f} dB",
            "各滤波器 PSNR：",
        ]
    )

    for name, value in sorted(task2["filter_psnrs"].items(), key=lambda item: item[1], reverse=True):
        report_lines.append(f"- {name}: {value:.2f} dB")

    report_lines.extend(
        [
            f"分析：本次椒盐噪声恢复中，{best_task2} 的 PSNR 最高，为 {task2['filter_psnrs'][best_task2]:.2f} dB。"
            "反谐波均值滤波中，Q>0 更适合压制胡椒噪声，Q<0 更适合压制盐噪声，Q=0 时退化为算术均值滤波；"
            "当盐和胡椒噪声同时存在时，中值滤波通常更稳健。",
            "结果图：outputs/task2_salt_pepper/comparison_panel.png",
            "",
            "3. 维纳滤波恢复",
            f"运动模糊参数：角度={format_number(task3['angle'])}°，T={format_number(task3['T'])}，运动模糊后 PSNR={task3['psnr_blurred']:.2f} dB",
            f"加性高斯噪声参数：均值={format_number(task3['noise_mean'])}，方差={format_number(task3['noise_var'])}，退化图像 PSNR={task3['psnr_degraded']:.2f} dB",
            f"简化维纳滤波参数：K={task3['simple_K']:.6f}",
            "恢复结果 PSNR：",
        ]
    )

    for name, value in sorted(task3["restoration_psnrs"].items(), key=lambda item: item[1], reverse=True):
        report_lines.append(f"- {name}: {value:.2f} dB")

    report_lines.extend(
        [
            f"分析：本次退化恢复中，{best_task3} 的 PSNR 最高，为 {task3['restoration_psnrs'][best_task3]:.2f} dB。"
            "Eq. (5.9-4) 采用常数 K 近似噪声与信号谱比，实现简单；Eq. (5.8-6) 则按频率点使用估计的噪声功率谱与原图功率谱，因此对不同频段的抑噪与去模糊权衡更细。",
            "结果图：outputs/task3_wiener_filter/comparison_panel.png",
            "",
            "四、结论",
            f"1. 对高斯噪声，{best_task1} 在本次实验中表现最好，说明线性平滑滤波更适合处理服从高斯分布的随机噪声。",
            f"2. 对椒盐噪声，{best_task2} 在本次实验中表现最好，说明非线性滤波或有选择性的反谐波滤波对脉冲噪声更有效。",
            f"3. 对运动模糊加噪退化图像，{best_task3} 在本次实验中恢复效果最佳，体现了维纳滤波在含噪逆问题中的优势。",
            "4. 图像恢复方法的效果同时受退化模型准确性、噪声统计特性和滤波参数选择影响，参数失配会直接降低恢复质量。",
        ]
    )

    markdown_lines = [
        "# 数字图像和视频处理-第六次作业实验报告",
        "",
        "班级：自动化2305  ",
        "姓名：周湛昊  ",
        "学号：2233712088",
        "",
        "## 一、实验内容",
        "",
        "1. 在测试图像上产生高斯噪声 lena 图，指定均值和方差，并用多种滤波器恢复图像，分析各自优缺点。",
        "2. 在测试图像 lena 图加入椒盐噪声，椒和盐噪声密度均为 0.1，用学过的滤波器恢复图像，并分析反谐波滤波中 Q 大于 0 和小于 0 的作用。",
        "3. 推导维纳滤波器并完成以下内容：",
        "   (a) 实现模糊滤波器，如方程 Eq. (5.6-11)。",
        "   (b) 对 lena 图像施加 45 度方向、T=1 的运动模糊。",
        "   (c) 在模糊图像中加入均值为 0、方差为 10 的高斯噪声。",
        "   (d) 分别利用方程 Eq. (5.8-6) 和 Eq. (5.9-4) 恢复图像，并分析算法优缺点。",
        "",
        "主程序文件为 hw6.py，输入图像位于 assets 文件夹，结果输出到 outputs 文件夹。",
        "",
        "## 二、实验环境",
        "",
        "- Python 3.11",
        "- NumPy",
        "- OpenCV",
        "- Matplotlib",
        "- SciPy",
        "- Pillow",
        "",
        "## 三、实验过程与结果分析",
        "",
        "### 1. 高斯噪声图像恢复",
        "",
        f"输入图像为 lena.bmp，尺寸为 {image.shape}，均值为 {image_mean:.2f}，标准差为 {image_std:.2f}。",
        f"本实验设置高斯噪声均值为 {format_number(task1['noise_mean'])}、方差为 {format_number(task1['noise_var'])}，加噪后图像 PSNR 为 {task1['psnr_noisy']:.2f} dB。",
        "",
        "| 滤波器 | PSNR/dB |",
        "|:---|:---:|",
    ]

    for name, value in sorted(task1["filter_psnrs"].items(), key=lambda item: item[1], reverse=True):
        markdown_lines.append(f"| {name} | {value:.2f} |")

    markdown_lines.extend(
        [
            "",
            f"分析：{best_task1} 的 PSNR 最高，为 {task1['filter_psnrs'][best_task1]:.2f} dB。算术均值与几何均值能够有效平滑高斯噪声，但会带来一定模糊；中值滤波更偏向脉冲噪声抑制，因此在该任务中优势不明显。",
            "",
            "结果图：",
            "",
            "![](outputs/task1_gaussian_noise/comparison_panel.png)",
            "",
            "### 2. 椒盐噪声图像恢复",
            "",
            f"本实验设置盐噪声密度和胡椒噪声密度均为 {format_number(task2['salt_prob'])}，加噪后图像 PSNR 为 {task2['psnr_noisy']:.2f} dB。",
            "",
            "| 滤波器 | PSNR/dB |",
            "|:---|:---:|",
        ]
    )

    for name, value in sorted(task2["filter_psnrs"].items(), key=lambda item: item[1], reverse=True):
        markdown_lines.append(f"| {name} | {value:.2f} |")

    markdown_lines.extend(
        [
            "",
            f"分析：{best_task2} 的 PSNR 最高，为 {task2['filter_psnrs'][best_task2]:.2f} dB。反谐波均值滤波中，Q>0 更利于去除胡椒噪声，Q<0 更利于去除盐噪声；当两类脉冲噪声同时存在时，中值滤波通常更稳健。",
            "",
            "结果图：",
            "",
            "![](outputs/task2_salt_pepper/comparison_panel.png)",
            "",
            "### 3. 维纳滤波恢复",
            "",
            f"先按题目要求构造角度为 {format_number(task3['angle'])}°、参数 T={format_number(task3['T'])} 的运动模糊，再叠加均值为 {format_number(task3['noise_mean'])}、方差为 {format_number(task3['noise_var'])} 的高斯噪声。运动模糊图像 PSNR 为 {task3['psnr_blurred']:.2f} dB，最终退化图像 PSNR 为 {task3['psnr_degraded']:.2f} dB。",
            "",
            "| 恢复方法 | PSNR/dB |",
            "|:---|:---:|",
        ]
    )

    for name, value in sorted(task3["restoration_psnrs"].items(), key=lambda item: item[1], reverse=True):
        markdown_lines.append(f"| {name} | {value:.2f} |")

    markdown_lines.extend(
        [
            "",
            f"简化维纳滤波中采用的 K 值为 {task3['simple_K']:.6f}。{best_task3} 的恢复效果最好，PSNR 为 {task3['restoration_psnrs'][best_task3]:.2f} dB。",
            "Eq. (5.9-4) 用常数 K 抑制直接逆滤波的不稳定性；Eq. (5.8-6) 则按频率点结合噪声功率谱与原图估计功率谱，因此理论上更贴近完整维纳滤波模型。",
            "",
            "结果图：",
            "",
            "![](outputs/task3_wiener_filter/comparison_panel.png)",
            "",
            "## 四、结论",
            "",
            f"1. 对高斯噪声，{best_task1} 在本次实验中效果最好，说明线性平滑滤波更适合处理随机高斯扰动。",
            f"2. 对椒盐噪声，{best_task2} 在本次实验中效果最好，说明对脉冲噪声更需要利用非线性或选择性滤波。",
            f"3. 对运动模糊并叠加噪声的退化图像，{best_task3} 的恢复效果最佳，说明维纳滤波在含噪逆问题中更有优势。",
            "4. 图像恢复质量不仅取决于滤波器形式，还取决于退化模型和参数估计是否准确。",
        ]
    )

    return "\n".join(report_lines), "\n".join(markdown_lines)


def task1_gaussian_noise(image: np.ndarray, report: list, rng: np.random.Generator):
    """任务1：高斯噪声处理"""
    report.append("=" * 60)
    report.append("Task 1: Gaussian Noise Processing")
    report.append("=" * 60)

    # 添加高斯噪声 (均值为0，方差为400即标准差20)
    mean = TASK1_GAUSSIAN_MEAN
    var = TASK1_GAUSSIAN_VAR
    noisy = add_gaussian_noise(image, mean, var, rng)
    psnr_noisy = calculate_psnr(image, noisy)

    report.append(f"Gaussian Noise: mean={mean}, variance={var}")
    report.append(f"Noisy Image PSNR: {psnr_noisy:.2f} dB")

    # 应用各种滤波器
    filters = {
        "Arithmetic Mean (3x3)": arithmetic_mean_filter(noisy, 3),
        "Arithmetic Mean (5x5)": arithmetic_mean_filter(noisy, 5),
        "Geometric Mean (3x3)": geometric_mean_filter(noisy, 3),
        "Harmonic Mean (3x3)": harmonic_mean_filter(noisy, 3),
        "Contraharmonic Q=1 (3x3)": contraharmonic_mean_filter(noisy, Q=1.0, kernel_size=3),
        "Contraharmonic Q=-1 (3x3)": contraharmonic_mean_filter(noisy, Q=-1.0, kernel_size=3),
        "Median (3x3)": median_filter(noisy, 3),
        "Median (5x5)": median_filter(noisy, 5),
    }

    psnrs = {"Original": float('inf'), f"Noisy (var={var})": psnr_noisy}

    report.append("\nFilter Results (PSNR in dB):")
    report.append("-" * 40)

    filter_psnrs = {}

    for name, filtered in filters.items():
        psnr = calculate_psnr(image, filtered)
        filter_psnrs[name] = psnr
        psnrs[name] = psnr
        report.append(f"{name}: {psnr:.2f} dB")
        cv_imwrite_unicode(TASK1_DIR / f"filtered_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}.bmp", filtered)

    # 保存原始和噪声图像
    cv_imwrite_unicode(TASK1_DIR / "original.bmp", image)
    cv_imwrite_unicode(TASK1_DIR / f"noisy_gaussian_var{var}.bmp", noisy)

    # 保存对比图
    panel_imgs = {
        "Original": image,
        "Noisy": noisy,
        "Arithmetic (3x3)": filters["Arithmetic Mean (3x3)"],
        "Geometric (3x3)": filters["Geometric Mean (3x3)"],
        "Harmonic (3x3)": filters["Harmonic Mean (3x3)"],
        "Median (3x3)": filters["Median (3x3)"],
    }
    panel_psnrs = {k: psnrs.get(k.replace(" (3x3)", " Mean (3x3)").replace("Arithmetic", "Arithmetic Mean").replace("Geometric", "Geometric Mean").replace("Harmonic", "Harmonic Mean").replace("Median", "Median"), 0) for k in panel_imgs.keys()}
    panel_psnrs["Original"] = float('inf')
    panel_psnrs["Noisy"] = psnr_noisy

    save_comparison_panel("Task 1: Gaussian Noise Filtering", panel_imgs, TASK1_DIR / "comparison_panel.png", panel_psnrs)

    # 分析
    report.append("\nAnalysis:")
    report.append("- Arithmetic Mean: 有效平滑高斯噪声，但会模糊边缘")
    report.append("- Geometric Mean: 对高斯噪声效果与算术均值类似，略好")
    report.append("- Harmonic Mean: 对高斯噪声效果较差，适合脉冲噪声")
    report.append("- Contraharmonic: Q>0适合胡椒噪声，Q<0适合盐噪声")
    report.append("- Median Filter: 对高斯噪声效果一般，但保留边缘较好")

    return {
        "noise_mean": mean,
        "noise_var": var,
        "psnr_noisy": psnr_noisy,
        "filter_psnrs": filter_psnrs,
    }


def task2_salt_and_pepper(image: np.ndarray, report: list, rng: np.random.Generator):
    """任务2：椒盐噪声处理"""
    report.append("\n" + "=" * 60)
    report.append("Task 2: Salt-and-Pepper Noise Processing")
    report.append("=" * 60)

    # 添加椒盐噪声 (密度均为0.1)
    salt_prob = TASK2_SALT_PROB
    pepper_prob = TASK2_PEPPER_PROB
    noisy = add_salt_and_pepper_noise(image, salt_prob, pepper_prob, rng)
    psnr_noisy = calculate_psnr(image, noisy)

    report.append(f"Salt-and-Pepper Noise: salt_density={salt_prob}, pepper_density={pepper_prob}")
    report.append(f"Noisy Image PSNR: {psnr_noisy:.2f} dB")

    # 应用各种滤波器
    filters = {
        "Arithmetic Mean (3x3)": arithmetic_mean_filter(noisy, 3),
        "Geometric Mean (3x3)": geometric_mean_filter(noisy, 3),
        "Harmonic Mean (3x3)": harmonic_mean_filter(noisy, 3),
        "Contraharmonic Q=1.5 (3x3)": contraharmonic_mean_filter(noisy, Q=1.5, kernel_size=3),
        "Contraharmonic Q=-1.5 (3x3)": contraharmonic_mean_filter(noisy, Q=-1.5, kernel_size=3),
        "Contraharmonic Q=0 (3x3)": contraharmonic_mean_filter(noisy, Q=0, kernel_size=3),
        "Median (3x3)": median_filter(noisy, 3),
    }

    psnrs = {"Original": float('inf'), "Noisy": psnr_noisy}

    report.append("\nFilter Results (PSNR in dB):")
    report.append("-" * 40)

    filter_psnrs = {}

    for name, filtered in filters.items():
        psnr = calculate_psnr(image, filtered)
        filter_psnrs[name] = psnr
        psnrs[name] = psnr
        report.append(f"{name}: {psnr:.2f} dB")
        cv_imwrite_unicode(TASK2_DIR / f"filtered_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_').replace('.', 'p')}.bmp", filtered)

    # 保存原始和噪声图像
    cv_imwrite_unicode(TASK2_DIR / "original.bmp", image)
    cv_imwrite_unicode(TASK2_DIR / f"noisy_salt{salt_prob}_pepper{pepper_prob}.bmp".replace("0.1", "0p1"), noisy)

    # 保存对比图
    panel_imgs = {
        "Original": image,
        "Noisy": noisy,
        "Contraharmonic Q>0": filters["Contraharmonic Q=1.5 (3x3)"],
        "Contraharmonic Q<0": filters["Contraharmonic Q=-1.5 (3x3)"],
        "Contraharmonic Q=0": filters["Contraharmonic Q=0 (3x3)"],
        "Median": filters["Median (3x3)"],
    }
    save_comparison_panel("Task 2: Salt-and-Pepper Noise Filtering", panel_imgs, TASK2_DIR / "comparison_panel.png")

    # 反谐波滤波器分析
    report.append("\nContraharmonic Mean Filter Analysis:")
    report.append("- Q > 0: 适合去除胡椒噪声(黑点)，但会增强盐噪声")
    report.append("- Q < 0: 适合去除盐噪声(白点)，但会增强胡椒噪声")
    report.append("- Q = 0: 等同于算术均值滤波器")
    report.append("- |Q|越大，滤波器的非线性选择性越强")

    return {
        "salt_prob": salt_prob,
        "pepper_prob": pepper_prob,
        "psnr_noisy": psnr_noisy,
        "filter_psnrs": filter_psnrs,
    }


def task3_wiener_filter(image: np.ndarray, report: list, rng: np.random.Generator):
    """任务3：维纳滤波器"""
    report.append("\n" + "=" * 60)
    report.append("Task 3: Wiener Filter Derivation and Implementation")
    report.append("=" * 60)

    # (a) & (b) 运动模糊
    angle = TASK3_MOTION_ANGLE
    T = TASK3_MOTION_T

    report.append(f"\n(a) Motion Blur Filter Implementation (Eq. 5.6-11)")
    report.append(f"(b) Blur Lena: angle={angle}°, T={T}")

    blurred = apply_motion_blur(image, angle, T)
    psnr_blurred = calculate_psnr(image, blurred)

    report.append(f"Blurred Image PSNR: {psnr_blurred:.2f} dB")

    # (c) 添加高斯噪声
    noise_mean = TASK3_NOISE_MEAN
    noise_var = TASK3_NOISE_VAR
    noise = rng.normal(noise_mean, np.sqrt(noise_var), image.shape)
    degraded = blurred.astype(np.float64) + noise
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    psnr_degraded = calculate_psnr(image, degraded)

    report.append(f"\n(c) Add Gaussian Noise: mean={noise_mean}, variance={noise_var}")
    report.append(f"Degraded Image PSNR: {psnr_degraded:.2f} dB")

    # 计算运动模糊核
    H = motion_blur_kernel_freq(image.shape, angle, T)

    # (d) 分别使用 Eq. (5.8-6) 和 Eq. (5.9-4) 恢复
    report.append(f"\n(d) Image Restoration:")

    # 简化的维纳滤波 (Eq. 5.9-4)
    K = noise_var / (np.var(image.astype(np.float64)) + 1e-10)  # 噪声信号比
    restored_simple = wiener_filter_simple(degraded, H, K=K)
    psnr_simple = calculate_psnr(image, restored_simple.astype(np.uint8))

    report.append(f"\nEq. 5.9-4 (Simplified Wiener) K={K:.6f}:")
    report.append(f"  Restored PSNR: {psnr_simple:.2f} dB")

    # 完整维纳滤波 (Eq. 5.8-6)
    signal_psd = np.abs(fft2(image.astype(np.float64))) ** 2 / image.size
    noise_psd = np.full(image.shape, noise_var, dtype=np.float64)
    restored_full = wiener_filter_full(degraded, H, noise_psd, signal_psd)
    psnr_full = calculate_psnr(image, restored_full.astype(np.uint8))

    report.append(f"\nEq. 5.8-6 (Full Wiener):")
    report.append("  Sn(u,v) 使用已知白噪声方差构造，Sf(u,v) 由原图功率谱估计")
    report.append(f"  Restored PSNR: {psnr_full:.2f} dB")

    # 保存图像
    cv_imwrite_unicode(TASK3_DIR / "original.bmp", image)
    cv_imwrite_unicode(TASK3_DIR / f"blurred_{format_number(angle)}deg_T{format_number(T)}.bmp", blurred)
    cv_imwrite_unicode(TASK3_DIR / f"degraded_noise_var{format_number(noise_var)}.bmp", degraded)
    cv_imwrite_unicode(TASK3_DIR / "restored_eq5_9_4_simple.bmp", restored_simple.astype(np.uint8))
    cv_imwrite_unicode(TASK3_DIR / "restored_eq5_8_6_full.bmp", restored_full.astype(np.uint8))

    # 保存对比图
    panel_imgs = {
        "Original": image,
        "Motion Blurred": blurred,
        "Noisy Degraded": degraded,
        "Eq.5.9-4 Simple": restored_simple.astype(np.uint8),
        "Eq.5.8-6 Full": restored_full.astype(np.uint8),
    }
    panel_psnrs = {
        "Original": float('inf'),
        "Motion Blurred": psnr_blurred,
        "Noisy Degraded": psnr_degraded,
        "Eq.5.9-4 Simple": psnr_simple,
        "Eq.5.8-6 Full": psnr_full,
    }
    save_comparison_panel("Task 3: Wiener Filter Restoration", panel_imgs, TASK3_DIR / "comparison_panel.png", panel_psnrs)

    # 算法分析
    report.append(f"\nAlgorithm Analysis:")
    report.append("-" * 40)
    report.append("Eq. 5.9-4 (Simplified Wiener Filter):")
    report.append("  Formula: F̂(u,v) = [1/H(u,v)] × [|H(u,v)|²/(|H(u,v)|²+K)] × G(u,v)")
    report.append("  Pros: 实现简单，计算量小，不需要知道原始信号统计特性")
    report.append("  Cons: K值需要经验选择，对噪声敏感")
    report.append("")
    report.append("Eq. 5.8-6 (Full Wiener Filter):")
    report.append("  Formula: F̂(u,v) = [H*(u,v)/(|H(u,v)|² + Sn(u,v)/Sf(u,v))] × G(u,v)")
    report.append("  Pros: 理论上最优的线性估计，最小化均方误差")
    report.append("  Cons: 需要知道噪声和信号的功率谱，实际中难以获得")

    return {
        "angle": angle,
        "T": T,
        "noise_mean": noise_mean,
        "noise_var": noise_var,
        "psnr_blurred": psnr_blurred,
        "psnr_degraded": psnr_degraded,
        "simple_K": K,
        "restoration_psnrs": {
            "Eq. (5.9-4) 简化维纳滤波": psnr_simple,
            "Eq. (5.8-6) 完整维纳滤波": psnr_full,
        },
    }


def main():
    # 读取图像
    lena_path = ASSET_DIR / "lena.bmp"
    image = cv_imread_unicode(lena_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read {lena_path}")

    rng = np.random.default_rng(RNG_SEED)

    # 执行任务
    scratch_report = []
    task1_result = task1_gaussian_noise(image, scratch_report, rng)
    task2_result = task2_salt_and_pepper(image, scratch_report, rng)
    task3_result = task3_wiener_filter(image, scratch_report, rng)

    report_text, report_markdown = build_reports(image, task1_result, task2_result, task3_result)

    # 保存报告
    report_path = OUT_DIR / "report.txt"
    markdown_path = ROOT / "第六次作业实验报告.md"
    report_path.write_text(report_text, encoding="utf-8")
    markdown_path.write_text(report_markdown, encoding="utf-8")

    print("=" * 60)
    print("Homework 6 Processing Complete!")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    print(f"Report saved to: {report_path}")
    print(f"Markdown report saved to: {markdown_path}")
    print("\nSummary:")
    print("  - Task 1: Gaussian noise filtering results in", TASK1_DIR)
    print("  - Task 2: Salt-and-pepper noise filtering results in", TASK2_DIR)
    print("  - Task 3: Wiener filter restoration results in", TASK3_DIR)


if __name__ == "__main__":
    main()
