from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
OUT_DIR   = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 2026

def cv_imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags) if data.size else None

def cv_imwrite(path, image):
    ok, enc = cv2.imencode(Path(path).suffix or ".png", image)
    if ok:
        enc.tofile(str(path))

def get_seven_points(gray_src, gray_dst, seed=RANDOM_SEED):
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(gray_src, None)
    kp2, des2 = orb.detectAndCompute(gray_dst, None)

    matches = sorted(
        cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2),
        key=lambda m: m.distance,
    )
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = [m for m, k in zip(matches, mask.ravel()) if k]

    rng = random.Random(seed)
    sel = sorted(rng.sample(range(len(inliers)), 7))
    src = np.float64([kp1[inliers[i].queryIdx].pt for i in sel])
    dst = np.float64([kp2[inliers[i].trainIdx].pt for i in sel])
    return src, dst

def compute_H(src, dst):
    def normalize(pts):
        c = pts.mean(0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
        T = np.array([[s, 0, -s*c[0]], [0, s, -s*c[1]], [0, 0, 1]])
        return (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2], T

    sn, Ts = normalize(src)
    dn, Td = normalize(dst)
    A = []
    for (x, y), (u, v) in zip(sn, dn):
        A += [[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
              [ 0,  0,  0,-x,-y,-1, v*x, v*y, v]]
    _, _, Vt = np.linalg.svd(np.array(A))
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]

def mark_points(image, points, out_path):
    canvas = image.copy()
    for i, (x, y) in enumerate(points, 1):
        c = (int(round(x)), int(round(y)))
        cv2.circle(canvas, c, 18, (0, 0, 255), 4)
        cv2.putText(canvas, str(i), (c[0]+18, c[1]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv_imwrite(out_path, canvas)

def main():
    lbase = cv_imread(ASSET_DIR / "Image A.jpg")
    lin   = cv_imread(ASSET_DIR / "Image B.jpg")
    moving_pts, fixed_pts = get_seven_points(
        cv2.cvtColor(lin,   cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(lbase, cv2.COLOR_BGR2GRAY),
    )

    H = compute_H(moving_pts, fixed_pts)

    mark_points(lin,   moving_pts, OUT_DIR / "marked_points_imageB.png")
    mark_points(lbase, fixed_pts,  OUT_DIR / "marked_points_imageA.png")
    hb, wb = lin.shape[:2]
    corners = np.float32([[0,0],[wb,0],[wb,hb],[0,hb]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x_min, y_min = warped_corners.min(axis=0).astype(int)
    x_max, y_max = warped_corners.max(axis=0).astype(int)

    T_shift = np.array([[1, 0, -x_min],
                        [0, 1, -y_min],
                        [0, 0,      1]], dtype=np.float64)
    H_shifted = T_shift @ H
    out_w, out_h = x_max - x_min, y_max - y_min

    lout = cv2.warpPerspective(lin, H_shifted, (out_w, out_h))
    cv_imwrite(OUT_DIR / "warped_b_to_a.png", lout)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(lout,  cv2.COLOR_BGR2RGB)); axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(lbase, cv2.COLOR_BGR2RGB)); axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "registration_panel.png", dpi=150)
    plt.close(fig)

    print("H =")
    for row in H:
        print("  " + "  ".join(f"{v:.8f}" for v in row))

if __name__ == "__main__":
    main()