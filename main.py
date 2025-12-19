import os
import re
import glob
import math
import argparse
from typing import List, Tuple, Optional

import numpy as np
import cv2


# ============================================================
# Utils (allowed: cv2 load/save, basic math, matrix ops)
# ============================================================

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def to_float01(img_u8_rgb: np.ndarray) -> np.ndarray:
    return img_u8_rgb.astype(np.float32) / 255.0

def to_u8_rgb(img_f01: np.ndarray) -> np.ndarray:
    x = (clamp01(img_f01) * 255.0 + 0.5).astype(np.uint8)
    return x

def rgb_to_gray(rgb01: np.ndarray) -> np.ndarray:
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)

def normalize_gray(gray: np.ndarray) -> np.ndarray:
    """Simple contrast normalization (helps brightness changes)."""
    m = float(np.mean(gray))
    s = float(np.std(gray) + 1e-6)
    return ((gray - m) / s).astype(np.float32)

def gray_hist_equalize(gray01: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    Simple histogram equalization (self-implemented).
    Input: gray in [0,1] float32
    Output: equalized gray in [0,1]
    """
    g = np.clip(gray01, 0.0, 1.0)
    x = (g * (nbins - 1)).astype(np.int32)
    hist = np.bincount(x.reshape(-1), minlength=nbins).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf = cdf / (cdf[-1] + 1e-12)
    y = cdf[x]
    return y.astype(np.float32)

def natural_key(path: str):
    base = os.path.basename(path)
    parts = re.split(r"(\d+)", base)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

def list_images(input_dir: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(input_dir, pattern))
    return sorted(paths, key=natural_key)

def load_images_from_folder(folder: str, pattern: str) -> Tuple[List[np.ndarray], List[str]]:
    paths = list_images(folder, pattern)
    imgs = []
    kept = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imgs.append(to_float01(rgb))
        kept.append(p)
    return imgs, kept

def save_image_rgb(path: str, rgb_u8: np.ndarray) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


# ============================================================
# Minimal image processing (no cv2 filters; implement ourselves)
# Speed: separable blur uses np.convolve (C-fast core)
# ============================================================

def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if radius is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma + 1e-12))
    k = k / (np.sum(k) + 1e-12)
    return k.astype(np.float32)

def convolve1d_edge(signal: np.ndarray, k: np.ndarray) -> np.ndarray:
    r = (k.shape[0] - 1) // 2
    pad = np.pad(signal, (r, r), mode="edge")
    out = np.convolve(pad, k[::-1], mode="valid")
    return out.astype(np.float32)

def convolve_separable(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    H, W = img.shape
    tmp = np.empty_like(img, dtype=np.float32)
    for y in range(H):
        tmp[y, :] = convolve1d_edge(img[y, :], k)

    out = np.empty_like(img, dtype=np.float32)
    for x in range(W):
        out[:, x] = convolve1d_edge(tmp[:, x], k)
    return out

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = gaussian_kernel1d(sigma)
    return convolve_separable(img.astype(np.float32), k)

def sobel_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = np.pad(gray.astype(np.float32), ((1, 1), (1, 1)), mode="edge")

    gx = (
        (-1.0) * g[0:-2, 0:-2] + (1.0) * g[0:-2, 2:] +
        (-2.0) * g[1:-1, 0:-2] + (2.0) * g[1:-1, 2:] +
        (-1.0) * g[2:,   0:-2] + (1.0) * g[2:,   2:]
    ).astype(np.float32)

    gy = (
        (-1.0) * g[0:-2, 0:-2] + (-2.0) * g[0:-2, 1:-1] + (-1.0) * g[0:-2, 2:] +
        ( 1.0) * g[2:,   0:-2] + ( 2.0) * g[2:,   1:-1] + ( 1.0) * g[2:,   2:]
    ).astype(np.float32)

    return gx, gy

def harris_response(gray: np.ndarray, k: float = 0.04, sigma: float = 1.5) -> np.ndarray:
    gx, gy = sobel_gradients(gray)
    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy

    Sxx = gaussian_blur(Ixx, sigma)
    Syy = gaussian_blur(Iyy, sigma)
    Sxy = gaussian_blur(Ixy, sigma)

    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    R = det - k * (trace * trace)
    return R.astype(np.float32)

def nonmax_suppression_candidates(R: np.ndarray, radius: int = 8, thresh_rel: float = 0.01) -> np.ndarray:
    """
    Faster NMS:
    - first collect candidates where R >= thr
    - only check local maxima around candidates
    """
    H, W = R.shape
    m = float(np.max(R))
    if not np.isfinite(m) or m <= 0:
        return np.zeros_like(R, dtype=bool)
    thr = m * float(thresh_rel)

    cand = np.argwhere(R >= thr)
    if cand.shape[0] == 0:
        return np.zeros_like(R, dtype=bool)

    keep = np.zeros((H, W), dtype=bool)
    r = int(radius)

    for y, x in cand:
        y0 = max(0, y - r)
        y1 = min(H - 1, y + r)
        x0 = max(0, x - r)
        x1 = min(W - 1, x + r)
        patch = R[y0:y1 + 1, x0:x1 + 1]
        if R[y, x] >= np.max(patch):
            keep[y, x] = True
    return keep

def anms(kps_yx: np.ndarray, scores: np.ndarray, max_pts: int = 800, c_robust: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    N = kps_yx.shape[0]
    if N <= max_pts:
        return kps_yx, scores

    idx = np.argsort(scores)[::-1]
    kps = kps_yx[idx]
    sc = scores[idx]

    radii = np.full((N,), np.inf, dtype=np.float32)
    for i in range(N):
        yi, xi = kps[i]
        si = sc[i]
        stronger = np.where(sc > c_robust * si)[0]
        if stronger.size == 0:
            continue
        dy = (kps[stronger, 0] - yi).astype(np.float32)
        dx = (kps[stronger, 1] - xi).astype(np.float32)
        d2 = dx * dx + dy * dy
        radii[i] = float(np.min(d2))

    pick = np.argsort(radii)[::-1][:max_pts]
    out = kps[pick]
    out_sc = sc[pick]
    return out.astype(np.int32), out_sc.astype(np.float32)

def resize_nn(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return img.copy()
    h, w = img.shape[:2]
    nh = max(2, int(round(h * scale)))
    nw = max(2, int(round(w * scale)))
    ys = (np.linspace(0, h - 1, nh)).astype(np.int32)
    xs = (np.linspace(0, w - 1, nw)).astype(np.int32)
    if img.ndim == 2:
        return img[ys[:, None], xs[None, :]].astype(np.float32)
    return img[ys[:, None], xs[None, :], :].astype(np.float32)

def extract_descriptors(gray: np.ndarray,
                        kps_yx: np.ndarray,
                        patch_radius: int = 20,
                        desc_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    H, W = gray.shape
    r = int(patch_radius)
    ps = 2 * r + 1
    desc = []
    pts = []
    for (y, x) in kps_yx:
        y = int(y); x = int(x)
        if y - r < 0 or x - r < 0 or y + r >= H or x + r >= W:
            continue
        patch = gray[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32)

        m = float(np.mean(patch))
        s = float(np.std(patch) + 1e-6)
        patch = (patch - m) / s

        ds = int(desc_size)
        grid = np.zeros((ds, ds), dtype=np.float32)
        for yy in range(ds):
            for xx in range(ds):
                y0 = int(round(yy * ps / ds))
                y1 = int(round((yy + 1) * ps / ds))
                x0 = int(round(xx * ps / ds))
                x1 = int(round((xx + 1) * ps / ds))
                y1 = max(y0 + 1, y1)
                x1 = max(x0 + 1, x1)
                block = patch[y0:y1, x0:x1]
                grid[yy, xx] = float(np.mean(block))
        v = grid.reshape(-1)
        nrm = float(np.linalg.norm(v) + 1e-12)
        v = (v / nrm).astype(np.float32)

        desc.append(v)
        pts.append([y, x])

    if len(desc) == 0:
        return np.zeros((0, desc_size * desc_size), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    return np.stack(desc).astype(np.float32), np.array(pts, dtype=np.int32)

def match_descriptors(descA: np.ndarray, descB: np.ndarray,
                      ratio: float = 0.75, max_matches: int = 1200) -> List[Tuple[int, int, float]]:
    if descA.shape[0] == 0 or descB.shape[0] == 0:
        return []
    sim = descA @ descB.T
    dist = 1.0 - sim

    j1 = np.argmin(dist, axis=1)
    best = dist[np.arange(dist.shape[0]), j1]

    dist2 = dist.copy()
    dist2[np.arange(dist2.shape[0]), j1] = np.inf
    j2 = np.argmin(dist2, axis=1)
    second = dist2[np.arange(dist.shape[0]), j2]

    keep = best < ratio * (second + 1e-12)
    cand = [(i, int(j1[i]), float(best[i])) for i in range(dist.shape[0]) if keep[i]]
    if not cand:
        return []

    i1 = np.argmin(dist, axis=0)
    out = []
    for i, j, d in cand:
        if i1[j] == i:
            out.append((i, j, d))

    out.sort(key=lambda t: t[2])
    if len(out) > max_matches:
        out = out[:max_matches]
    return out


# ============================================================
# Homography (DLT + RANSAC)
# ============================================================

def normalize_points(pts_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = pts_xy.astype(np.float64)
    c = np.mean(pts, axis=0)
    pts0 = pts - c[None, :]
    mean_dist = np.mean(np.sqrt(np.sum(pts0 ** 2, axis=1))) + 1e-12
    s = math.sqrt(2.0) / float(mean_dist)
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1]], dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    q = (T @ pts_h.T).T
    q = q[:, :2] / (q[:, 2:3] + 1e-12)
    return q.astype(np.float64), T

def dlt_homography(pts_src: np.ndarray, pts_dst: np.ndarray) -> Optional[np.ndarray]:
    if pts_src.shape[0] < 4:
        return None
    a_n, Ta = normalize_points(pts_src)
    b_n, Tb = normalize_points(pts_dst)

    N = a_n.shape[0]
    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        x, y = a_n[i]
        u, v = b_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    h = Vt[-1]
    Hn = h.reshape(3, 3)

    try:
        H = np.linalg.inv(Tb) @ Hn @ Ta
    except np.linalg.LinAlgError:
        return None

    if abs(H[2, 2]) < 1e-12:
        return None
    H = H / H[2, 2]
    return H.astype(np.float64)

def apply_homography(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = pts_xy.astype(np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    q = (H @ pts_h.T).T
    q = q[:, :2] / (q[:, 2:3] + 1e-12)
    return q.astype(np.float64)

def ransac_homography(pts_src: np.ndarray, pts_dst: np.ndarray,
                      iters: int = 1800, thresh: float = 3.5, conf: float = 0.995,
                      seed: int = 42) -> Tuple[Optional[np.ndarray], np.ndarray]:
    N = pts_src.shape[0]
    if N < 4:
        return None, np.zeros((N,), dtype=bool)

    best_inliers = np.zeros((N,), dtype=bool)
    best_cnt = 0
    best_H = None

    rng = np.random.default_rng(seed)
    max_iters = int(iters)
    cur_iters = max_iters

    for t in range(cur_iters):
        idx = rng.choice(N, size=4, replace=False)
        H = dlt_homography(pts_src[idx], pts_dst[idx])
        if H is None:
            continue

        proj = apply_homography(H, pts_src)
        err = np.sqrt(np.sum((proj - pts_dst) ** 2, axis=1))
        inliers = err < float(thresh)
        cnt = int(np.sum(inliers))

        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers
            best_H = H

            w = cnt / float(N)
            s = 4
            denom = math.log(1.0 - (w ** s) + 1e-12)
            if abs(denom) > 1e-12:
                est = int(math.ceil(math.log(1.0 - conf) / denom))
                cur_iters = max(t + 1, min(cur_iters, est))

        if t + 1 >= cur_iters:
            break

    if best_H is None or best_cnt < 10:
        return None, best_inliers

    H2 = dlt_homography(pts_src[best_inliers], pts_dst[best_inliers])
    if H2 is None:
        return best_H, best_inliers
    return H2, best_inliers


# ============================================================
# ✅ Affine fallback (RANSAC) : reduces "scattered panorama"
# (lecture-level: affine model when homography unstable)
# ============================================================

def affine_from_3pts(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    if src.shape[0] < 3:
        return None
    # Solve [x y 1 0 0 0; 0 0 0 x y 1] * p = [u v]
    A = np.zeros((6, 6), dtype=np.float64)
    b = np.zeros((6,), dtype=np.float64)
    for i in range(3):
        x, y = float(src[i, 0]), float(src[i, 1])
        u, v = float(dst[i, 0]), float(dst[i, 1])
        A[2*i]   = [x, y, 1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, x, y, 1]
        b[2*i]   = u
        b[2*i+1] = v
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    H = np.array([[p[0], p[1], p[2]],
                  [p[3], p[4], p[5]],
                  [0.0,  0.0,  1.0]], dtype=np.float64)
    return H

def ransac_affine(src: np.ndarray, dst: np.ndarray,
                  iters: int = 1200, thresh: float = 4.0, seed: int = 42) -> Tuple[Optional[np.ndarray], np.ndarray]:
    N = src.shape[0]
    if N < 3:
        return None, np.zeros((N,), dtype=bool)

    rng = np.random.default_rng(seed)
    best_inl = np.zeros((N,), dtype=bool)
    best_cnt = 0
    best_H = None

    for _ in range(int(iters)):
        idx = rng.choice(N, size=3, replace=False)
        H = affine_from_3pts(src[idx], dst[idx])
        if H is None:
            continue
        proj = apply_homography(H, src)
        err = np.sqrt(np.sum((proj - dst) ** 2, axis=1))
        inl = err < float(thresh)
        cnt = int(np.sum(inl))
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
            best_H = H

    if best_H is None or best_cnt < 8:
        return None, best_inl

    # refine with least squares on inliers
    src_in = src[best_inl]
    dst_in = dst[best_inl]
    M = src_in.shape[0]
    A = np.zeros((2*M, 6), dtype=np.float64)
    b = np.zeros((2*M,), dtype=np.float64)
    for i in range(M):
        x, y = float(src_in[i, 0]), float(src_in[i, 1])
        u, v = float(dst_in[i, 0]), float(dst_in[i, 1])
        A[2*i]   = [x, y, 1, 0, 0, 0]
        A[2*i+1] = [0, 0, 0, x, y, 1]
        b[2*i]   = u
        b[2*i+1] = v
    try:
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return best_H, best_inl

    H2 = np.array([[p[0], p[1], p[2]],
                   [p[3], p[4], p[5]],
                   [0.0,  0.0,  1.0]], dtype=np.float64)
    return H2, best_inl


# ============================================================
# Match scaling
# ============================================================

def choose_match_scale(h: int, w: int, max_dim: int = 900) -> float:
    m = max(h, w)
    if m <= max_dim:
        return 1.0
    return float(max_dim) / float(m)

def scale_homography(H_small: np.ndarray, s_from: float, s_to: float) -> np.ndarray:
    S_from = np.array([[s_from, 0, 0],
                       [0, s_from, 0],
                       [0, 0, 1]], dtype=np.float64)
    S_to = np.array([[s_to, 0, 0],
                     [0, s_to, 0],
                     [0, 0, 1]], dtype=np.float64)
    H = np.linalg.inv(S_to) @ H_small @ S_from
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H.astype(np.float64)


# ============================================================
# Homography sanity
# ============================================================

def corners_xy(h: int, w: int) -> np.ndarray:
    return np.array([[0, 0],
                     [w - 1, 0],
                     [w - 1, h - 1],
                     [0, h - 1]], dtype=np.float64)

def polygon_area(pts_xy: np.ndarray) -> float:
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

def is_homography_reasonable(H: np.ndarray, h: int, w: int,
                            max_area_ratio: float = 25.0,
                            min_area_ratio: float = 0.05,
                            max_persp: float = 0.015,
                            max_trans_mul: float = 4.5,
                            max_anisotropy: float = 8.0) -> bool:
    if H is None:
        return False
    if not np.all(np.isfinite(H)):
        return False
    if abs(H[2, 2]) < 1e-12:
        return False

    A = H[0:2, 0:2]
    det = float(np.linalg.det(A))
    if det <= 0:
        return False

    if abs(H[2, 0]) > max_persp or abs(H[2, 1]) > max_persp:
        return False

    try:
        s = np.linalg.svd(A, compute_uv=False)
        if s[1] < 1e-12:
            return False
        if float(s[0] / s[1]) > max_anisotropy:
            return False
    except np.linalg.LinAlgError:
        return False

    tx = float(H[0, 2])
    ty = float(H[1, 2])
    if abs(tx) > max_trans_mul * w or abs(ty) > max_trans_mul * h:
        return False

    c = corners_xy(h, w)
    wc = apply_homography(H, c)
    if not np.all(np.isfinite(wc)):
        return False

    area_src = abs(polygon_area(c))
    area_dst = abs(polygon_area(wc))
    if area_src < 1e-6:
        return False
    ratio = area_dst / area_src
    if ratio < min_area_ratio or ratio > max_area_ratio:
        return False

    return True

def bbox_from_pts(pts_xy: np.ndarray) -> Tuple[float, float, float, float]:
    xs = pts_xy[:, 0]
    ys = pts_xy[:, 1]
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))

def overlap_ratio_A_with_warpedB(H_B_to_A: np.ndarray, hA: int, wA: int, hB: int, wB: int) -> float:
    cA = corners_xy(hA, wA)
    cB = corners_xy(hB, wB)
    wB_in_A = apply_homography(H_B_to_A, cB)

    ax0, ay0, ax1, ay1 = bbox_from_pts(cA)
    bx0, by0, bx1, by1 = bbox_from_pts(wB_in_A)

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    areaA = float((wA - 1) * (hA - 1) + 1e-12)
    return float(inter / areaA)


# ============================================================
# Inverse warping + feather blending
# ============================================================

def project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = pts_xy.astype(np.float64)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    q = (H @ pts_h.T).T
    denom = (q[:, 2:3] + 1e-12)
    qxy = q[:, :2] / denom
    return qxy.astype(np.float64)

def compute_warp_bbox(H_img_to_canvas: np.ndarray, h: int, w: int) -> Tuple[float, float, float, float]:
    c = corners_xy(h, w)
    wc = apply_homography(H_img_to_canvas, c)
    return bbox_from_pts(wc)

def make_feather_weight(h: int, w: int, feather: float = 35.0, base: float = 0.12) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum.reduce([xx, yy, (w - 1) - xx, (h - 1) - yy])
    ramp = np.clip(d / max(1.0, float(feather)), 0.0, 1.0)
    ramp = np.sqrt(ramp)
    wgt = base + (1.0 - base) * ramp
    return wgt.astype(np.float32)

# ✅ 안전화: nan/inf가 섞여 들어오면 floor/cast에서 warning + 이상좌표
def _safe_xy(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs_safe = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    ys_safe = np.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return xs_safe, ys_safe, finite

def bilinear_sample_rgb(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = img.shape[:2]

    xs_safe, ys_safe, finite = _safe_xy(xs, ys)

    x0 = np.floor(xs_safe).astype(np.int32, copy=False)
    y0 = np.floor(ys_safe).astype(np.int32, copy=False)
    x1 = x0 + 1
    y1 = y0 + 1

    inside = finite & (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    Ia = img[y0c, x0c, :]
    Ib = img[y0c, x1c, :]
    Ic = img[y1c, x0c, :]
    Id = img[y1c, x1c, :]

    wx = (xs_safe - x0.astype(np.float32)).astype(np.float32)
    wy = (ys_safe - y0.astype(np.float32)).astype(np.float32)

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    out = (Ia * wa[:, None] + Ib * wb[:, None] + Ic * wc[:, None] + Id * wd[:, None]).astype(np.float32)
    return out, inside

def bilinear_sample_gray(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = img.shape[:2]

    xs_safe, ys_safe, finite = _safe_xy(xs, ys)

    x0 = np.floor(xs_safe).astype(np.int32, copy=False)
    y0 = np.floor(ys_safe).astype(np.int32, copy=False)
    x1 = x0 + 1
    y1 = y0 + 1

    inside = finite & (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]

    wx = (xs_safe - x0.astype(np.float32)).astype(np.float32)
    wy = (ys_safe - y0.astype(np.float32)).astype(np.float32)

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    out = (Ia * wa + Ib * wb + Ic * wc + Id * wd).astype(np.float32)
    return out, inside


# ============================================================
# Debug visualization: draw matches WITHOUT cv2 drawing
# ============================================================

def put_pixel(img: np.ndarray, x: int, y: int, color: Tuple[int, int, int]):
    H, W = img.shape[:2]
    if 0 <= x < W and 0 <= y < H:
        img[y, x, 0] = color[0]
        img[y, x, 1] = color[1]
        img[y, x, 2] = color[2]

def draw_disk(img: np.ndarray, x: int, y: int, r: int, color: Tuple[int, int, int]):
    for yy in range(y - r, y + r + 1):
        for xx in range(x - r, x + r + 1):
            if (xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r:
                put_pixel(img, xx, yy, color)

def draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        put_pixel(img, x, y, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

def make_match_viz(imgA_rgb01: np.ndarray, imgB_rgb01: np.ndarray,
                   ptsA_xy: np.ndarray, ptsB_xy: np.ndarray,
                   inliers: Optional[np.ndarray] = None,
                   max_draw: int = 120) -> np.ndarray:
    A = to_u8_rgb(imgA_rgb01)
    B = to_u8_rgb(imgB_rgb01)
    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]
    H = max(hA, hB)
    W = wA + wB
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:hA, :wA] = A
    canvas[:hB, wA:wA+wB] = B

    if ptsA_xy.shape[0] == 0:
        return canvas

    N = ptsA_xy.shape[0]
    idx = np.arange(N)
    if inliers is not None and inliers.shape[0] == N:
        inl_idx = idx[inliers]
        out_idx = idx[~inliers]
        order = np.concatenate([inl_idx, out_idx], axis=0)
    else:
        order = idx

    if order.shape[0] > max_draw:
        order = order[:max_draw]

    green = (0, 255, 0)
    red = (255, 0, 0)
    yellow = (255, 255, 0)

    for k in order:
        ax, ay = int(round(ptsA_xy[k, 0])), int(round(ptsA_xy[k, 1]))
        bx, by = int(round(ptsB_xy[k, 0])), int(round(ptsB_xy[k, 1]))
        bx2 = bx + wA

        ok = True
        if inliers is not None and inliers.shape[0] == N:
            ok = bool(inliers[k])

        col_line = green if ok else red
        col_pt = yellow if ok else red

        draw_disk(canvas, ax, ay, 3, col_pt)
        draw_disk(canvas, bx2, by, 3, col_pt)
        draw_line(canvas, ax, ay, bx2, by, col_line)

    return canvas


# ============================================================
# Pairwise registration
# ============================================================

def estimate_pair_homography(imgA_rgb: np.ndarray, imgB_rgb: np.ndarray,
                             debug_dir: Optional[str] = None,
                             pair_name: str = "",
                             max_corners: int = 850,
                             nms_radius: int = 8,
                             patch_radius: int = 20,
                             desc_size: int = 8,
                             ransac_iters: int = 2000,
                             ransac_thresh: float = 3.8,
                             match_ratio: float = 0.78,
                             max_dim_match: int = 800,
                             use_hist_eq: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:

    HA, WA = imgA_rgb.shape[:2]
    HB, WB = imgB_rgb.shape[:2]

    sA = choose_match_scale(HA, WA, max_dim=max_dim_match)
    sB = choose_match_scale(HB, WB, max_dim=max_dim_match)

    A_small = resize_nn(imgA_rgb, sA)
    B_small = resize_nn(imgB_rgb, sB)

    gA01 = rgb_to_gray(A_small)
    gB01 = rgb_to_gray(B_small)

    if use_hist_eq:
        gA01 = gray_hist_equalize(gA01)
        gB01 = gray_hist_equalize(gB01)

    gA = normalize_gray(gA01)
    gB = normalize_gray(gB01)

    RA = harris_response(gA, k=0.04, sigma=1.5)
    RB = harris_response(gB, k=0.04, sigma=1.5)

    keepA = nonmax_suppression_candidates(RA, radius=nms_radius, thresh_rel=0.01)
    keepB = nonmax_suppression_candidates(RB, radius=nms_radius, thresh_rel=0.01)

    kpsA = np.argwhere(keepA)
    kpsB = np.argwhere(keepB)

    if kpsA.shape[0] < 50 or kpsB.shape[0] < 50:
        return None, None, None, 0

    scA = RA[keepA]
    scB = RB[keepB]

    kpsA, scA = anms(kpsA, scA, max_pts=max_corners)
    kpsB, scB = anms(kpsB, scB, max_pts=max_corners)

    dA, kpsA2 = extract_descriptors(gA, kpsA, patch_radius=patch_radius, desc_size=desc_size)
    dB, kpsB2 = extract_descriptors(gB, kpsB, patch_radius=patch_radius, desc_size=desc_size)
    if dA.shape[0] < 30 or dB.shape[0] < 30:
        return None, None, None, 0

    matches = match_descriptors(dA, dB, ratio=match_ratio, max_matches=1200)
    if len(matches) < 14:
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            ptsA_small = []
            ptsB_small = []
            for (i, j, _) in matches:
                yA, xA = kpsA2[i]
                yB, xB = kpsB2[j]
                ptsA_small.append([float(xA), float(yA)])
                ptsB_small.append([float(xB), float(yB)])
            if len(ptsA_small) > 0:
                ptsA_small = np.array(ptsA_small, dtype=np.float64)
                ptsB_small = np.array(ptsB_small, dtype=np.float64)
                viz = make_match_viz(A_small, B_small, ptsA_small, ptsB_small, None, max_draw=120)
                save_image_rgb(os.path.join(debug_dir, f"match_{pair_name}_raw.png"), viz)
        return None, None, None, 0

    ptsA_small = []
    ptsB_small = []
    for (i, j, _) in matches:
        yA, xA = kpsA2[i]
        yB, xB = kpsB2[j]
        ptsA_small.append([float(xA), float(yA)])
        ptsB_small.append([float(xB), float(yB)])
    ptsA_small = np.array(ptsA_small, dtype=np.float64)
    ptsB_small = np.array(ptsB_small, dtype=np.float64)

    H_small, inl = ransac_homography(
        ptsB_small, ptsA_small,
        iters=ransac_iters, thresh=ransac_thresh, conf=0.995, seed=42
    )

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        viz_raw = make_match_viz(A_small, B_small, ptsA_small, ptsB_small, None, max_draw=120)
        save_image_rgb(os.path.join(debug_dir, f"match_{pair_name}_raw.png"), viz_raw)

    if H_small is None:
        return None, None, None, 0

    inlier_cnt = int(np.sum(inl))

    if debug_dir:
        viz_inl = make_match_viz(A_small, B_small, ptsA_small, ptsB_small, inl, max_draw=140)
        save_image_rgb(os.path.join(debug_dir, f"match_{pair_name}_inliers.png"), viz_inl)

    H_orig = scale_homography(H_small, s_from=sB, s_to=sA)

    ptsA_orig = (ptsA_small / max(sA, 1e-12)).astype(np.float64)
    ptsB_orig = (ptsB_small / max(sB, 1e-12)).astype(np.float64)

    return H_orig, ptsB_orig, ptsA_orig, inlier_cnt

def translation_fallback_from_matches(ptsB: np.ndarray, ptsA: np.ndarray) -> np.ndarray:
    d = (ptsA - ptsB).astype(np.float64)
    dx = float(np.median(d[:, 0]))
    dy = float(np.median(d[:, 1]))
    H = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]], dtype=np.float64)
    return H


# ============================================================
# Drift correction (helps scattered cases)
# ============================================================

def apply_drift_correction(H_to_ref: List[np.ndarray], images_rgb: List[np.ndarray], verbose: bool = True) -> List[np.ndarray]:
    """
    Simple drift correction:
    - compute transformed center points in ref frame
    - fit y = a*x + b
    - apply affine correction that removes slope
    """
    n = len(images_rgb)
    if n < 3:
        return H_to_ref

    centers = []
    for i, img in enumerate(images_rgb):
        h, w = img.shape[:2]
        c = np.array([[0.5 * (w - 1), 0.5 * (h - 1)]], dtype=np.float64)
        p = apply_homography(H_to_ref[i], c)[0]
        if np.all(np.isfinite(p)):
            centers.append(p)
        else:
            centers.append(np.array([np.nan, np.nan], dtype=np.float64))
    centers = np.array(centers, dtype=np.float64)

    ok = np.isfinite(centers[:, 0]) & np.isfinite(centers[:, 1])
    if np.sum(ok) < 3:
        return H_to_ref

    xs = centers[ok, 0]
    ys = centers[ok, 1]

    x_mean = float(np.mean(xs))
    denom = float(np.sum((xs - x_mean) ** 2) + 1e-12)
    a = float(np.sum((xs - x_mean) * (ys - float(np.mean(ys)))) / denom)

    if abs(a) < 1e-4:
        return H_to_ref

    C = np.array([[1, 0, 0],
                  [-a, 1, a * x_mean],
                  [0, 0, 1]], dtype=np.float64)

    if verbose:
        print(f"[drift] slope a={a:.6f} -> apply correction")

    return [C @ H for H in H_to_ref]

def global_transform_sanity(H_to_ref: List[np.ndarray], images_rgb: List[np.ndarray], verbose: bool = True) -> List[np.ndarray]:
    """
    Prevent one image from flying far away (causes huge black canvas + pop-up).
    If a transform is wildly large, replace it with neighbor translation-ish.
    """
    n = len(images_rgb)
    if n < 2:
        return H_to_ref

    ref = n // 2
    href, wref = images_rgb[ref].shape[:2]
    c_ref = np.array([[0.5*(wref-1), 0.5*(href-1)]], dtype=np.float64)
    p_ref = apply_homography(H_to_ref[ref], c_ref)[0]

    out = [H.copy() for H in H_to_ref]
    prev_good = ref

    for i in range(n):
        h, w = images_rgb[i].shape[:2]
        c = np.array([[0.5*(w-1), 0.5*(h-1)]], dtype=np.float64)
        p = apply_homography(out[i], c)[0]
        if not np.all(np.isfinite(p)):
            bad = True
        else:
            dx = float(p[0] - p_ref[0])
            dy = float(p[1] - p_ref[1])
            bad = (abs(dx) > 6.0*wref) or (abs(dy) > 6.0*href)

        if bad:
            # replace with neighbor-based mild translation
            j = prev_good
            hj, wj = images_rgb[j].shape[:2]
            step = np.array([[1, 0, 0.9*wj*(i-j)],
                             [0, 1, 0],
                             [0, 0, 1]], dtype=np.float64)
            out[i] = step @ out[j]
            if verbose:
                print(f"[sanity] img {i} transform too far -> soften using neighbor {j}")
        else:
            prev_good = i

    return out


# ============================================================
# Build global transforms
# ============================================================

def build_global_transforms(images_rgb: List[np.ndarray],
                            verbose: bool = True,
                            debug_matches_dir: Optional[str] = None,
                            min_inliers: int = 16) -> List[np.ndarray]:
    n = len(images_rgb)
    ref = n // 2

    pair = [None] * (n - 1)

    for i in range(n - 1):
        A = images_rgb[i]
        B = images_rgb[i + 1]

        H_B_to_A, ptsB, ptsA, inl_cnt = estimate_pair_homography(
            imgA_rgb=A, imgB_rgb=B,
            debug_dir=debug_matches_dir,
            pair_name=f"{i:02d}_{i+1:02d}",
            max_corners=900,
            nms_radius=8,
            patch_radius=18,
            desc_size=8,
            ransac_iters=1800,
            ransac_thresh=4.0,
            match_ratio=0.80,
            max_dim_match=750,
            use_hist_eq=True
        )

        hA, wA = A.shape[:2]
        hB, wB = B.shape[:2]

        ok = (H_B_to_A is not None) and is_homography_reasonable(H_B_to_A, hA, wA)

        if ok:
            ov = overlap_ratio_A_with_warpedB(H_B_to_A, hA, wA, hB, wB)
            if ov < 0.07:
                ok = False
                if verbose:
                    print(f"[pair {i}->{i+1}] overlap too small ({ov*100:.1f}%) -> mark bad")

        if ok and inl_cnt < int(min_inliers):
            ok = False
            if verbose:
                print(f"[pair {i}->{i+1}] inliers too small ({inl_cnt}) -> mark bad")

        if not ok:
            # ✅ NEW: try affine fallback before translation
            if ptsB is not None and ptsA is not None and ptsB.shape[0] >= 18:
                H_aff, inl_aff = ransac_affine(ptsB, ptsA, iters=900, thresh=4.5, seed=42)
                if H_aff is not None and is_homography_reasonable(H_aff, hA, wA, max_persp=1e-6):
                    H_B_to_A = H_aff
                    if verbose:
                        print(f"[pair {i}->{i+1}] bad H -> affine fallback (inliers {int(np.sum(inl_aff))})")
                else:
                    H_B_to_A = translation_fallback_from_matches(ptsB, ptsA)
                    if verbose:
                        print(f"[pair {i}->{i+1}] bad H -> translation fallback (median)")
            elif ptsB is not None and ptsA is not None and ptsB.shape[0] >= 12:
                H_B_to_A = translation_fallback_from_matches(ptsB, ptsA)
                if verbose:
                    print(f"[pair {i}->{i+1}] bad H -> translation fallback (median)")
            else:
                dx = float(wA) * 0.9
                H_B_to_A = np.array([[1, 0, -dx],
                                     [0, 1, 0],
                                     [0, 0, 1]], dtype=np.float64)
                if verbose:
                    print(f"[pair {i}->{i+1}] fail -> hard translation fallback")

        pair[i] = H_B_to_A.astype(np.float64)

        if verbose:
            print(f"[pair] H({i+1}-> {i}) ready")

    H_to_ref = [np.eye(3, dtype=np.float64) for _ in range(n)]
    H_to_ref[ref] = np.eye(3, dtype=np.float64)

    for i in range(ref + 1, n):
        H = np.eye(3, dtype=np.float64)
        for k in range(i, ref, -1):
            H = H @ pair[k - 1]
        H_to_ref[i] = H

    for i in range(ref - 1, -1, -1):
        H = np.eye(3, dtype=np.float64)
        for k in range(i, ref):
            try:
                inv_step = np.linalg.inv(pair[k])
            except np.linalg.LinAlgError:
                wK = float(images_rgb[k].shape[1])
                inv_step = np.array([[1, 0, wK * 0.9],
                                     [0, 1, 0],
                                     [0, 0, 1]], dtype=np.float64)
            H = H @ inv_step
        H_to_ref[i] = H

    H_to_ref = apply_drift_correction(H_to_ref, images_rgb, verbose=verbose)
    H_to_ref = global_transform_sanity(H_to_ref, images_rgb, verbose=verbose)

    return H_to_ref


# ============================================================
# Fast hole filling (vectorized)
# ============================================================

def fill_small_holes_fast(pano: np.ndarray, wsum: np.ndarray,
                          hole_thr: float = 1e-3,
                          iters: int = 8,
                          min_nbr: int = 2) -> np.ndarray:
    out = pano.astype(np.float32).copy()
    mask = (wsum > hole_thr)

    for _ in range(int(iters)):
        holes = ~mask
        if not np.any(holes):
            break

        m = mask.astype(np.float32)

        up = np.pad(m[0:-1, :], ((1, 0), (0, 0)), mode="constant")
        dn = np.pad(m[1:, :],   ((0, 1), (0, 0)), mode="constant")
        lf = np.pad(m[:, 0:-1], ((0, 0), (1, 0)), mode="constant")
        rt = np.pad(m[:, 1:],   ((0, 0), (0, 1)), mode="constant")
        cnt = up + dn + lf + rt

        def shift_img(img, dy, dx):
            H, W = img.shape[:2]
            out2 = np.zeros_like(img)
            y0 = max(0, dy); y1 = H + min(0, dy)
            x0 = max(0, dx); x1 = W + min(0, dx)
            out2[y0:y1, x0:x1] = img[y0-dy:y1-dy, x0-dx:x1-dx]
            return out2

        s = (
            shift_img(out * m[:, :, None], 1, 0) +
            shift_img(out * m[:, :, None], -1, 0) +
            shift_img(out * m[:, :, None], 0, 1) +
            shift_img(out * m[:, :, None], 0, -1)
        )

        fill = holes & (cnt >= float(min_nbr))
        if not np.any(fill):
            break

        out[fill] = (s[fill] / (cnt[fill][:, None] + 1e-8)).astype(np.float32)
        mask[fill] = True

    return clamp01(out)


# ============================================================
# Stitching (tile warp + work_scale)
# ============================================================

def stitch_images(images_rgb: List[np.ndarray],
                  max_canvas_side: int = 9000,
                  work_scale: float = 1.0,
                  feather: float = 35.0,
                  crop: bool = True,
                  hole_fill: bool = True,
                  hole_fill_iters: int = 8,
                  verbose: bool = True,
                  tile: int = 384,
                  debug_matches_dir: Optional[str] = None,
                  min_inliers: int = 16) -> Optional[np.ndarray]:

    n = len(images_rgb)
    if n == 0:
        return None
    if n == 1:
        return images_rgb[0]

    H_to_ref = build_global_transforms(
        images_rgb,
        verbose=verbose,
        debug_matches_dir=debug_matches_dir,
        min_inliers=min_inliers
    )

    bboxes = []
    for i, img in enumerate(images_rgb):
        h, w = img.shape[:2]
        bboxes.append(compute_warp_bbox(H_to_ref[i], h, w))

    minx = min(b[0] for b in bboxes)
    miny = min(b[1] for b in bboxes)
    maxx = max(b[2] for b in bboxes)
    maxy = max(b[3] for b in bboxes)

    tx = -minx
    ty = -miny
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)

    H_to_canvas = [T @ H for H in H_to_ref]

    out_w = int(math.ceil(maxx - minx + 1))
    out_h = int(math.ceil(maxy - miny + 1))

    if out_w > max_canvas_side or out_h > max_canvas_side:
        scale = max(out_w, out_h) / float(max_canvas_side)
        out_w = int(out_w / scale)
        out_h = int(out_h / scale)
        S = np.array([[1 / scale, 0, 0],
                      [0, 1 / scale, 0],
                      [0, 0, 1]], dtype=np.float64)
        H_to_canvas = [S @ Hc for Hc in H_to_canvas]
        if verbose:
            print(f"[canvas clamp] scale={scale:.3f} -> {out_w}x{out_h}")

    if work_scale < 1.0:
        ws = float(work_scale)
        out_w2 = max(2, int(round(out_w * ws)))
        out_h2 = max(2, int(round(out_h * ws)))
        S2 = np.array([[ws, 0, 0],
                       [0, ws, 0],
                       [0, 0, 1]], dtype=np.float64)
        H_to_canvas = [S2 @ Hc for Hc in H_to_canvas]
        out_w, out_h = out_w2, out_h2
        if verbose:
            print(f"[work scale] {ws:.3f} -> {out_w}x{out_h}")

    acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wsum = np.zeros((out_h, out_w), dtype=np.float32)

    wmaps = [make_feather_weight(im.shape[0], im.shape[1], feather=feather, base=0.12) for im in images_rgb]

    for i, img in enumerate(images_rgb):
        h, w = img.shape[:2]
        Hc = H_to_canvas[i]
        try:
            Hinv = np.linalg.inv(Hc)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"[skip] img {i} singular H")
            continue

        bx0, by0, bx1, by1 = compute_warp_bbox(Hc, h, w)
        x0 = max(0, int(math.floor(bx0)))
        y0 = max(0, int(math.floor(by0)))
        x1 = min(out_w - 1, int(math.ceil(bx1)))
        y1 = min(out_h - 1, int(math.ceil(by1)))
        if x1 <= x0 or y1 <= y0:
            continue

        total_written = 0

        for yy0 in range(y0, y1 + 1, tile):
            yy1 = min(y1, yy0 + tile - 1)
            for xx0 in range(x0, x1 + 1, tile):
                xx1 = min(x1, xx0 + tile - 1)

                YY, XX = np.mgrid[yy0:yy1 + 1, xx0:xx1 + 1]
                pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1).astype(np.float64)

                src = project_points(Hinv, pts)
                xs = src[:, 0].astype(np.float32)
                ys = src[:, 1].astype(np.float32)

                colors, inside = bilinear_sample_rgb(img, xs, ys)
                weights, inside_w = bilinear_sample_gray(wmaps[i], xs, ys)

                inside = inside & inside_w
                if not np.any(inside):
                    continue

                xi = pts[inside, 0].astype(np.int32)
                yi = pts[inside, 1].astype(np.int32)

                c2 = colors[inside]
                w2 = weights[inside]

                acc[yi, xi, :] += c2 * w2[:, None]
                wsum[yi, xi] += w2

                total_written += int(np.sum(inside))

        if verbose:
            print(f"[warp] img {i} -> wrote {total_written} px")

    pano = acc / (wsum[:, :, None] + 1e-8)
    pano = clamp01(pano)

    if hole_fill:
        pano = fill_small_holes_fast(pano, wsum, hole_thr=1e-3, iters=hole_fill_iters, min_nbr=2)
        pano = clamp01(pano)

    if crop:
        mask = wsum > 1e-3
        if np.any(mask):
            ys, xs = np.where(mask)
            y0 = int(np.min(ys)); y1 = int(np.max(ys))
            x0 = int(np.min(xs)); x1 = int(np.max(xs))
            pano = pano[y0:y1 + 1, x0:x1 + 1, :]

    return pano


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input folder")
    parser.add_argument("--pattern", required=True, help='glob pattern, e.g. "testimg*.jpg"')
    parser.add_argument("--out", required=True, help="output image path")
    parser.add_argument("--max_canvas", type=int, default=6500, help="max canvas side (safety clamp)")
    parser.add_argument("--work_scale", type=float, default=0.55,
                        help="additional downscale for speed (0.5~1.0). 1.0 = full.")
    parser.add_argument("--feather", type=float, default=45.0, help="feather ramp in pixels")
    parser.add_argument("--tile", type=int, default=384, help="warp tile size")
    parser.add_argument("--hole_iters", type=int, default=6, help="hole fill iterations (fast)")
    parser.add_argument("--min_inliers", type=int, default=16, help="minimum inliers to accept homography")
    parser.add_argument("--no_crop", action="store_true", help="disable auto-crop")
    parser.add_argument("--no_hole_fill", action="store_true", help="disable hole filling")
    parser.add_argument("--debug_matches_dir", default="", help="save match visualization images to this dir")
    parser.add_argument("--show", action="store_true", help="show result using cv2.imshow (optional)")
    parser.add_argument("--quiet", action="store_true", help="less logs")
    args = parser.parse_args()

    imgs, paths = load_images_from_folder(args.input, args.pattern)
    if len(imgs) == 0:
        print("No images found.")
        return

    print(f"Loaded {len(imgs)} images (natural sorted):")
    for p in paths:
        print(" -", p)

    debug_dir = args.debug_matches_dir.strip() if args.debug_matches_dir.strip() else None

    pano = stitch_images(
        imgs,
        max_canvas_side=args.max_canvas,
        work_scale=float(args.work_scale),
        feather=float(args.feather),
        crop=(not args.no_crop),
        hole_fill=(not args.no_hole_fill),
        hole_fill_iters=int(args.hole_iters),
        verbose=(not args.quiet),
        tile=int(args.tile),
        debug_matches_dir=debug_dir,
        min_inliers=int(args.min_inliers)
    )
    if pano is None:
        print("Stitch failed.")
        return

    out_u8 = to_u8_rgb(pano)
    save_image_rgb(args.out, out_u8)
    print("Saved:", args.out)

    if args.show:
        bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
        cv2.imshow("panorama", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
