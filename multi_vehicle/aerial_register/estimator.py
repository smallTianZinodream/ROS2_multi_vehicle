"""残差分析引导的单应矩阵估计。

对应 Cai 等，Remote Sensing 2018, 10(2):291。
固定窗口的 Multi-GS 在低空航拍上容易受重复纹理和视角倾斜影响。
这里实现文中的三种相似度：

- rw：松弛窗口，在一段窗口长度里取最高重叠比例
- sw：滑动窗口，重叠结构数固定，窗口长度可变
- rro：逆序残差，重叠之外再惩罚残差名次颠倒
"""

import time

import cv2
import numpy as np


def _fit_homography(src, dst):
    if len(src) < 4:
        return None
    homography, _mask = cv2.findHomography(src, dst, 0)
    if homography is None or not np.isfinite(homography).all():
        return None
    return homography


def _residuals(homographies, src, dst):
    errors = np.empty((len(src), len(homographies)), dtype=np.float64)
    points = src.reshape(-1, 1, 2).astype(np.float64)
    for index, homography in enumerate(homographies):
        projected = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
        errors[:, index] = np.linalg.norm(projected - dst, axis=1)
    return errors


def _overlap(order, width):
    count, models = order.shape
    presence = np.zeros((count, models), dtype=np.float32)
    rows = np.arange(count)[:, None]
    presence[rows, order[:, :width]] = 1.0
    return presence @ presence.T


def similarity_from_order(order, kind):
    count, models = order.shape
    if kind == 'multigs':
        width = max(1, int(round(0.1 * models)))
        return _overlap(order, width) / width
    if kind == 'rw':
        start = max(1, int(round(0.08 * models)))
        extra = max(1, int(round(0.12 * models)))
        best = np.zeros((count, count), dtype=np.float32)
        for width in range(start, min(models, start + extra) + 1):
            best = np.maximum(best, _overlap(order, width) / width)
        return best
    if kind == 'sw':
        target = max(1, int(round(0.05 * models)))
        found = np.full((count, count), models, dtype=np.float32)
        for width in range(1, models + 1):
            reached = _overlap(order, width) >= target
            update = reached & (found == models)
            found[update] = width
            if not np.any(found == models):
                break
        return target / found
    if kind == 'rro':
        width = max(4, int(round(0.1 * models)))
        penalty = 1.0 / (width * width)
        score = np.zeros((count, count), dtype=np.float32)
        rank = np.empty_like(order)
        rank[np.arange(count)[:, None], order] = np.arange(models)[None, :]
        tops = order[:, :width]
        for i in range(count):
            set_i = set(tops[i].tolist())
            for j in range(i + 1, count):
                ordered = [model for model in tops[i].tolist() if model in set(tops[j].tolist())]
                reverses = 0
                for left in range(len(ordered)):
                    for right in range(left + 1, len(ordered)):
                        if rank[j, ordered[left]] > rank[j, ordered[right]]:
                            reverses += 1
                value = max(0.0, len(ordered) / width - penalty * reverses)
                score[i, j] = score[j, i] = value
        return score
    raise ValueError(f'未知相似度: {kind}')


def _sample_indices(similarity, sample_size, rng):
    count = similarity.shape[0]
    chosen = [int(rng.integers(0, count))]
    for _ in range(sample_size - 1):
        weight = np.ones(count, dtype=np.float64)
        for index in chosen:
            weight *= np.maximum(similarity[index], 0.0)
        weight[chosen] = 0.0
        total = weight.sum()
        if total <= 1e-12:
            weight = np.ones(count, dtype=np.float64)
            weight[chosen] = 0.0
            total = weight.sum()
        weight /= total
        chosen.append(int(rng.choice(count, p=weight)))
    return np.array(chosen, dtype=int)


def _fit_fundamental(src, dst):
    if len(src) < 8:
        return None
    model, _mask = cv2.findFundamentalMat(src, dst, cv2.FM_8POINT)
    if model is None or model.shape != (3, 3) or not np.isfinite(model).all():
        return None
    return model


def _epipolar_residuals(models, src, dst):
    ones = np.ones((len(src), 1), dtype=np.float64)
    left = np.concatenate([src, ones], axis=1)
    right = np.concatenate([dst, ones], axis=1)
    errors = np.empty((len(src), len(models)), dtype=np.float64)
    for index, model in enumerate(models):
        left_line = left @ model.T
        right_line = right @ model
        signed = np.sum(right * left_line, axis=1)
        denom = np.sqrt(
            left_line[:, 0] ** 2 + left_line[:, 1] ** 2
            + right_line[:, 0] ** 2 + right_line[:, 1] ** 2
        )
        errors[:, index] = np.abs(signed) / np.maximum(denom, 1e-12)
    return errors


def _estimate_models(src, dst, kind, sample_size, fit, residual_fn, batches, batch_size, threshold, seed):
    rng = np.random.default_rng(seed)
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    started = time.perf_counter()
    models = []
    for _ in range(batch_size):
        pick = rng.choice(len(src), size=sample_size, replace=False)
        model = fit(src[pick], dst[pick])
        if model is not None:
            models.append(model)
    best_model = None
    best_mask = np.zeros(len(src), dtype=bool)
    if not models:
        return None, best_mask, time.perf_counter() - started

    for _ in range(batches):
        if len(models) > 50:
            scores = (residual_fn(models, src, dst) < threshold).sum(axis=0)
            keep = np.argsort(scores)[-50:]
            models = [models[index] for index in keep]
        order = np.argsort(residual_fn(models, src, dst), axis=1)
        if kind == 'ransac':
            similarity = np.ones((len(src), len(src)), dtype=np.float32)
        else:
            similarity = similarity_from_order(order, kind)
        np.fill_diagonal(similarity, 0.0)
        for _sample in range(batch_size):
            if kind == 'ransac':
                pick = rng.choice(len(src), size=sample_size, replace=False)
            else:
                pick = _sample_indices(similarity, sample_size, rng)
            model = fit(src[pick], dst[pick])
            if model is None:
                continue
            mask = residual_fn([model], src, dst)[:, 0] < threshold
            if mask.sum() > best_mask.sum():
                best_model = model
                best_mask = mask
            models.append(model)
    return best_model, best_mask, time.perf_counter() - started


def estimate_homography(
    src,
    dst,
    kind='rro',
    batches=5,
    batch_size=8,
    inlier_threshold=3.0,
    seed=0,
):
    """用残差相似度引导采样，返回最佳单应矩阵、内点和耗时。"""
    return _estimate_models(
        src, dst, kind, 4, _fit_homography, _residuals,
        batches, batch_size, inlier_threshold, seed,
    )


def estimate_fundamental(
    src,
    dst,
    kind='rro',
    batches=8,
    batch_size=12,
    inlier_threshold=2.5,
    seed=0,
):
    """8 点基础矩阵。两个平面上的点都应满足同一条极线约束。"""
    return _estimate_models(
        src, dst, kind, 8, _fit_fundamental, _epipolar_residuals,
        batches, batch_size, inlier_threshold, seed,
    )


def estimate_multi_homography(
    src,
    dst,
    kind='rro',
    planes=2,
    min_inliers=12,
    batches=5,
    batch_size=12,
    inlier_threshold=4.0,
    seed=0,
):
    """抽出一个平面的内点后，在剩下的匹配里再估下一个平面。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    remaining = np.ones(len(src), dtype=bool)
    found = []
    for index in range(planes):
        chosen = np.flatnonzero(remaining)
        if len(chosen) < 8:
            break
        model, mask, elapsed = estimate_homography(
            src[chosen], dst[chosen], kind=kind, batches=batches,
            batch_size=batch_size, inlier_threshold=inlier_threshold,
            seed=seed + index,
        )
        if model is None or int(mask.sum()) < min_inliers:
            break
        full_mask = np.zeros(len(src), dtype=bool)
        full_mask[chosen[mask]] = True
        found.append((model, full_mask, elapsed))
        remaining[full_mask] = False
    return found


def triangulate(fundamental, intrinsics, src, dst, mask):
    """由基础矩阵恢复第二台相机的姿态，并把内点三角化到第一台相机坐标系。"""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    chosen = np.flatnonzero(mask)
    if fundamental is None or len(chosen) < 8:
        return None
    essential = intrinsics.T @ fundamental @ intrinsics
    _count, rotation, translation, pose_mask = cv2.recoverPose(
        essential, src[chosen], dst[chosen], intrinsics)
    pose_mask = pose_mask.ravel().astype(bool)
    if int(pose_mask.sum()) < 8:
        return None
    chosen = chosen[pose_mask]
    projection_1 = intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_2 = intrinsics @ np.hstack([rotation, translation.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_1, projection_2, src[chosen].T, dst[chosen].T)
    points = (homogeneous[:3] / homogeneous[3]).T
    finite = np.isfinite(points).all(axis=1)
    return points[finite], rotation, translation.ravel(), chosen[finite]


def fundamental_from_homographies(ground_h, facade_h):
    """两个平面的单应矩阵共用一个极点，用它确定基础矩阵。"""
    try:
        values = np.linalg.eigvals(np.linalg.inv(facade_h) @ ground_h)
    except np.linalg.LinAlgError:
        return None
    best = None
    best_ratio = 1e9
    for value in values:
        if abs(value.imag) > 1e-5:
            continue
        matrix = ground_h - float(value.real) * facade_h
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[0] < 1e-8:
            continue
        ratio = singular[-1] / singular[0]
        if ratio < best_ratio:
            best_ratio = ratio
            best = matrix
    if best is None:
        return None
    left, _singular, _right = np.linalg.svd(best)
    epipole = left[:, -1]
    model = _skew_matrix(epipole) @ ground_h
    norm = np.linalg.norm(model)
    if norm < 1e-12 or not np.isfinite(model).all():
        return None
    return model / norm


def _skew_matrix(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def mean_epipolar_error(model, src, dst):
    if model is None or len(src) == 0:
        return float('inf')
    return float(_epipolar_residuals([model], src, dst).mean())


def mean_transfer_error(homography, src, dst):
    if homography is None or len(src) == 0:
        return float('inf')
    projected = cv2.perspectiveTransform(
        np.asarray(src, dtype=np.float64).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    return float(np.linalg.norm(projected - np.asarray(dst), axis=1).mean())
