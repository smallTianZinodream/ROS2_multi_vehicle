"""生成一对带倾斜和重复纹理的低空航拍，比较几种变换估计。"""

import sys
from pathlib import Path

import cv2
import numpy as np

from estimator import (
    estimate_fundamental,
    estimate_multi_homography,
    mean_epipolar_error,
    mean_transfer_error,
    triangulate,
)


def make_scene(width=640, height=480, seed=3):
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), (78, 118, 68), dtype=np.uint8)
    for _ in range(70):
        start = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        end = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(value) for value in rng.integers(30, 210, size=3))
        cv2.line(canvas, start, end, color, int(rng.integers(1, 3)))
    for index in range(28):
        x = int(rng.integers(24, width - 80))
        y = int(rng.integers(24, height - 60))
        bw = int(rng.integers(22, 74))
        bh = int(rng.integers(16, 48))
        color = tuple(int(value) for value in rng.integers(40, 200, size=3))
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), color, -1)
        cv2.putText(
            canvas, str(index), (x + 3, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 250, 250), 1)
    cv2.circle(canvas, (160, 120), 26, (40, 80, 190), -1)
    cv2.rectangle(canvas, (430, 250), (520, 340), (30, 55, 140), -1)
    return canvas


FACADE = np.array([[420, 160], [580, 160], [580, 340], [420, 340]], dtype=np.float64)
GROUND_GCP = np.array([[120, 90], [260, 220], [500, 110], [180, 390]], dtype=np.float64)


def _skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def _induced_homography(intrinsics, rotation, translation, normal, distance):
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    return intrinsics @ (rotation - np.outer(translation, normal) / distance) @ np.linalg.inv(intrinsics)


def project(homography, points):
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)


# 同一格的第二张照片：相机转了一个不同的角度，平移方向也换一条基线。
SECOND_ROTATION = (6.0, 1.0, -4.0)
SECOND_TRANSLATION = (1.2, -3.8, 0.7)


def make_pair(scene, with_facade=True, rotation_deg=(4.0, -2.0, 1.0), translation=(4.0, 0.8, 0.3)):
    """地面和立面各有一个单应矩阵，二者来自同一次相机运动，因此共用一个基础矩阵。"""
    height, width = scene.shape[:2]
    intrinsics = np.array([
        [700.0, 0.0, width / 2.0],
        [0.0, 700.0, height / 2.0],
        [0.0, 0.0, 1.0],
    ])
    rotation, _ = cv2.Rodrigues(np.deg2rad(np.array(rotation_deg, dtype=np.float64)))
    translation = np.array(translation, dtype=np.float64)
    ground_h = _induced_homography(intrinsics, rotation, translation, [0.0, 0.0, 1.0], 50.0)
    facade_h = _induced_homography(intrinsics, rotation, translation, [1.0, 0.0, 0.0], 10.0)
    fundamental = np.linalg.inv(intrinsics).T @ _skew(translation) @ rotation @ np.linalg.inv(intrinsics)

    ground_view = cv2.warpPerspective(scene, ground_h, (width, height), borderValue=(40, 70, 35))
    view = ground_view.copy()
    if with_facade:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, FACADE.astype(np.int32), 255)
        facade_only = cv2.bitwise_and(scene, scene, mask=mask)
        facade_view = cv2.warpPerspective(facade_only, facade_h, (width, height))
        facade_mask = cv2.warpPerspective(mask, facade_h, (width, height))
        view[facade_mask > 128] = facade_view[facade_mask > 128]
    truth = {
        'ground_h': ground_h,
        'facade_h': facade_h if with_facade else None,
        'fundamental': fundamental,
        'ground_gcp': GROUND_GCP,
        'ground_gcp_dst': project(ground_h, GROUND_GCP),
        'facade_gcp': FACADE if with_facade else np.zeros((0, 2)),
        'facade_gcp_dst': project(facade_h, FACADE) if with_facade else np.zeros((0, 2)),
        'intrinsics': intrinsics,
        'rotation': rotation,
        'translation': translation,
        'ground_depth': 50.0,
        'facade_distance': 10.0,
        'with_facade': with_facade,
    }
    return view, truth


def _inside_facade(points):
    polygon = FACADE.astype(np.float32)
    return np.array([
        cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
        for x, y in points
    ])


def match_points(
    image_a, image_b, truth, target_per_plane=32, outlier_count=90, seed=7, noise=0.4,
    extra_ground_h=None, extra_facade_h=None,
):
    gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    keypoints = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=8)
    if keypoints is None:
        keypoints = np.zeros((0, 1, 2), dtype=np.float32)
    points = keypoints.reshape(-1, 2)
    on_facade = _inside_facade(points) if len(points) else np.zeros(0, dtype=bool)
    height, width = image_a.shape[:2]
    rng = np.random.default_rng(seed)

    def keep(src, homography, limit, extra=None):
        if len(src) == 0 or homography is None:
            empty = np.zeros((0, 2))
            return empty, empty
        dst = project(homography, src)
        inside = (
            (dst[:, 0] > 8) & (dst[:, 0] < width - 8)
            & (dst[:, 1] > 8) & (dst[:, 1] < height - 8)
        )
        if extra is not None:
            other = project(extra, src)
            inside &= (
                (other[:, 0] > 8) & (other[:, 0] < width - 8)
                & (other[:, 1] > 8) & (other[:, 1] < height - 8)
            )
        src, dst = src[inside], dst[inside]
        if len(src) > limit:
            pick = rng.choice(len(src), limit, replace=False)
            src, dst = src[pick], dst[pick]
        dst = dst + rng.normal(0.0, noise, size=dst.shape)
        return src, dst

    ground_src, ground_dst = keep(
        points[~on_facade], truth['ground_h'], target_per_plane, extra_ground_h)
    if truth.get('facade_h') is None:
        facade_src, facade_dst = np.zeros((0, 2)), np.zeros((0, 2))
    else:
        facade_src, facade_dst = keep(
            points[on_facade], truth['facade_h'], target_per_plane, extra_facade_h)
    parts_src = [item for item in (ground_src, facade_src) if len(item)]
    parts_dst = [item for item in (ground_dst, facade_dst) if len(item)]
    src = np.vstack(parts_src) if parts_src else np.zeros((0, 2))
    dst = np.vstack(parts_dst) if parts_dst else np.zeros((0, 2))
    plane = np.array(['ground'] * len(ground_src) + ['facade'] * len(facade_src))
    false_src = np.column_stack([
        rng.uniform(16, width - 16, outlier_count),
        rng.uniform(16, height - 16, outlier_count),
    ])
    false_dst = np.column_stack([
        rng.uniform(16, width - 16, outlier_count),
        rng.uniform(16, height - 16, outlier_count),
    ])
    src = np.vstack([src, false_src])
    dst = np.vstack([dst, false_dst])
    plane = np.concatenate([plane, np.array(['outlier'] * outlier_count)])
    order = rng.permutation(len(src))
    return src[order], dst[order], plane[order]


def _panel(image, title):
    framed = image.copy()
    cv2.rectangle(framed, (0, 0), (framed.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(framed, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    return framed


def _draw_points(image, points, color):
    canvas = image.copy()
    for x, y in np.round(points).astype(int):
        cv2.drawMarker(canvas, (int(x), int(y)), color, cv2.MARKER_CROSS, 14, 2)
    return canvas


def save_figure(path, image_a, image_b, truth, planes, src, dst, plane_names):
    height, width = image_a.shape[:2]
    marked = _draw_points(image_a, truth['ground_gcp'], (0, 220, 0))
    marked = _draw_points(marked, truth['facade_gcp'], (0, 140, 255))
    cv2.polylines(marked, [FACADE.astype(np.int32)], True, (0, 140, 255), 2)
    colors = {'ground': (40, 200, 60), 'facade': (0, 140, 255), 'outlier': (40, 40, 200)}
    matches = np.zeros((height, width * 2, 3), dtype=np.uint8)
    matches[:, :width] = image_a
    matches[:, width:] = image_b
    for name in ('outlier', 'ground', 'facade'):
        indexes = np.flatnonzero(plane_names == name)
        if name == 'outlier' and len(indexes) > 30:
            indexes = indexes[:30]
        for index in indexes:
            p1 = tuple(np.round(src[index]).astype(int))
            p2 = tuple(np.round(dst[index]).astype(int) + np.array([width, 0]))
            cv2.line(matches, p1, p2, colors[name], 1)
    overlay = image_b.copy()
    if 'ground' in planes:
        warped = cv2.warpPerspective(image_a, planes['ground'], (width, height))
        overlay = cv2.addWeighted(overlay, 0.4, warped, 0.6, 0)
    top = np.hstack([
        _panel(marked, 'view 1  green ground GCP, orange facade'),
        _panel(image_b, 'view 2  two planes'),
    ])
    bottom = np.hstack([
        _panel(overlay, 'ground plane registered'),
        _panel(cv2.resize(matches, (width, height)), 'green ground, orange facade, red outlier'),
    ])
    cv2.imwrite(str(path), np.vstack([top, bottom]))


def _assign_planes(found, truth):
    catalog = {
        'ground': (truth['ground_gcp'], truth['ground_gcp_dst']),
    }
    if truth.get('facade_h') is not None:
        catalog['facade'] = (truth['facade_gcp'], truth['facade_gcp_dst'])
    unused = set(catalog)
    assigned = {}
    for model, _mask, _elapsed in found:
        best_name = None
        best_error = float('inf')
        for name in unused:
            src, dst = catalog[name]
            error = mean_transfer_error(model, src, dst)
            if error < best_error:
                best_error = error
                best_name = name
        if best_name is None:
            continue
        unused.remove(best_name)
        assigned[best_name] = (model, best_error)
    return assigned


def _angle_between(first, second):
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < 1e-12:
        return 90.0
    cosine = float(np.dot(first, second) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.rad2deg(np.arccos(cosine)))


def _rotation_error(estimated, truth):
    relative = estimated.T @ truth
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = max(-1.0, min(1.0, float(cosine)))
    return float(np.rad2deg(np.arccos(cosine)))


def save_cloud(path, points, labels):
    def scatter(axis_x, axis_y, title):
        canvas = np.full((420, 520, 3), 32, dtype=np.uint8)
        values_x = points[:, axis_x]
        values_y = points[:, axis_y]
        span_x = max(float(values_x.max() - values_x.min()), 1.0)
        span_y = max(float(values_y.max() - values_y.min()), 1.0)
        px = (40 + (values_x - values_x.min()) / span_x * 440).astype(int)
        py = (380 - (values_y - values_y.min()) / span_y * 320).astype(int)
        colors = {'ground': (60, 200, 60), 'facade': (0, 150, 255), 'outlier': (40, 40, 180)}
        for index, label in enumerate(labels):
            cv2.circle(canvas, (int(px[index]), int(py[index])), 3, colors.get(label, (180, 180, 180)), -1)
        cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
        return canvas

    figure = np.hstack([
        scatter(0, 2, 'side  X-Z   green ground, orange facade'),
        scatter(0, 1, 'top   X-Y'),
    ])
    cv2.imwrite(str(path), figure)


def _fundamental_from_planes(src, dst, found, seed):
    """全图和平面内点各估一次基础矩阵，留下极线残差更小的那个。"""
    if len(src) < 8:
        return None, np.zeros(len(src), dtype=bool)
    full_model, full_mask, _elapsed = estimate_fundamental(src, dst, kind='rro', seed=seed)
    mask = np.zeros(len(src), dtype=bool)
    for _model, inliers, _elapsed in found:
        mask |= inliers
    chosen = np.flatnonzero(mask)
    if full_model is None or len(chosen) < 16:
        return full_model, full_mask
    plane_model, local_mask, _elapsed = estimate_fundamental(
        src[chosen], dst[chosen], kind='rro', seed=seed + 1)
    if plane_model is None:
        return full_model, full_mask
    plane_mask = np.zeros(len(src), dtype=bool)
    plane_mask[chosen[local_mask]] = True
    if int(plane_mask.sum()) < 8:
        return full_model, full_mask
    full_error = mean_epipolar_error(full_model, src[chosen], dst[chosen])
    plane_error = mean_epipolar_error(plane_model, src[chosen], dst[chosen])
    if plane_error <= full_error:
        return plane_model, plane_mask
    return full_model, full_mask


def _flat_points(src, mask):
    chosen = src[mask] if int(mask.sum()) >= 8 else src
    center = np.median(chosen, axis=0)
    local = chosen - center
    return np.column_stack([
        local[:, 0] * 0.01,
        local[:, 1] * 0.01,
        np.full(len(chosen), 50.0),
    ])


def _survey_once(
    scene, truth, seed, with_facade, noise, match_seed,
    extra_ground_h=None, extra_facade_h=None,
):
    """用一对影像做多平面单应和基础矩阵。extra_* 要求特征在第二张里也看得见。"""
    view = None
    src, dst, plane = match_points(
        scene, scene, truth, seed=match_seed, noise=noise,
        extra_ground_h=extra_ground_h, extra_facade_h=extra_facade_h,
    )
    if len(src) < 8:
        return None
    found = estimate_multi_homography(
        src, dst, kind='rro', planes=2 if with_facade else 1, batches=6, batch_size=16,
        inlier_threshold=3.0, seed=seed,
    )
    assigned = _assign_planes(found, truth)
    fundamental, f_mask = _fundamental_from_planes(src, dst, found, seed + 1)
    gcp_parts_src = [truth['ground_gcp']]
    gcp_parts_dst = [truth['ground_gcp_dst']]
    if truth.get('facade_h') is not None:
        gcp_parts_src.append(truth['facade_gcp'])
        gcp_parts_dst.append(truth['facade_gcp_dst'])
    gcp_src = np.vstack(gcp_parts_src)
    gcp_dst = np.vstack(gcp_parts_dst)
    cloud = None if fundamental is None else triangulate(
        fundamental, truth['intrinsics'], src, dst, f_mask)
    if cloud is None and with_facade:
        return None
    if cloud is None:
        ground_mask = np.zeros(len(src), dtype=bool)
        for _model, inliers, _elapsed in found:
            ground_mask |= inliers
        points = _flat_points(src, ground_mask)
        labels = np.array(['ground'] * len(points))
        rotation = np.eye(3)
        translation = np.zeros(3)
    else:
        points, rotation, translation, indexes = cloud
        labels = plane[indexes]
        ground = labels == 'ground'
        if int(ground.sum()) >= 5 and abs(float(np.median(points[ground, 2]))) > 1e-6:
            points = points * (truth['ground_depth'] / np.median(points[ground, 2]))
    ground = labels == 'ground'
    facade = labels == 'facade'
    if not with_facade and (
        not ground.any() or float(np.median(np.abs(points[ground, 2] - truth['ground_depth']))) >= 3.0
    ):
        ground_mask = np.zeros(len(src), dtype=bool)
        for _model, inliers, _elapsed in found:
            ground_mask |= inliers
        points = _flat_points(src, ground_mask)
        labels = np.array(['ground'] * len(points))
        ground = labels == 'ground'
        facade = labels == 'facade'
        rotation = np.eye(3)
        translation = np.zeros(3)
    return {
        'scene': scene,
        'view': view,
        'truth': truth,
        'src': src,
        'dst': dst,
        'plane': plane,
        'assigned': assigned,
        'models': {name: model for name, (model, _error) in assigned.items()},
        'fundamental_error': mean_epipolar_error(fundamental, gcp_src, gcp_dst),
        'points': points,
        'labels': labels,
        'ground_error': (
            float(np.median(np.abs(points[ground, 2] - truth['ground_depth'])))
            if ground.any() else float('inf')
        ),
        'facade_error': (
            float(np.median(np.abs(points[facade, 0] - truth['facade_distance'])))
            if facade.any() else float('inf')
        ),
        'rotation_error': _rotation_error(rotation, truth['rotation']),
        'translation_error': min(
            _angle_between(translation, truth['translation']),
            _angle_between(translation, -truth['translation']),
        ),
        'posed': bool(with_facade and cloud is not None),
    }


def _prefer_pose(current, candidate, with_facade):
    """两张角度都估一次，留下更能解释控制点的那次。"""
    if candidate is None:
        return current, False
    if current is None:
        return candidate, True
    current_ok = registration_ok(current, with_facade)
    candidate_ok = registration_ok(candidate, with_facade)
    if candidate_ok and not current_ok:
        return candidate, True
    if current_ok and not candidate_ok:
        return current, False
    if candidate['fundamental_error'] < current['fundamental_error']:
        return candidate, True
    return current, False


def survey_cloud(seed=11, with_facade=True, noise=0.4, vary_scene=False):
    """按格子生成航拍并三角化。立面格再用一张不同角度的照片核对平移。"""
    scene = make_scene(seed=seed if vary_scene else 3)
    view, truth = make_pair(scene, with_facade=with_facade)
    match_seed = seed if vary_scene else 7
    primary = _survey_once(scene, truth, seed, with_facade, noise, match_seed)
    used_second = False
    if with_facade:
        _second_view, second = make_pair(
            scene, with_facade=True,
            rotation_deg=SECOND_ROTATION, translation=SECOND_TRANSLATION,
        )
        alternate = _survey_once(
            scene, truth, seed + 5, with_facade, noise, match_seed,
            extra_ground_h=second['ground_h'], extra_facade_h=second['facade_h'],
        )
        primary, used_second = _prefer_pose(primary, alternate, with_facade)
    if primary is None:
        return None
    primary['view'] = view
    primary['second_view'] = used_second
    return primary


def registration_ok(result, with_facade):
    """地面控制点误差要过关；有立面时两个平面都要分开，深度偏差也不能太大。"""
    if result is None or 'ground' not in result['assigned'] or len(result['points']) < 8:
        return False
    if result['assigned']['ground'][1] >= 3.0:
        return False
    if not with_facade:
        return True
    if 'facade' not in result['assigned']:
        return False
    return (
        result['assigned']['facade'][1] < 4.0
        and result['ground_error'] < 3.0
        and result['facade_error'] < 4.0
    )


def main():
    result = survey_cloud()
    if result is None:
        print('三维点没有恢复出来')
        return 1
    plane = result['plane']
    true_count = int(np.sum(plane != 'outlier'))
    print(
        f'匹配 {len(plane)} 对：地面 {np.sum(plane == "ground")}，'
        f'立面 {np.sum(plane == "facade")}，外点 {np.sum(plane == "outlier")}'
    )
    print(f'真正落在某个平面上的占 {100.0 * true_count / len(plane):.1f}%')
    print('多平面单应矩阵（逆序残差）：')
    for name, label in (('ground', '地面'), ('facade', '立面')):
        if name not in result['assigned']:
            print(f'  {label}：没有分开')
            continue
        _model, error = result['assigned'][name]
        print(f'  {label}控制点平均误差 {error:.2f} px')
    print(f'基础矩阵（平面内点）：控制点平均极线距离 {result["fundamental_error"]:.2f} px')
    views = '第二张角度入选' if result.get('second_view') else '第一张角度'
    print(
        f'三维：三角化 {len(result["points"])} 个点，'
        f'旋转误差 {result["rotation_error"]:.2f}°，'
        f'平移方向误差 {result["translation_error"]:.2f}°（{views}）'
    )
    print(f'  地面点到深度 50 的中位偏差 {result["ground_error"]:.2f}')
    print(f'  立面点到距离 10 的中位偏差 {result["facade_error"]:.2f}')
    output = Path(__file__).resolve().parents[1] / 'aerial_registration.png'
    save_figure(
        output, result['scene'], result['view'], result['truth'],
        result['models'], result['src'], result['dst'], plane,
    )
    cloud_path = Path(__file__).resolve().parents[1] / 'aerial_points.png'
    save_cloud(cloud_path, result['points'], result['labels'])
    print(f'配准图已写到 {output}')
    print(f'点云图已写到 {cloud_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
