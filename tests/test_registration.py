import numpy as np
from sherd_merge.types import Fragment
from sherd_merge.registration import (
    extract_boundary, initial_pose_from_mirror, register_pair,
)
from tests.conftest import make_shell


def _ry180(pts):
    """y 軸まわり 180 度回転：(x,y,z)→(-x,y,-z)。"""
    out = pts.copy()
    out[:, 0] *= -1
    out[:, 2] *= -1
    return out


def _make_front_back_scans(radius=18.0, thickness=6.0, n=700, seed=7,
                           curvature=0.004):
    """物理的に正しい表/裏スキャンを作る。

    表スキャン   ＝ 外面 ＋ 縁壁（破断面）
    裏スキャン   ＝ (内面 ＋ 縁壁) を取得時の左右反転 Ry180 した生データ
    両者は縁壁を共有するので、Ry180 で戻すと縁壁が重なる。
    """
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    x = rad * np.cos(ang)
    y = rad * np.sin(ang)
    curve = curvature * (x ** 2 + y ** 2)
    outer = np.column_stack([x, y, curve + thickness / 2.0])
    inner = np.column_stack([x, y, curve - thickness / 2.0])

    # 縁壁 W：半径 R の輪に沿って厚み方向に点を張る（表裏が共有する重複領域）
    m = n // 2
    wang = rng.uniform(0, 2 * np.pi, m)
    wx = radius * np.cos(wang)
    wy = radius * np.sin(wang)
    wcurve = curvature * (wx ** 2 + wy ** 2)
    wz = wcurve + rng.uniform(-thickness / 2.0, thickness / 2.0, m)
    wall = np.column_stack([wx, wy, wz])

    front = np.vstack([outer, wall])        # 外面＋縁壁
    back_phys = np.vstack([inner, wall])    # 内面＋縁壁（原フレーム）
    back_scan = _ry180(back_phys)           # 取得時は左右反転された生スキャン
    return front, back_scan


def test_extract_boundary_returns_outer_ring():
    front, _ = make_shell(radius=20.0, thickness=6.0, n=800, seed=1)
    bnd = extract_boundary(front, quantile=0.15)
    r_all = np.linalg.norm(front[:, :2], axis=1)
    r_bnd = np.linalg.norm(bnd[:, :2], axis=1)
    assert r_bnd.mean() > r_all.mean()


def test_initial_pose_from_mirror_is_180_about_y():
    pts = np.array([[1.0, 2.0, 3.0]])
    out = initial_pose_from_mirror(pts)
    np.testing.assert_allclose(out, [[-1.0, 2.0, -3.0]])


def test_register_pair_recovers_known_overlap():
    front, back_scan = _make_front_back_scans(seed=7)
    # 取得後の小さな位置ずれ（ICP が補正すべき擾乱）
    theta = np.deg2rad(6.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    perturbed = back_scan @ Rz.T + np.array([1.5, -1.0, 0.5])
    f = Fragment("f", "front", front)
    b = Fragment("b", "back", perturbed)
    merged, info = register_pair(f, b, max_corr_dist=5.0, max_iter=80,
                                 boundary_quantile=0.2)
    # 実際に対応が取れていること（空振りでないことの検証）
    assert info["fitness"] > 0.3
    assert info["rmse"] < 1.2
    assert len(merged) >= len(front)
