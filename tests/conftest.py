import numpy as np


def make_shell(radius=20.0, thickness=6.0, n=400, seed=0, curvature=0.004):
    """緩く湾曲した殻状破片の表/裏面点群を返す。z は曲面、厚み方向に表裏を分離。"""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = radius * np.sqrt(rng.uniform(0, 1, n))
    x = rad * np.cos(ang)
    y = rad * np.sin(ang)
    z_curve = curvature * (x ** 2 + y ** 2)  # お椀状の湾曲
    front = np.column_stack([x, y, z_curve + thickness / 2.0])
    back = np.column_stack([x, y, z_curve - thickness / 2.0])
    return front, back


def make_tray(n_frags=3, seed=0, plane_points=2000):
    """トレイ平面 + 複数の破片（表面のみ）を結合した点群と破片ラベルを返す。
    ラベル -1 はトレイ平面。"""
    rng = np.random.default_rng(seed)
    all_pts = []
    all_labels = []
    grid = [(-50, -50), (50, -50), (0, 50), (-50, 50), (50, 50)]
    for i in range(n_frags):
        gx, gy = grid[i]
        front, _ = make_shell(radius=15.0, thickness=6.0, n=300, seed=seed + i)
        front[:, 0] += gx
        front[:, 1] += gy
        all_pts.append(front)
        all_labels.append(np.full(len(front), i))
    pts = np.vstack(all_pts)
    labels = np.concatenate(all_labels)
    return pts, labels


def make_tray_with_plane(n_frags=3, seed=0, plane_points=2000):
    """トレイ平面付き。平面は z≈0、破片はその上に乗る。"""
    pts, labels = make_tray(n_frags=n_frags, seed=seed)
    pts[:, 2] += 6.0  # 破片を平面の上へ持ち上げる
    rng = np.random.default_rng(seed + 99)
    px = rng.uniform(-80, 80, plane_points)
    py = rng.uniform(-80, 80, plane_points)
    pz = rng.normal(0, 0.2, plane_points)
    plane = np.column_stack([px, py, pz])
    pts = np.vstack([pts, plane])
    labels = np.concatenate([labels, np.full(plane_points, -1)])
    return pts, labels
