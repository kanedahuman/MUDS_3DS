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


def make_edged_shell(radius=18.0, thickness=6.0, n=700, seed=0,
                     curvature=0.004, jag=2.0):
    """ギザギザの共有縁を持つ殻。返り値 (front, back_phys, edge)。

    front      = 外面（z=+t/2）＋ 縁
    back_phys  = 内面（z=-t/2）＋ 縁（原フレーム）
    edge       = 両者が共有する3D波状リング（半径 R 付近、法線方向に jag だけ揺らぐ）
    """
    rng = np.random.default_rng(seed)
    # 共有のギザギザ縁リング
    m = max(80, n // 4)
    ea = np.sort(rng.uniform(0, 2 * np.pi, m))
    er = radius + rng.normal(0, 0.3, m)
    ex = er * np.cos(ea)
    ey = er * np.sin(ea)
    ec = curvature * (ex ** 2 + ey ** 2)
    ez = ec + rng.uniform(-jag, jag, m)
    edge = np.column_stack([ex, ey, ez])
    # 外面・内面（円板内部）
    a = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    x = r * np.cos(a)
    y = r * np.sin(a)
    zc = curvature * (x ** 2 + y ** 2)
    outer = np.column_stack([x, y, zc + thickness / 2.0])
    inner = np.column_stack([x, y, zc - thickness / 2.0])
    front = np.vstack([outer, edge])
    back_phys = np.vstack([inner, edge])
    return front, back_phys, edge


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
