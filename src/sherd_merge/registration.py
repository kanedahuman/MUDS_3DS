import numpy as np
import open3d as o3d
from .types import Fragment
from .descriptors import pca_axes


def extract_boundary(points: np.ndarray, quantile: float = 0.1) -> np.ndarray:
    """主平面投影で外周近傍（半径方向外側）の点を境界として抽出する。"""
    axes = pca_axes(points)
    centered = points - points.mean(axis=0)
    proj = centered @ axes.T
    r = np.linalg.norm(proj[:, :2], axis=1)
    thresh = np.quantile(r, 1.0 - quantile)
    return points[r >= thresh]


def initial_pose_from_mirror(back_points: np.ndarray) -> np.ndarray:
    """裏面を表へ向かい合わせる初期姿勢を適用して返す。

    取得時の「同じ位置で左右反転」は物理的には y 軸まわりの 180 度回転
    （Ry180：(x,y,z)→(-x,y,-z)）である。これは行列式 +1 の剛体変換であり、
    x のみ反転する鏡映（行列式 -1）ではない点に注意。これを裏スキャンに
    適用すると、内面が表の外面と向かい合い、縁壁（破断面）が重なる。
    """
    out = back_points.copy()
    out[:, 0] *= -1
    out[:, 2] *= -1
    return out


def _to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def extract_edge_loop(points: np.ndarray, k: int = 30,
                      gap_deg: float = 90.0) -> np.ndarray:
    """開いた縁（境界）点を返す境界検出。

    各点について k 近傍を局所接平面へ射影し、接平面内での近傍方位角を並べて
    最大角度ギャップを求める。内部点は近傍が全周に分布しギャップ小、縁点は
    片側に偏りギャップ大。最大ギャップ > gap_deg の点を縁とする。
    現行の半径帯ベース extract_boundary とは別物（細い縁ループを取る）。
    """
    n = len(points)
    if n < k + 1:
        return points
    pcd = _to_pcd(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    normals = np.asarray(pcd.normals)
    tree = o3d.geometry.KDTreeFlann(pcd)
    gap_rad = np.deg2rad(gap_deg)
    edge_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        _, idx, _ = tree.search_knn_vector_3d(points[i], k + 1)
        idx = [j for j in idx if j != i]
        if len(idx) < 3:
            continue
        nb = points[idx] - points[i]
        ni = normals[i]
        # 接平面基底
        ref = np.array([1.0, 0.0, 0.0])
        if abs(ni @ ref) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - (ref @ ni) * ni
        u /= np.linalg.norm(u) + 1e-12
        v = np.cross(ni, u)
        ang = np.arctan2(nb @ v, nb @ u)
        ang = np.sort(ang)
        gaps = np.diff(ang)
        wrap = 2 * np.pi - (ang[-1] - ang[0])
        max_gap = max(gaps.max(), wrap)
        if max_gap > gap_rad:
            edge_mask[i] = True
    return points[edge_mask]


def _rot180_about(axis: np.ndarray) -> np.ndarray:
    """単位軸 axis まわりの 180 度回転行列（R = 2 a aᵀ - I、det=+1）。"""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    return 2.0 * np.outer(a, a) - np.eye(3)


def shell_separation(front: np.ndarray, back: np.ndarray) -> dict:
    """整合後の2面が厚み方向に分離しているか定量化する。

    統合点群の PCA 最小分散方向（法線 n）に沿って front/back を投影し、
    平均差 sep=|mean_f - mean_b|、重なり overlap、全幅 span を返す。
    殻なら sep>0 で overlap 小、潰れなら sep≈0 で overlap≒span。
    """
    M = np.vstack([front, back])
    c = M.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((M - c).T))
    n = vecs[:, 0]  # 最小分散方向＝法線
    pf = (front - c) @ n
    pb = (back - c) @ n
    overlap = max(0.0, min(pf.max(), pb.max()) - max(pf.min(), pb.min()))
    allp = (M - c) @ n
    span = float(allp.max() - allp.min())
    return {"sep": float(abs(pf.mean() - pb.mean())),
            "overlap": float(overlap), "span": span}


def _edge_icp(src_edge: np.ndarray, dst_edge: np.ndarray,
              init: np.ndarray, max_corr_dist: float, max_iter: int):
    src = _to_pcd(src_edge)
    dst = _to_pcd(dst_edge)
    return o3d.pipelines.registration.registration_icp(
        src, dst, max_corr_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )


def coarse_align_edges(front_edge: np.ndarray, back_edge: np.ndarray,
                       max_corr_dist: float, max_iter: int,
                       n_angles: int = 12) -> tuple[np.ndarray, object]:
    """裏返し（面内軸まわり 180 度回転）の軸方位 φ を探索し、縁 ICP の残差が
    最小の整合を返す。返り値 (4x4 変換, 最良 ICP 結果)。"""
    cf = front_edge.mean(axis=0)
    cb = back_edge.mean(axis=0)
    axes = pca_axes(front_edge)  # 行: 主・副・法線（固有値降順）
    best_T, best_res, best_key = None, None, None
    for kk in range(n_angles):
        phi = np.pi * kk / n_angles
        ax = np.cos(phi) * axes[0] + np.sin(phi) * axes[1]
        R = _rot180_about(ax)
        # 初期変換: (p - cb) @ Rᵀ + cf
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = cf - R @ cb
        res = _edge_icp(back_edge, front_edge, T0, max_corr_dist, max_iter)
        key = (res.fitness, -res.inlier_rmse)
        if best_key is None or key > best_key:
            best_key = key
            best_res = res
            best_T = res.transformation
    return best_T, best_res


def register_pair(front: Fragment, back: Fragment,
                  max_corr_dist: float = 2.0, max_iter: int = 50,
                  edge_k: int = 30, edge_gap_deg: float = 90.0,
                  shell_min_sep: float = 0.8) -> tuple[np.ndarray, dict]:
    """表裏 1 組を統合する。共有する縁ループを合わせて殻を再構成する。

    手順：縁ループ抽出 → 反転軸探索つき縁 ICP（粗→精） → 全裏面へ変換適用 →
    マージ。縁は3D空間曲線なので面内姿勢に加え厚み方向も拘束し、面が潰れない。

    返り値：(統合点群 (N,3),
             {'rmse','fitness','transform','shell_sep','overlap','collapsed'})
    """
    fe = extract_edge_loop(front.points, k=edge_k, gap_deg=edge_gap_deg)
    be = extract_edge_loop(back.points, k=edge_k, gap_deg=edge_gap_deg)
    if len(fe) < 4 or len(be) < 4:
        # 縁が取れない場合は全点を縁の代わりに使う（退避）
        fe = front.points if len(fe) < 4 else fe
        be = back.points if len(be) < 4 else be

    T, res = coarse_align_edges(fe, be, max_corr_dist, max_iter)

    back_full = _to_pcd(back.points)
    back_full.transform(T)
    back_aligned = np.asarray(back_full.points)
    merged = np.vstack([front.points, back_aligned])

    sh = shell_separation(front.points, back_aligned)
    info = {
        "rmse": float(res.inlier_rmse),
        "fitness": float(res.fitness),
        "transform": T,
        "shell_sep": sh["sep"],
        "overlap": sh["overlap"],
        "collapsed": bool(sh["sep"] < shell_min_sep),
    }
    return merged, info


def poisson_mesh(points: np.ndarray, depth: int = 9):
    """計測・図面用の水密メッシュを返す（法線を推定してポアソン再構成）。"""
    pcd = _to_pcd(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30)
    )
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh
