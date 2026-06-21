import numpy as np
import open3d as o3d


def _to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def remove_tray_plane(points: np.ndarray, plane_dist_threshold: float = 1.0,
                      ransac_n: int = 3, num_iterations: int = 1000) -> np.ndarray:
    """RANSAC で最大平面（トレイ面）を検出し、その点を除去して返す。"""
    pcd = _to_pcd(points)
    _, inliers = pcd.segment_plane(
        distance_threshold=plane_dist_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    rest = pcd.select_by_index(inliers, invert=True)
    return np.asarray(rest.points)


def denoise(points: np.ndarray, neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """統計的外れ値除去。"""
    pcd = _to_pcd(points)
    clean, _ = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
    return np.asarray(clean.points)


def estimate_normals(points: np.ndarray, radius: float = 3.0, max_nn: int = 30) -> np.ndarray:
    """法線を推定して (N,3) で返す。"""
    pcd = _to_pcd(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return np.asarray(pcd.normals)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """ボクセルダウンサンプル。voxel<=0 のときは恒等（入力をそのまま返す）。"""
    if voxel is None or voxel <= 0:
        return points
    pcd = _to_pcd(points)
    ds = pcd.voxel_down_sample(voxel)
    return np.asarray(ds.points)
