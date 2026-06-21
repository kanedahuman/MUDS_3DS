import numpy as np
import open3d as o3d


def normalize_unit(points: np.ndarray, target_unit_mm: float) -> np.ndarray:
    """入力座標 1 単位 = target_unit_mm ミリとして mm にスケールする。"""
    return points * float(target_unit_mm)


def load_points(path: str, target_unit_mm: float = 1.0) -> np.ndarray:
    """点群またはメッシュを読み込み、(N,3) の mm 単位点群を返す。"""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        mesh = o3d.io.read_triangle_mesh(path)
        pts = np.asarray(mesh.vertices)
    if pts.size == 0:
        raise ValueError(f"点群/メッシュを読み込めない: {path}")
    return normalize_unit(pts, target_unit_mm)
