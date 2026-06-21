import numpy as np
import open3d as o3d
from .types import Fragment


def segment_fragments(points: np.ndarray, side: str, eps: float = 3.0,
                      min_points: int = 50) -> list[Fragment]:
    """DBSCAN（ユークリッド・クラスタリング）で破片に分離し Fragment 列を返す。

    min_points はクラスタの最小点数フィルタとして使用する。
    Open3D の cluster_dbscan には内部用の小さい値を渡し、
    得られたクラスタを min_points でサイズフィルタリングする。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Open3D の min_points はコア点判定の近傍数。
    # ここでは小さい固定値を渡してクラスタを形成し、
    # 後段でサイズ >= min_points のクラスタのみを残す。
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=5))
    frags = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue  # ノイズ
        mask = labels == lab
        if mask.sum() < min_points:
            continue
        frags.append(Fragment(frag_id=f"{side}_{lab:03d}", side=side,
                              points=points[mask]))
    return frags
