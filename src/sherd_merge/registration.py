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


def register_pair(front: Fragment, back: Fragment,
                  max_corr_dist: float = 2.0, max_iter: int = 50,
                  boundary_quantile: float = 0.1) -> tuple[np.ndarray, dict]:
    """表裏 1 組を統合する。境界（縁）点で ICP を行い、変換後にマージする。

    返り値：(統合点群 (N,3), {'rmse':..., 'fitness':..., 'transform':4x4})

    粗合わせは「裏破片を自身の重心まわりで反転し、表破片の重心へ移す」。
    トレイ上での絶対位置に依存せず（各破片は (px,py) と様々な位置を取る）、
    反転後に表裏の重心が共在するため、続く境界 ICP が狭重複でも収束する。
    """
    c_front = front.points.mean(axis=0)
    c_back = back.points.mean(axis=0)
    # 裏破片を自身の重心へ寄せて反転し、表破片の重心へ整列（粗合わせ）。
    back_init = initial_pose_from_mirror(back.points - c_back) + c_front

    fb = extract_boundary(front.points, boundary_quantile)
    bb = extract_boundary(back_init, boundary_quantile)

    src = _to_pcd(bb)
    dst = _to_pcd(fb)
    result = o3d.pipelines.registration.registration_icp(
        src, dst, max_corr_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )

    back_full = _to_pcd(back_init)
    back_full.transform(result.transformation)
    merged = np.vstack([front.points, np.asarray(back_full.points)])

    info = {
        "rmse": float(result.inlier_rmse),
        "fitness": float(result.fitness),
        "transform": result.transformation,
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
