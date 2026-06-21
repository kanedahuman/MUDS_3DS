import numpy as np
from scipy.spatial import ConvexHull
from .types import Fragment, Descriptor


def pca_axes(points: np.ndarray) -> np.ndarray:
    """点群の主軸（行ベクトル、固有値降順）を直交正規で返す。"""
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axes = vecs[:, order].T
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes


def _project_to_main_plane(points: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """主軸座標へ変換した (N,3) を返す（[:, :2] が主平面）。"""
    centered = points - points.mean(axis=0)
    return centered @ axes.T


def compute_descriptor(frag: Fragment, thickness_bins: int = 16) -> Descriptor:
    axes = pca_axes(frag.points)
    proj = _project_to_main_plane(frag.points, axes)
    xy = proj[:, :2]
    thickness_vals = proj[:, 2]

    try:
        hull = ConvexHull(xy)
        area = float(hull.volume)  # 2D の volume は面積
        contour = xy[hull.vertices]
    except Exception:
        area = 0.0
        contour = xy[:0]

    hist, _ = np.histogram(thickness_vals, bins=thickness_bins, density=False)
    total = hist.sum()
    thickness_hist = hist / total if total > 0 else hist.astype(float)

    extent = xy.max(axis=0) - xy.min(axis=0)
    max_dim = float(extent.max())
    aspect = float(extent.max() / extent.min()) if extent.min() > 1e-9 else 1.0

    return Descriptor(
        frag_id=frag.frag_id,
        contour_xy=contour,
        area=area,
        thickness_hist=thickness_hist,
        max_dim=max_dim,
        aspect=aspect,
        pca_axes=axes,
    )
