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


def contour_signature(contour_xy: np.ndarray, n_bins: int = 72) -> np.ndarray:
    """外周輪郭の極半径シグネチャ r(θ) を真スケールで返す。

    重心中心化 → PCA 主軸で回転正準化 → 角度を n_bins 等分し各角度の半径を
    補間。スケールは保持する（大小の破片を区別するため）。
    """
    pts = np.asarray(contour_xy, dtype=float)
    c = pts - pts.mean(axis=0)
    cov = np.cov(c.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]            # 最大分散方向
    ang0 = np.arctan2(axis[1], axis[0])
    rot = np.array([[np.cos(-ang0), -np.sin(-ang0)],
                    [np.sin(-ang0),  np.cos(-ang0)]])
    c = c @ rot.T
    theta = np.mod(np.arctan2(c[:, 1], c[:, 0]), 2*np.pi)
    radius = np.linalg.norm(c, axis=1)
    order = np.argsort(theta)
    theta_s, radius_s = theta[order], radius[order]
    theta_ext = np.concatenate([theta_s - 2*np.pi, theta_s, theta_s + 2*np.pi])
    radius_ext = np.concatenate([radius_s, radius_s, radius_s])
    bins = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
    return np.interp(bins, theta_ext, radius_ext)
