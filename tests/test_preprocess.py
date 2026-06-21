import numpy as np
from sherd_merge.preprocess import remove_tray_plane
from tests.conftest import make_tray_with_plane


def test_remove_tray_plane_drops_plane_points():
    pts, labels = make_tray_with_plane(n_frags=3, seed=0)
    kept = remove_tray_plane(pts, plane_dist_threshold=1.0)
    # 残った点は大半が破片（z が高い）で、平面点（z≈0）はほぼ消える
    assert (kept[:, 2] < 1.0).mean() < 0.1
    # 破片点の総数程度は残る
    assert len(kept) > 0.6 * (labels >= 0).sum()
