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


def test_voxel_downsample_reduces_points():
    from sherd_merge.preprocess import voxel_downsample
    import numpy as np
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, (5000, 3))
    out = voxel_downsample(pts, 1.0)
    assert out.shape[1] == 3
    assert len(out) < len(pts)


def test_voxel_downsample_noop_when_zero():
    from sherd_merge.preprocess import voxel_downsample
    import numpy as np
    pts = np.random.rand(100, 3)
    out = voxel_downsample(pts, 0.0)
    assert len(out) == 100
