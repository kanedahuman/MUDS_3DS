import numpy as np
import open3d as o3d
from sherd_merge.io_adapter import load_points, normalize_unit


def test_normalize_unit_scales_points():
    pts = np.array([[1.0, 2.0, 3.0]])
    out = normalize_unit(pts, target_unit_mm=10.0)  # 1 単位 = 10mm
    np.testing.assert_allclose(out, [[10.0, 20.0, 30.0]])


def test_load_points_reads_ply(tmp_path):
    pts = np.random.rand(100, 3) * 50
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    p = tmp_path / "x.ply"
    o3d.io.write_point_cloud(str(p), pcd)
    loaded = load_points(str(p), target_unit_mm=1.0)
    assert loaded.shape == (100, 3)
