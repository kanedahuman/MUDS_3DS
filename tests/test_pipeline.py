import os
import numpy as np
import open3d as o3d
from sherd_merge.config import Config
from sherd_merge.pipeline import run_pipeline
from tests.conftest import make_shell


def _write_tray(path, fronts):
    pts = np.vstack(fronts)
    rng = np.random.default_rng(0)
    plane = np.column_stack([rng.uniform(-90, 90, 1500),
                             rng.uniform(-90, 90, 1500),
                             rng.normal(0, 0.2, 1500)])
    allp = np.vstack([pts, plane])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(allp)
    o3d.io.write_point_cloud(path, pcd)


def test_run_pipeline_end_to_end(tmp_path):
    positions = [(-40, 0), (40, 0)]
    radii = [14.0, 24.0]
    fronts, backs = [], []
    for i, ((px, py), r) in enumerate(zip(positions, radii)):
        f, b = make_shell(radius=r, thickness=6.0, n=500, seed=30 + i)
        f[:, 2] += 6.0; f[:, 0] += px; f[:, 1] += py
        bb = b.copy(); bb[:, 0] *= -1  # 左右反転して同位置へ
        bb[:, 2] += 6.0; bb[:, 0] += px; bb[:, 1] += py
        fronts.append(f); backs.append(bb)

    front_path = str(tmp_path / "front.ply")
    back_path = str(tmp_path / "back.ply")
    _write_tray(front_path, fronts)
    _write_tray(back_path, backs)

    cfg = Config(input_front=front_path, input_back=back_path,
                 output_dir=str(tmp_path / "out"), work_dir=str(tmp_path / "work"),
                 cluster_eps=5.0, cluster_min_points=50,
                 position_prune_radius=30.0, mesh_for_measurement=False)
    report_path = run_pipeline(cfg)

    assert os.path.exists(report_path)
    frag_dir = os.path.join(str(tmp_path / "out"), "fragments")
    plys = [f for f in os.listdir(frag_dir) if f.endswith(".ply")]
    assert len(plys) == 2
