import csv
import numpy as np
import open3d as o3d
from sherd_merge.output import write_model, write_report


def test_write_model_creates_ply(tmp_path):
    pts = np.random.rand(200, 3) * 30
    path = write_model(pts, str(tmp_path), "fragment_001")
    assert path.endswith(".ply")
    loaded = o3d.io.read_point_cloud(path)
    assert len(loaded.points) == 200


def test_write_report_has_rows_and_flags(tmp_path):
    rows = [
        {"frag_id": "fragment_001", "back_id": "back_002", "area": 1200.0,
         "max_dim": 40.0, "match_cost": 0.1, "icp_rmse": 0.2, "needs_review": False},
        {"frag_id": "fragment_002", "back_id": "back_001", "area": 800.0,
         "max_dim": 30.0, "match_cost": 0.4, "icp_rmse": 0.9, "needs_review": True},
    ]
    path = write_report(rows, str(tmp_path))
    with open(path, encoding="utf-8") as fh:
        data = list(csv.DictReader(fh))
    assert len(data) == 2
    assert data[1]["needs_review"] == "True"
