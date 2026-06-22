import os
import csv
import numpy as np
import open3d as o3d


def write_model(points: np.ndarray, out_dir: str, name: str) -> str:
    """個別破片を PLY で書き出し、パスを返す。"""
    frag_dir = os.path.join(out_dir, "fragments")
    os.makedirs(frag_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    path = os.path.join(frag_dir, f"{name}.ply")
    o3d.io.write_point_cloud(path, pcd)
    return path


def write_mesh(mesh, out_dir: str, name: str) -> str:
    """計測用メッシュを OBJ で書き出す。"""
    frag_dir = os.path.join(out_dir, "fragments")
    os.makedirs(frag_dir, exist_ok=True)
    path = os.path.join(frag_dir, f"{name}.obj")
    o3d.io.write_triangle_mesh(path, mesh)
    return path


REPORT_FIELDS = ["frag_id", "back_id", "area", "max_dim",
                 "match_cost", "icp_rmse", "shell_sep", "needs_review"]


def write_report(rows: list[dict], out_dir: str) -> str:
    """QC レポート CSV を書き出し、パスを返す。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in REPORT_FIELDS})
    return path
