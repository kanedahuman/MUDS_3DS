import os
import click
from .config import load_config, Config
from .io_adapter import load_points
from .preprocess import remove_tray_plane, denoise, voxel_downsample
from .segment import segment_fragments
from .matching import match_front_back
from .descriptors import compute_descriptor
from .registration import register_pair, poisson_mesh
from .output import write_model, write_mesh, write_report


def run_pipeline(cfg: Config) -> str:
    """設定に従い表裏トレイを統合し、個別モデルと QC レポートを出力する。

    返り値：report.csv のパス。
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # M0 入力
    front_pts = load_points(cfg.input_front, cfg.target_unit_mm)
    back_pts = load_points(cfg.input_back, cfg.target_unit_mm)

    # M1 トレイ除去＋デノイズ＋ダウンサンプル
    def _prep(pts):
        pts = remove_tray_plane(pts, cfg.plane_dist_threshold)
        pts = denoise(pts, cfg.denoise_neighbors, cfg.denoise_std_ratio)
        pts = voxel_downsample(pts, cfg.voxel_size)
        return pts

    front_pts = _prep(front_pts)
    back_pts = _prep(back_pts)

    # M2 個別切り出し
    fronts = segment_fragments(front_pts, "front", cfg.cluster_eps, cfg.cluster_min_points)
    backs = segment_fragments(back_pts, "back", cfg.cluster_eps, cfg.cluster_min_points)

    # M4 表裏照合（M3 記述子は内部で計算）
    pairs = match_front_back(fronts, backs, cfg.position_prune_radius,
                             cfg.match_cost_threshold)

    front_by_id = {f.frag_id: f for f in fronts}
    back_by_id = {b.frag_id: b for b in backs}

    rows = []
    for idx, pair in enumerate(pairs, start=1):
        f = front_by_id[pair.front_id]
        b = back_by_id[pair.back_id]
        # M5 統合（縁ベース）
        merged, info = register_pair(f, b, cfg.icp_max_corr_dist,
                                     cfg.icp_max_iter,
                                     edge_k=cfg.edge_k,
                                     edge_gap_deg=cfg.edge_gap_deg,
                                     shell_min_sep=cfg.shell_min_sep)
        name = f"fragment_{idx:03d}"
        write_model(merged, cfg.output_dir, name)
        if cfg.mesh_for_measurement:
            write_mesh(poisson_mesh(merged), cfg.output_dir, name)

        desc = compute_descriptor(f)
        rows.append({
            "frag_id": name,
            "back_id": pair.back_id,
            "area": round(desc.area, 2),
            "max_dim": round(desc.max_dim, 2),
            "match_cost": round(pair.cost, 4),
            "icp_rmse": round(info["rmse"], 4),
            "shell_sep": round(info["shell_sep"], 4),
            "needs_review": pair.needs_review or info["collapsed"],
        })

    return write_report(rows, cfg.output_dir)


@click.command()
@click.option("--config", "config_path", required=True, help="YAML 設定ファイル")
def main(config_path: str):
    cfg = load_config(config_path)
    report = run_pipeline(cfg)
    click.echo(f"完了：{report}")


if __name__ == "__main__":
    main()
