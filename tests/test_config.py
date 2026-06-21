from sherd_merge.config import load_config, Config


def test_load_config_applies_defaults(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("input:\n  front: a.ply\n  back: b.ply\n", encoding="utf-8")
    cfg = load_config(str(cfg_file))
    assert cfg.input_front == "a.ply"
    assert cfg.input_back == "b.ply"
    assert cfg.cluster_eps > 0
    assert cfg.match_cost_threshold > 0
    assert cfg.position_prune_radius > 0


def test_load_config_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "input:\n  front: a.ply\n  back: b.ply\n"
        "segment:\n  cluster_eps: 5.0\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    assert cfg.cluster_eps == 5.0
