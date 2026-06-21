import numpy as np
from sherd_merge.segment import segment_fragments
from sherd_merge.types import Fragment
from tests.conftest import make_tray


def test_segment_returns_one_fragment_per_blob():
    pts, labels = make_tray(n_frags=3, seed=0)
    frags = segment_fragments(pts, side="front", eps=5.0, min_points=50)
    assert len(frags) == 3
    assert all(isinstance(f, Fragment) for f in frags)
    assert all(f.side == "front" for f in frags)
    assert all(len(f.points) > 50 for f in frags)


def test_segment_assigns_unique_ids():
    pts, _ = make_tray(n_frags=3, seed=1)
    frags = segment_fragments(pts, side="back", eps=5.0, min_points=50)
    ids = [f.frag_id for f in frags]
    assert len(ids) == len(set(ids))
