import numpy as np
from sherd_merge.types import Fragment, Descriptor, MatchPair


def test_fragment_centroid_xy_computed_from_points():
    pts = np.array([[0.0, 0.0, 1.0], [2.0, 4.0, 1.0]])
    frag = Fragment(frag_id="f1", side="front", points=pts)
    np.testing.assert_allclose(frag.centroid_xy, [1.0, 2.0])


def test_match_pair_holds_ids_and_score():
    pair = MatchPair(front_id="f1", back_id="b3", cost=0.12, needs_review=False)
    assert pair.front_id == "f1"
    assert pair.back_id == "b3"
    assert pair.cost == 0.12
    assert pair.needs_review is False
