import numpy as np
from sherd_merge.descriptors import compute_descriptor, pca_axes
from sherd_merge.types import Fragment
from tests.conftest import make_shell


def test_pca_axes_orthonormal():
    pts = np.random.rand(200, 3) * 10
    axes = pca_axes(pts)
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-6)


def test_compute_descriptor_area_and_thickness():
    front, _ = make_shell(radius=20.0, thickness=6.0, n=600, seed=3)
    frag = Fragment(frag_id="f", side="front", points=front)
    desc = compute_descriptor(frag, thickness_bins=16)
    assert 800 < desc.area < 2000
    assert desc.thickness_hist.shape == (16,)
    np.testing.assert_allclose(desc.thickness_hist.sum(), 1.0, atol=1e-6)
    assert desc.contour_xy.shape[1] == 2
    assert desc.max_dim > 0


def test_contour_signature_scale_and_shape():
    from sherd_merge.descriptors import contour_signature
    import numpy as np
    ang = np.linspace(0, 2*np.pi, 60, endpoint=False)
    circle10 = np.column_stack([10*np.cos(ang), 10*np.sin(ang)])
    sig = contour_signature(circle10, n_bins=72)
    assert sig.shape == (72,)
    assert abs(sig.mean() - 10.0) < 0.5
    sig_shift = contour_signature(circle10 + np.array([5.0, -3.0]), n_bins=72)
    assert np.abs(sig - sig_shift).mean() < 0.3


def test_contour_signature_distinguishes_size():
    from sherd_merge.descriptors import contour_signature
    import numpy as np
    ang = np.linspace(0, 2*np.pi, 60, endpoint=False)
    c10 = np.column_stack([10*np.cos(ang), 10*np.sin(ang)])
    c20 = np.column_stack([20*np.cos(ang), 20*np.sin(ang)])
    s10 = contour_signature(c10); s20 = contour_signature(c20)
    assert abs(s20.mean() - 2*s10.mean()) < 1.0
