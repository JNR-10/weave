"""Installed-source contract check on the M4; not a distributed runtime test."""

import pytest

pytest.importorskip("mlx.core")
PipelineMixin = pytest.importorskip("mlx_lm.models.pipeline").PipelineMixin


@pytest.mark.parametrize("rank,start,end", [(0, 10, 28), (1, 0, 10)])
def test_installed_unequal_split(rank, start, end):
    class Group:
        def rank(self):
            return rank

        def size(self):
            return 2

    class Layers(PipelineMixin):
        def __init__(self):
            super().__init__()
            self.layers = list(range(28))

    model = Layers()
    model.pipeline(Group(), split=[18, 10])
    assert (model.start_idx, model.end_idx) == (start, end)
    assert model.pipeline_layers == list(range(start, end))
