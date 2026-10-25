"""Seam 2 of spec 0001: assignments lays layers out per stage and maps stages to MLX ranks.

Expected values come from outside the planner: plain range() for layer coverage, the guide's
section 7.1 table for [10, 18], and the start-index formula of MLX's installed PipelineMixin.
"""

import itertools
import re

import pytest

from weave.scheduler.candidates import assignments


def splits(total_layers: int, stages: int):
    """Every way to give each of `stages` stages at least one of `total_layers` layers."""
    for cuts in itertools.combinations(range(1, total_layers), stages - 1):
        bounds = (0, *cuts, total_layers)
        yield [b - a for a, b in itertools.pairwise(bounds)]


def all_splits(max_stages: int = 3):
    for total in range(1, 29):
        for stages in range(1, max_stages + 1):
            for counts in splits(total, stages):
                yield total, counts


def devices(n: int) -> list[str]:
    return [f"mac-{i}" for i in range(n)]


def test_every_split_covers_each_layer_exactly_once_in_order():
    for total, counts in all_splits():
        rows = assignments(devices(len(counts)), counts)
        covered = [
            layer for r in rows for layer in range(r["start_layer"], r["end_layer_exclusive"])
        ]
        assert covered == list(range(total)), counts


def test_stage_layer_counts_are_the_requested_counts():
    for _, counts in all_splits():
        rows = assignments(devices(len(counts)), counts)
        assert [r["layer_count"] for r in rows] == counts


def test_ranks_match_mlx_pipeline_start_index_formula():
    # MLX PipelineMixin: start_idx = sum(split[rank + 1:]), where split is indexed by rank.
    for _, counts in all_splits():
        rows = assignments(devices(len(counts)), counts)
        native_split = [r["layer_count"] for r in sorted(rows, key=lambda r: r["mlx_rank"])]
        for r in rows:
            assert r["start_layer"] == sum(native_split[r["mlx_rank"] + 1 :]), counts


def test_logical_10_18_matches_the_guide_table():
    assert assignments(["A", "B"], [10, 18]) == [
        {
            "device_id": "A",
            "logical_stage": 0,
            "mlx_rank": 1,
            "start_layer": 0,
            "end_layer_exclusive": 10,
            "layer_count": 10,
        },
        {
            "device_id": "B",
            "logical_stage": 1,
            "mlx_rank": 0,
            "start_layer": 10,
            "end_layer_exclusive": 28,
            "layer_count": 18,
        },
    ]


def test_single_device_is_one_stage_at_rank_zero():
    assert assignments(["A"], [28]) == [
        {
            "device_id": "A",
            "logical_stage": 0,
            "mlx_rank": 0,
            "start_layer": 0,
            "end_layer_exclusive": 28,
            "layer_count": 28,
        }
    ]


BAD_ARGUMENTS = [
    ("logical_counts", [], []),
    ("device_order", ["A"], [10, 18]),
    ("logical_counts[1]", ["A", "B"], [10, 0]),
    ("logical_counts[0]", ["A"], [-1]),
    ("logical_counts[0]", ["A", "B"], [True, 27]),
    ("device_order[0]", ["", "B"], [10, 18]),
    ("device_order[1]", ["A", "A"], [10, 18]),
]


@pytest.mark.parametrize("path, order, counts", BAD_ARGUMENTS)
def test_bad_argument_is_named(path, order, counts):
    with pytest.raises(ValueError, match=rf"^{re.escape(path)}: "):
        assignments(order, counts)
