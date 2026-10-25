"""Seam 3 of spec 0001: estimate_stage_memory applies guide section 10, term by term.

Expected values are worked by hand from the section 10 formula, never recomputed the way the code
computes them.
"""

import re

import pytest

from weave.estimator.memory import estimate_stage_memory

MIB = 2**20


def pressure_mac_10_layers(**changes):
    """The [10, 18] first stage on the committed fixture's 3 GiB Mac."""
    terms = dict(
        layer_weight_bytes=[128 * MIB] * 10,
        replicated_weight_bytes=1024 * MIB,
        total_tokens=512 + 64,
        kv_heads=4,
        head_dim=128,
        kv_bytes_per_element=4,
        concurrency=1,
        workspace_reserve_bytes=512 * MIB,
        safety_margin_bytes=512 * MIB,
    )
    return {**terms, **changes}


def test_fixture_stage_matches_the_worked_example():
    # 10 x 134,217,728 = 1,342,177,280 layer bytes; KV = 2*10*576*4*128*4 = 23,592,960.
    assert list(estimate_stage_memory(**pressure_mac_10_layers()).items()) == [
        ("layer_weight_bytes", 1_342_177_280),
        ("replicated_weight_bytes", 1_073_741_824),
        ("kv_upper_bound_bytes", 23_592_960),
        ("workspace_reserve_bytes", 536_870_912),
        ("safety_margin_bytes", 536_870_912),
    ]


def test_smallest_kv_bound_is_one_key_plus_one_value():
    terms = dict(
        layer_weight_bytes=[1],
        replicated_weight_bytes=1,
        total_tokens=1,
        kv_heads=1,
        head_dim=1,
        kv_bytes_per_element=1,
        concurrency=1,
        workspace_reserve_bytes=1,
        safety_margin_bytes=1,
    )
    assert estimate_stage_memory(**terms)["kv_upper_bound_bytes"] == 2


def test_kv_bound_grows_with_concurrency():
    # Three concurrent requests hold three caches: 3 x 23,592,960.
    result = estimate_stage_memory(**pressure_mac_10_layers(concurrency=3))
    assert result["kv_upper_bound_bytes"] == 70_778_880


def test_kv_bound_grows_with_context():
    # A doubled token envelope (1,152 tokens) doubles the cache: 2*10*1152*4*128*4.
    result = estimate_stage_memory(**pressure_mac_10_layers(total_tokens=1152))
    assert result["kv_upper_bound_bytes"] == 47_185_920


def test_unequal_layers_are_summed_and_each_holds_a_cache():
    # 5 + 7 + 11 = 23 bytes of weights; KV = 2 * 3 layers * 576 * 4 * 128 * 4.
    result = estimate_stage_memory(**pressure_mac_10_layers(layer_weight_bytes=[5, 7, 11]))
    assert (result["layer_weight_bytes"], result["kv_upper_bound_bytes"]) == (23, 7_077_888)


def test_no_term_has_a_default():
    # Guide section 10: workspace and safety reserves "may not silently default to zero".
    terms = pressure_mac_10_layers()
    del terms["workspace_reserve_bytes"]
    with pytest.raises(TypeError):
        estimate_stage_memory(**terms)


BAD_TERMS = [
    ("layer_weight_bytes", {"layer_weight_bytes": []}),
    ("layer_weight_bytes[1]", {"layer_weight_bytes": [5, 0]}),
    ("layer_weight_bytes[0]", {"layer_weight_bytes": [True]}),
    ("replicated_weight_bytes", {"replicated_weight_bytes": 0}),
    ("total_tokens", {"total_tokens": -1}),
    ("kv_heads", {"kv_heads": True}),
    ("head_dim", {"head_dim": 1.5}),
    ("kv_bytes_per_element", {"kv_bytes_per_element": 0}),
    ("concurrency", {"concurrency": 0}),
    ("workspace_reserve_bytes", {"workspace_reserve_bytes": 0}),
    ("safety_margin_bytes", {"safety_margin_bytes": "1"}),
]


@pytest.mark.parametrize("name, change", BAD_TERMS)
def test_bad_term_is_named(name, change):
    with pytest.raises(ValueError, match=rf"^{re.escape(name)}: "):
        estimate_stage_memory(**pressure_mac_10_layers(**change))
