"""Seam 4 of spec 0001: validate_state reads the cluster state once and rejects bad input.

Every rejection is a ValueError whose message starts with the offending field's path, so the
person fixing the input knows exactly where to look.
"""

import re

import pytest

from weave.profiler.simulated_fixture import make_state
from weave.scheduler.candidates import validate_state


def test_valid_state_is_returned_unchanged():
    state = make_state()
    assert validate_state(state) is state


def test_non_object_state_is_rejected():
    with pytest.raises(ValueError, match=r"^state: expected a JSON object"):
        validate_state([])


TOP_LEVEL = [
    ("schema_version", lambda s: s.update(schema_version="1.0.0")),
    ("schema_version", lambda s: s.pop("schema_version")),
    ("snapshot_id", lambda s: s.update(snapshot_id="")),
    ("snapshot_id", lambda s: s.pop("snapshot_id")),
    ("simulation", lambda s: s.update(simulation="yes")),
    ("model", lambda s: s.pop("model")),
    ("model", lambda s: s.update(model=[])),
    ("request_envelope", lambda s: s.update(request_envelope=[])),
    ("devices", lambda s: s.update(devices={})),
    ("devices", lambda s: s.update(devices=[])),
]


@pytest.mark.parametrize("path, mutate", TOP_LEVEL)
def test_bad_top_level_field_is_named(path, mutate):
    state = make_state()
    mutate(state)
    with pytest.raises(ValueError, match=rf"^{re.escape(path)}: "):
        validate_state(state)
