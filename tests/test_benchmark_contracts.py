"""Ablations and source-transition samples must retain their declared cache state."""
import sys

import pytest

from profiling.engine_sweep import arguments, source_bound_context_key


def test_source_bound_ablation_does_not_admit_mutable_library_pools():
    assert source_bound_context_key(None, None, None, None) is None
    assert source_bound_context_key("snapshot", b"rows", None, None) == ("snapshot", b"rows")


@pytest.mark.parametrize("phase,reps,valid", [("combined", 1, False), ("search", 3, False), ("pool", 1, False), ("search", 1, True)])
def test_transition_sampling_requires_persisted_pool_and_one_advance(tmp_path, monkeypatch, phase, reps, valid):
    source = tmp_path / "source.parquet"
    source.touch()
    argv = ["sweep", "--case", "discover", "--parquet", str(source), "--output", str(tmp_path/"out.json"),
            "--pool-artifact", str(tmp_path/"pool"), "--discover-phase", phase,
            "--transition-searches", "--reps", str(reps)]
    monkeypatch.setattr(sys, "argv", argv)
    if valid:
        assert arguments().reps == 1
    else:
        with pytest.raises(SystemExit):
            arguments()
