import pytest
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from mill import get_data, serve
from mill.get_data import GetData
from mill.memo import ByteLRU
from conftest import ns


def test_offline_server_observes_newer_disk_cache_from_another_process(arrow_store, monkeypatch):
    times = [ns(2024, 1, 1), ns(2024, 1, 2)]
    arrow_store.write("time_1h", times[:1])
    monkeypatch.setattr(serve, "_RAW_CACHE", {})
    old, version = serve._raw_for("1h_time_bars")
    assert old.height == 1
    parent_cache = get_data._DATA
    arrow_store.write("time_1h", times)
    # Separate process memory, shared persistent cache.
    with monkeypatch.context() as child:
        child.setattr(get_data, "_DATA", {})
        assert GetData().get_time_bars("1h_time_bars").height == 2
    assert get_data._DATA is parent_cache
    (arrow_store.root / "time_1h" / "latest.arrow").unlink()
    assert GetData().get_time_bars("1h_time_bars").height == 2
    current, changed = serve._raw_for("1h_time_bars")
    assert current.height == 2 and changed != version


@pytest.mark.parametrize("budget", [0, 1])
def test_disabled_raw_ram_cache_preserves_versions_and_result_hits(arrow_store, monkeypatch, budget):
    arrow_store.write("time_1h", [ns(2024, 1, 1)])
    monkeypatch.setattr(serve, "_RAW_CACHE", ByteLRU(budget, lambda x: x[0].estimated_size()))
    monkeypatch.setattr(serve, "_RESULTS", {})
    payload, frame = serve._ensure("1h_time_bars", "none", [])
    version = serve._raw_for("1h_time_bars")[1]
    for _ in range(3):
        again, result = serve._ensure("1h_time_bars", "none", [])
        assert again is payload and result is frame
        assert serve._raw_for("1h_time_bars")[1] == version


def test_evicted_raw_frame_preserves_result_version(arrow_store, monkeypatch):
    arrow_store.write("time_1h", [ns(2024, 1, 1)])
    arrow_store.write("time_15m", [ns(2024, 1, 1)])
    size = GetData().get_time_bars("1h_time_bars").estimated_size()
    monkeypatch.setattr(serve, "_RAW_CACHE", ByteLRU(size, lambda x: x[0].estimated_size()))
    monkeypatch.setattr(serve, "_RESULTS", {})
    payload, frame = serve._ensure("1h_time_bars", "none", [])
    serve._raw_for("15m_time_bars")
    assert "1h_time_bars" not in serve._RAW_CACHE
    again, result = serve._ensure("1h_time_bars", "none", [])
    assert again is payload and result is frame


def test_cross_process_refresh_cannot_publish_old_bars_after_newer_bars(arrow_store, tmp_path):
    times = [ns(2024, 1, 1), ns(2024, 1, 2)]
    arrow_store.write("time_1h", times[:1])
    ready, release = tmp_path / "ready", tmp_path / "release"
    script = """
import sys, time
from mill.get_data import GetData
original = GetData._load_arrow
def paused(self, *args):
    frame = original(self, *args)
    Path(sys.argv[1]).touch()
    deadline = time.monotonic() + 20
    while not Path(sys.argv[2]).exists():
        if time.monotonic() > deadline: raise RuntimeError('release deadline')
        time.sleep(0.005)
    return frame
GetData._load_arrow = paused
print(GetData().get_time_bars('1h_time_bars').height)
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(ready), str(release)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        until = time.monotonic() + 10
        while not ready.exists():
            assert child.poll() is None and time.monotonic() < until
            time.sleep(0.005)
        source = arrow_store.root / "time_1h" / "latest.arrow"
        source.rename(source.with_name("old.arrow"))
        arrow_store.write("time_1h", times)
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(GetData().get_time_bars, "1h_time_bars")
            time.sleep(0.05)
            release.touch()
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 0, stderr
            assert stdout.strip() == "1"
            assert pending.result(timeout=5).height == 2
        source.unlink()
        # Both RAM and disk must retain the newer publication through an outage.
        assert GetData().get_time_bars("1h_time_bars").height == 2
        get_data._DATA.clear()
        assert GetData().get_time_bars("1h_time_bars").height == 2
    finally:
        release.touch()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=1)
