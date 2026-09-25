"""
Tests on a synthetic surface where the true shift is known.
Run with:  pytest
"""

import threading
import numpy as np
import pytest
import laspy

from ticp import ICPConfig, TICPCancelled, transicp, run_ticp, compare, compute_normals

SHIFT = np.array([0.8, -0.4, -0.25])


def surface(x, y):
    return 10 * np.sin(x / 40) * np.cos(y / 30) + 3 * np.sin(x / 13 + y / 17)


def make_clouds(n=60_000, size=300, seed=0):
    """ Pre-event cloud and a post-event cloud shifted by SHIFT everywhere """
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, size, n), rng.uniform(0, size, n)
    before = np.column_stack([x, y, surface(x, y) + rng.normal(0, 0.02, n)])
    x, y = rng.uniform(0, size, n), rng.uniform(0, size, n)
    after = np.column_stack([x, y, surface(x - SHIFT[0], y - SHIFT[1]) + SHIFT[2] + rng.normal(0, 0.02, n)])
    return before, after


def write_laz(path, xyz):
    header = laspy.LasHeader(point_format=3, version='1.4')
    header.scales = [0.001] * 3
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = np.full(len(xyz), 2, np.uint8)
    las.write(str(path))


@pytest.fixture(scope='module')
def clouds():
    before, after = make_clouds()
    return before, after, compute_normals(after)


def test_transicp_single_window(clouds):
    before, after, normals = clouds
    wb = (before[:, 0] < 100) & (before[:, 1] < 100)
    wa = (after[:, 0] < 105) & (after[:, 1] < 105)
    shift, rmse, n = transicp(before[wb], after[wa], normals[wa], ICPConfig())
    assert np.allclose(shift, SHIFT, atol=0.05)
    assert rmse < 0.1 and n > 100


def test_too_few_points_gives_nan(clouds):
    before, after, normals = clouds
    shift, rmse, n = transicp(before[:5], after[:5], normals[:5], ICPConfig())
    assert np.all(np.isnan(shift)) and n == 0


def test_serial_and_parallel_agree(clouds):
    before, after, normals = clouds
    config = ICPConfig(window_size=100, step_size=50, n_workers=1)
    r1 = run_ticp(before, after, normals, config)
    config.n_workers = 2
    r2 = run_ticp(before, after, normals, config)
    for k in ['dx', 'dy', 'dz']:
        assert np.allclose(r1[k], r2[k], equal_nan=True)
    assert np.allclose(np.nanmedian(r1['dx']), SHIFT[0], atol=0.05)


def test_cancel(clouds):
    before, after, normals = clouds
    stop = threading.Event()
    stop.set()
    with pytest.raises(TICPCancelled):
        run_ticp(before, after, normals, ICPConfig(window_size=100, step_size=50, n_workers=1), cancel=stop)


def test_compare_writes_files(tmp_path):
    before, after = make_clouds(n=30_000, size=200, seed=1)
    write_laz(tmp_path / 'before.laz', before)
    write_laz(tmp_path / 'after.laz', after)

    calls = []
    result = compare(tmp_path / 'before.laz', tmp_path / 'after.laz', window_size=100, step_size=50,
                     classes=[2], n_workers=1, output=str(tmp_path / 'out' / 'run'),
                     progress=lambda f, m: calls.append((f, m)))

    assert (tmp_path / 'out' / 'run.txt').exists() and (tmp_path / 'out' / 'run.tif').exists()
    assert np.allclose(np.nanmedian(result['dz']), SHIFT[2], atol=0.05)
    assert calls[-1] == (1.0, 'Done')
