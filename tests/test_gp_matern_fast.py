import contextlib
import io
import time

import numpy as np
import pytest

from gp_preprocessor import gp_matern_fast
from reference_gp_matern_fast import gp_matern_fast as reference


def _case(n, seed, irregular):
    rng = np.random.default_rng(seed)
    if irregular:
        x = np.sort(rng.uniform(1000.0, 1000.0 + n * 0.1, n))
        x[n // 3] = x[n // 3 - 1]  # a repeated position (dt = 0)
    else:
        x = np.round(np.arange(n) * 0.1 + 1000.0, 4)
    y = np.sin(x / 50) + rng.normal(0, 0.1, n)
    y[::17] = np.nan  # gaps in the training data
    x_test = np.linspace(x[0] - 5.0, x[-1] + 5.0, n // 2)
    return x, y, x_test


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("ell, sigma_f, sigma_n", [(30.0, 1.0, 0.1), (5.0, 0.3, 0.02)])
def test_matches_reference(irregular, ell, sigma_f, sigma_n):
    x, y, x_test = _case(3000, 1, irregular)
    with contextlib.redirect_stdout(io.StringIO()):  # the reference prints
        expected = reference(x, y, x_test, ell, sigma_f, sigma_n)
    actual = gp_matern_fast(x, y, x_test, ell, sigma_f, sigma_n)
    for got, want in zip(actual, expected):
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def test_is_faster_than_reference():
    x, y, x_test = _case(20000, 2, irregular=False)
    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        reference(x, y, x_test, 30.0, 1.0, 0.1)
        reference_s = time.perf_counter() - start
    start = time.perf_counter()
    gp_matern_fast(x, y, x_test, 30.0, 1.0, 0.1)
    new_s = time.perf_counter() - start
    assert new_s * 3 < reference_s
