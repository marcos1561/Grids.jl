"""Run: python3 benchmark/python_baseline.py /path/to/python/grids [N]."""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve() / "src"))
import grids


def measure(label, function, samples=3):
    function()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    print(f"{label:28s} {min(times):10.3f} ms")


n = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
grid = grids.RegularRectGridCfg(4, 3, 100, 75).get_grid()
i = np.arange(1, n + 1, dtype=float)
points = np.column_stack((np.mod(i * 0.61803398875, 1) * 4 - 2,
                          np.mod(i * 0.41421356237, 1) * 3 - 1.5))
values = np.column_stack((np.sin(i), np.cos(i), np.mod(i, 17) / 17))
cells = grid.coords(points, simplify_shape=True)
print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}; {n} points; 100 × 75 cells; 3 value components")
print("Best of 3 warmed runs; setup excluded")
measure("coords", lambda: grid.coords(points))
measure("count", lambda: grid.count(cells))
measure("sum_by_cell", lambda: grid.sum_by_cell(values, cells))
measure("mean_by_cell", lambda: grid.mean_by_cell(values, cells))
