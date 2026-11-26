# Local benchmark results

Measured on 2026-11-25 on an Intel Core i5-10400 (2.90 GHz), using Julia 1.13.4,
StaticArrays 1.9.25, Python 3.14.3, and NumPy 2.4.3. Both implementations processed the same 100,000
points on a 100 × 75 regular grid with three value components and exterior bins.
Julia uses `Vector{SVector{2,Float64}}` points and component-first values; Python uses the
original package's layouts. Setup, data generation, and compilation are excluded.
The implementations ran separately, reporting the best of seven warmed Julia
runs and three warmed Python runs.

| Allocating operation | Python | Julia | Speedup |
| --- | ---: | ---: | ---: |
| Cell lookup | 1.954 ms | 2.187 ms | 0.89× |
| Count | 85.587 ms | 0.150 ms | 571× |
| Sum, three components | 664.488 ms | 0.635 ms | 1,046× |
| Mean, three components | 745.202 ms | 0.713 ms | 1,045× |

| Julia operation with reused buffers | Time | Allocated bytes per call |
| --- | ---: | ---: |
| `cellindices!` | 1.736 ms | 0 |
| `cellcounts!` | 0.152 ms | 48 |
| `cellsum!` | 0.528 ms | 48 |
| `cellmean!`, including count scratch | 0.719 ms | 48 |

The fixed allocations do not grow with point count; the test suite checks this.
These results describe this workload and machine. Other grid sizes, tensor
shapes, and runtimes will have different timings.

Reproduce from the package directory:

```sh
julia --startup-file=no --project=. benchmark/benchmarks.jl 100000
python3 benchmark/python_baseline.py /path/to/python/grids 100000
```

## Shared cell lookup kernel

The following comparison was measured after replacing the grid-specific lookup
loops with shared loops calling `cellindex`. The tables above retain the earlier
baseline. The reference hoists the regular grid's edges, inverse spacing, and
dimensions before the loop, as the former implementation did. The shared version
inlines `cellindex` and obtains each cell count directly from its axis edges.

For 100,000 `SVector` points with `Float64` coordinates, best of 15 warmed,
interleaved runs on the same machine and Julia version:

| Dimensions | Former hoisted loop | Shared `cellindex` loop |
| --- | ---: | ---: |
| 1 | 1.002 ms | 0.751 ms |
| 2 | 1.628 ms | 1.518 ms |
| 3 | 2.580 ms | 2.491 ms |
| 4 | 3.426 ms | 3.559 ms |

Both loops allocated zero bytes in every measured case. The benchmark also
checks `Float32` and coordinate-column arrays: the shared version was faster in
1–3 dimensions and about 3–4% slower in 4 dimensions. This supports using the
simpler shared implementation for the common workloads; it does not imply equal
performance for every dimension or compiler version.

Reproduce the comparison, including equality and allocation checks:

```sh
julia --startup-file=no --project=. benchmark/lookup_kernels.jl 100000
```

## Velocity field layouts

Measured on the same machine and Julia version, comparing `Matrix{Float64}`
with shape `(D, P)` against `Vector{SVector{D,Float64}}`. Both layouts contain
identical values and use the same cell indices. Each run uses one batch and
7,500 cells: a 100 × 75 grid for 2D velocities, or 25 × 20 × 15 for 3D velocities.
The outside policy is `:ignore`. Data generation, layout conversion, cell lookup,
and compilation are excluded. Times are the best of 21 warmed, interleaved runs.
The benchmark checks equality of means and fused counts.

| Dimensions | Samples | Operation | Matrix | Vector of SVectors |
| --- | ---: | --- | ---: | ---: |
| 2 | 100,000 | `cellmean` | 0.443 ms | 0.411 ms |
| 2 | 100,000 | `cellmean!` | 0.456 ms | 0.402 ms |
| 3 | 100,000 | `cellmean` | 0.693 ms | 0.609 ms |
| 3 | 100,000 | `cellmean!` | 0.637 ms | 0.576 ms |
| 2 | 1,000,000 | `cellmean` | 4.960 ms | 4.682 ms |
| 2 | 1,000,000 | `cellmean!` | 5.028 ms | 4.593 ms |
| 3 | 1,000,000 | `cellmean` | 7.224 ms | 6.937 ms |
| 3 | 1,000,000 | `cellmean!` | 7.217 ms | 6.761 ms |

The `!` calls reuse both the output and count scratch buffers. They allocate
48 bytes per call for either layout, independent of sample count. Allocating
calls use approximately 180 KB in 2D and 240 KB in 3D for either layout.
Both input layouts store each velocity's components contiguously; the static
vector type additionally encodes the number of components at compile time.
These results show a modest advantage for static-vector values in this workload.

Reproduce, including minimum and median times and allocation measurements:

```sh
julia --startup-file=no --project=. benchmark/mean_layouts.jl
```
