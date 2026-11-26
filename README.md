# Grids.jl

Grids.jl is a pure Julia reimplementation of the Python package
[`grids`](https://github.com/marcos1561/grids).

Spatial grids and fast per-cell reductions for Julia 1.12+ . Geometric coordinates
use StaticArrays `SVector`s. Makie plotting and YAML file support load through
optional package extensions.

## Quick start

```julia
using Grids, Random, StaticArrays

grid = RegularGrid((40, 20); extent=SVector(4.0, 2.0))
points = rand(MersenneTwister(42), grid, 10_000) # Vector{SVector{2,Float64}}
indices = cellindices(grid, points)            # Vector{CartesianIndex{2}}
values = [sum(abs2, p) for p in points]

counts = cellcounts(grid, indices)             # 40 × 20
means = cellmean(grid, values, indices)        # 40 × 20
counts[indices[1]]                            # index a cell directly
```

## Constructing grids

Geometry goes directly into a grid constructor:

```julia
regular = RegularGrid((40, 20); extent=SVector(4.0, 2.0), center=SVector(1.0, -2.0))
regular = RegularGrid(((-1.0, 3.0), (-3.0, -1.0)); dims=(40, 20))
regular = RegularGrid((0:0.25:2, -1:0.5:1)) # verifies uniform spacing

rect = RectangularGrid(([-2, -1, 0, 3], [-1, 0, 2]))
polar = PolarGrid([0, 1, 2, 4], [0, π/2, π, 3π/2, 2π]; center=SVector(1, 2))
volume = RegularGrid((20, 30, 40); extent=(2, 3, 4))

rect = RectangularGrid(regular)
regular = RegularGrid(rect) # rejects nonuniform edges
```

Regular and nonuniform rectangular grids support any positive dimensionality.
Polar grids accept Cartesian points and have logical axes `(radius, angle)`.
Lengths and cell counts must be positive; edges must be finite and strictly
increasing. Convenience constructors copy supplied edge vectors. Typed inner
constructors validate canonical fields. Treat stored geometry as read-only.

Centers, physical points, extents, and uniform spacing are concrete `SVector`s.
Constructors also accept tuples and ordinary vectors for coordinate arguments.
Dimensions, bounds by axis, and collections of axis edges/widths remain tuples;
discrete cell indices remain `CartesianIndex` values.

Grid objects contain ordinary fields. Derived quantities use explicit functions:

| Operation | Result |
| --- | --- |
| `size(g)`, `ndims(g)`, `length(g)` | Cell counts, spatial dimensions, total cells |
| `edges(g)` | Stored axis edge vectors |
| `bounds(g)`, `extent(g)` | Tuple of rectangular axis bounds; physical lengths as an `SVector` |
| `gridcenter(g)` | Cartesian grid center as an `SVector` |
| `axiscenters(g)`, `cellwidths(g)` | Per-axis center and width vectors |
| `cellcenters(g)`, `cellcorners(g)` | Vectors of Cartesian `SVector` points; corners are rectangular-only |
| `coordinategrid(g)` | Tuple of Cartesian center arrays, each shaped `size(g)` |
| `cellvolumes(g)`, `cellareas(g)` | Explicitly allocated measures; areas require 2D |

Dense geometry is allocated when its function is called. Store results locally
when reusing them. There are no overloaded property accessors or stored aliases.

## Cell lookup and axis order

`cellindex(g, point)` returns one [CartesianIndex](https://docs.julialang.org/en/v1/base/arrays/#Base.IteratorsMD.CartesianIndex); `cellindices(g, points)` maps
a vector or matrix of static-vector points. Cells are 1-based, left-closed, and right-open. The lower boundary
is included; the upper boundary is outside. An exterior index has coordinate
`0` or `size(g, d)+1` on an axis. NaN positions raise an error.

All arrays use geometric axis order: rectangular cells are indexed `[x, y, z, …]`
and polar cells `[radius, angle]`. This is also the orientation Makie heatmaps use.
For example, `heatmap!(axis, axiscenters(g)..., cellareas(g))` needs no transpose.

## Scalar, tensor, and batched reductions

Each point is an `SVector{D,T}`. A vector of points represents one collection;
a `(P, B)` matrix represents batches. For component-array values, sample axes
come last. Tensor components come first, making the components
of each sample contiguous in Julia's [column-major storage](https://docs.julialang.org/en/v1/manual/performance-tips/#Access-arrays-in-memory-order,-along-columns).

| Quantity | Single input | Explicit batches |
| --- | --- | --- |
| Cartesian `SVector` points | `(P,)` | `(P, B)` |
| Cell indices | `(P,)` | `(P, B)` |
| Scalar values | `(P,)` | `(P, B)` |
| Tensor values | `(value_shape..., P)` | `(value_shape..., P, B)` |
| Count result | `size(g)` | `(size(g)..., B)` |
| Sum/mean result | `(value_shape..., size(g)...)` | `(value_shape..., size(g)..., B)` |

The sample axes determine output rank, even for a single point or a single batch.
There is no implicit batch axis and no shape-simplification flag.

Real coordinate matrices `(D, P)` and arrays `(D, P, B)` remain accepted by
`cellindices`, `cellindices!`, and `rand!`. For a vector of points, `stack(points)`
produces the equivalent coordinate-column matrix.

```julia
vectors = rand(3, length(points))
sums = cellsum(grid, vectors, indices) # 3 × 40 × 20

positions = rand(grid, 100, 2) # Matrix{SVector{2,Float64}}
tensors = rand(2, 2, 100, 2)
batch_indices = cellindices(grid, positions)
batch_means = cellmean(grid, tensors, batch_indices; lengths=[100, 65])
# size(batch_means) == (2, 2, 40, 20, 2)
```

Vector-valued data can also use `SVector` elements directly:

```julia
displacements = points .- Ref(gridcenter(grid))
mean_displacement = cellmean(grid, displacements, indices)
# 40 × 20 array with SVector{2,Float64} values
```

`lengths` is a count of valid samples per batch. Padding is ignored, including NaN
values after the last valid sample. Integer means return floating point results,
Boolean sums use integers, and empty cells have zero mean. Custom additive types
can pass `init` to `cellsum`; addition must return the new value.

Exterior samples are controlled by one keyword:

| `outside` | Behavior |
| --- | --- |
| `:ignore` (default) | Skip exterior samples; output spatial axes equal `size(g)` |
| `:error` | Raise `BoundsError` for an exterior sample |
| `:include` | Add lower and upper exterior bins on each spatial axis |

With `:include`, bins are ordered **lower, interior, upper**, so an interior cell
coordinate `i` maps to array index `i+1`. Use `cellshape(g; outside=:include)` to
size padded buffers. `isinside(g, I)` tests an index; `outsidemask(g, indices)`
preserves the sample axes. Ordinary `filter(I -> isinside(g, I), indices)` selects
interior samples without a package-specific filtering API.

## Reusing buffers

```julia
index_buffer = similar(indices)
count_buffer = zeros(Int, size(grid))
sum_buffer = zeros(3, size(grid)...)
mean_buffer = similar(sum_buffer)

cellindices!(index_buffer, grid, points)
cellcounts!(count_buffer, grid, index_buffer)
cellsum!(sum_buffer, grid, vectors, index_buffer)
cellmean!(mean_buffer, grid, vectors, index_buffer; count_buffer)
```

`!` functions reset and fill their destination. Passing `count_buffer` fuses
counting into summation; `counts=count_buffer` instead uses existing counts
without modifying them. Without either, `cellmean!` allocates scratch counts.
Buffers must have the exact expected shape and must not alias inputs. Independent
tasks can use independent buffers; the kernels do not spawn threads internally.

## Sampling, masks, and plotting

`rand(rng, grid)` returns one `SVector` point, `rand(rng, grid, n)` returns a vector
of points, and `rand(rng, grid, n, batches)` returns a matrix of points.
`rand!(rng, points, grid)` fills static-point arrays or real coordinate arrays.
Sampling is uniform in physical space, including area-uniform polar sampling.
`circlemask(rect, radius; center=(0, 0), mode=:intersect)` classifies cells by
their actual rectangle/circle intersection. Other modes are `:inside` and
`:outside`; tangency counts as intersection.

```julia
using Grids, CairoMakie

figure = Figure()
axis = Axis(figure[1, 1]; aspect=DataAspect())
plotgrid!(axis, grid)
plotcenters!(axis, grid)
plotcorners!(axis, grid)
plotindices!(axis, grid, rand(grid, 10))
save("grid.png", figure)
```

Plotting supports 2D grids; corners are rectangular-only. `gridlines(g)` returns
polylines as vectors of `SVector` points for any other plotting backend.

## Saving and loading

```julia
savegrid("grid.toml", grid)
restored = loadgrid("grid.toml")

using YAML # optional application dependency
savegrid("grid.yaml", grid)
```

IO streams accept `savegrid(io, grid; format=:toml)` and
`loadgrid(io; format=:toml)`. Files use a plain, versioned schema and preserve
Float16/32/64 coordinate types. Version-1 files from the initial port remain
readable. Python-specific YAML object dumps must be exported as plain data first.

## Tests and benchmarks

The root `[workspace]` includes `test/Project.toml`, which declares every direct
test dependency and uses `Grids = {path = ".."}` in `[sources]` to reference the checkout.
Pkg uses a single root manifest. The standard run includes YAML and Makie checks.

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=test test/runtests.jl
julia --project=test examples/plotting.jl
julia --project=. benchmark/benchmarks.jl 100000
python3 benchmark/python_baseline.py /path/to/python/grids 100000
```

Tests include independent reductions in 1D–4D, boundaries, shape contracts,
non-contiguous views, alias checks, portable IO, rendered plots, and allocation
growth. Original numerical fixtures are converted to native axis order only in
the tests; Julia tests need no Python runtime. Regenerate the fixtures with
`python3 test/python_reference.py /path/to/python/grids`.
See [measured results](benchmark/RESULTS.md) and the runnable `examples/` scripts.

## Migrating from the initial port

Construct grids directly: `RegularGrid`, `RectangularGrid`, and `PolarGrid` replace
the configuration/getter layer. `cellindices` replaces coordinate matrices with
Cartesian indices, and `cellcounts`, `cellsum`, and `cellmean` use natural spatial
axis order. Use arrays of `SVector` points, or transpose old point matrices and
move value/sample axes as shown above. Replace automatic attribute aliases with explicit geometry functions,
and use `rand`, `savegrid`, and the plotting functions documented here.
