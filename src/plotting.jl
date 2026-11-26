_makie_required() = throw(ArgumentError("load a Makie backend, e.g. `using CairoMakie`, and pass (axis, grid, …)"))

"""Draw a 2D grid on a Makie Axis; load a Makie backend first."""
plotgrid!(axis, g::AbstractGrid; kwargs...) = _makie_required()
"""Draw Cartesian cell centers on a Makie Axis."""
plotcenters!(axis, g::AbstractGrid; kwargs...) = _makie_required()
"""Draw rectangular cell corners on a Makie Axis."""
plotcorners!(axis, g::AbstractGrid; kwargs...) = _makie_required()
"""Draw static-vector points or coordinate columns and annotate their cell indices on a Makie Axis."""
plotindices!(axis, g::AbstractGrid, points; kwargs...) = _makie_required()
