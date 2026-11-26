module Grids

using Random
using StaticArrays: SVector, StaticVector
using TOML
export AbstractGrid, AbstractRectangularGrid, RectangularGrid, RegularGrid, PolarGrid,
       edges, bounds, extent, gridcenter, spacing, cellshape, axiscenters, cellwidths,
       cellindex, cellindices, cellindices!, isinside, outsidemask,
       cellcounts, cellcounts!, cellsum, cellsum!, cellmean, cellmean!,
       cellcenters, cellcorners, coordinategrid, cellvolumes, cellareas, circlemask,
       gridlines, savegrid, loadgrid,
       plotgrid!, plotcenters!, plotcorners!, plotindices!

include("validation.jl")
include("types.jl")
include("coordinates.jl")
include("reductions.jl")
include("geometry.jl")
include("io.jl")
include("plotting.jl")

end
