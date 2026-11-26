using Grids
using Test
using Random
using StaticArrays
using TOML

include("constructors.jl")
include("coordinates.jl")
include("static_vectors.jl")
include("reductions.jl")
include("geometry_io.jl")
include("python_reference.jl")
include("performance.jl")

# Load optional packages last so the core suite also exercises the fallback
# behavior before the extensions are loaded.
include("extensions.jl")
