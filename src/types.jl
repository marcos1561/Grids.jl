"""A spatial grid with coordinate type `T` and `N` cell axes."""
abstract type AbstractGrid{T,N} end
abstract type AbstractRectangularGrid{T,N} <: AbstractGrid{T,N} end

"""
    RectangularGrid((xedges, yedges, ...))

Construct a rectangular grid with arbitrary cell widths. Edge vectors are copied
and must be finite and strictly increasing. Cell axes are always in `(x, y, …)`
order. Treat stored geometry as read-only after construction.
"""
struct RectangularGrid{T<:AbstractFloat,N} <: AbstractRectangularGrid{T,N}
    edges::NTuple{N,Vector{T}}
    function RectangularGrid{T,N}(edges::NTuple{N,Vector{T}}) where {T<:AbstractFloat,N}
        N > 0 || throw(ArgumentError("at least one spatial axis is required"))
        foreach(_validate_edges, edges)
        new{T,N}(edges)
    end
end
function RectangularGrid(edges::Tuple)
    all(e -> e isa AbstractVector{<:Real}, edges) ||
        throw(ArgumentError("edges must be a tuple of real vectors or ranges"))
    T = _float_type(map(eltype, edges)...)
    return RectangularGrid{T,length(edges)}(map(e -> collect(T, e), edges))
end

"""
    RegularGrid(dims; extent, center=zero(SVector{length(dims),Int}))
    RegularGrid(bounds; dims)
    RegularGrid((xedges, yedges, ...))

Construct a uniformly spaced rectangular grid in any dimension. For example,
`RegularGrid((40, 20); extent=(4.0, 2.0))`, or
`RegularGrid(((-2.0, 2.0), (-1.0, 1.0)); dims=(40, 20))`.
Edge-based construction validates uniform spacing. Edge vectors and inverse
spacing are stored once for fast arithmetic lookup and exact boundary checks.
"""
struct RegularGrid{T<:AbstractFloat,N} <: AbstractRectangularGrid{T,N}
    edges::NTuple{N,Vector{T}}
    invspacing::SVector{N,T}
    spacing::SVector{N,T}
    function RegularGrid{T,N}(edges::NTuple{N,Vector{T}}, invspacing::SVector{N,T}) where {T<:AbstractFloat,N}
        isconcretetype(T) || throw(ArgumentError("coordinate type must be concrete"))
        N > 0 || throw(ArgumentError("at least one spatial axis is required"))
        for d in 1:N
            e = edges[d]
            _validate_edges(e)
            ideal = range(first(e), last(e); length=length(e))
            tolerance = 16eps(T) * (last(e)-first(e))
            all(i -> isapprox(e[i], ideal[i]; atol=tolerance, rtol=0), eachindex(e)) ||
                throw(ArgumentError("edges must be uniformly spaced"))
            isfinite(invspacing[d]) && invspacing[d] > 0 && invspacing[d] == (length(e)-1)/(last(e)-first(e)) ||
                throw(ArgumentError("inverse spacing must agree with edges"))
        end
        new{T,N}(edges, invspacing, invspacing.^-1)
    end
end
function RegularGrid(limits::NTuple{N,Tuple}; dims::NTuple{N,Integer}) where N
    N > 0 || throw(ArgumentError("at least one spatial axis is required"))
    all(b -> length(b) == 2 && all(x -> x isa Real, b), limits) ||
        throw(ArgumentError("each axis needs a pair of real bounds"))
    T = _float_type(map(b -> typeof(b[1]), limits)..., map(b -> typeof(b[2]), limits)...)
    b = map(v -> (T(v[1]), T(v[2])), limits)
    n = map(Int, dims)
    all(>(0), n) || throw(ArgumentError("cell counts must be positive"))
    all(v -> isfinite(v[1]) && isfinite(v[2]) && v[1] < v[2], b) ||
        throw(ArgumentError("bounds must be finite and increasing"))
    e = ntuple(d -> collect(range(b[d]...; length=n[d]+1)), Val(N))
    inverse = SVector{N,T}(ntuple(d -> n[d]/(b[d][2]-b[d][1]), Val(N)))
    return RegularGrid{T,N}(e, inverse)
end
function RegularGrid(dims::NTuple{N,Integer}; 
    extent=nothing, spacing=nothing, center=zero(SVector{N,Int})) where N
    if (extent !== nothing && spacing !== nothing) || (extent === nothing && spacing === nothing) 
        throw(ArgumentError("Exactly one of extent and spacing must be provided"))
    end

    length(center) == N || throw(DimensionMismatch("center must have $N components"))

    if extent !== nothing
        length(extent) == N || throw(DimensionMismatch("extent must have $N components"))
        all(x -> isfinite(x) && x > 0, extent) ||
            throw(ArgumentError("extents must be positive and finite"))
        T = _float_type(ntuple(d -> typeof(extent[d]), Val(N))..., ntuple(d -> typeof(center[d]), Val(N))...)
        center_vector = SVector{N,T}(ntuple(d -> center[d], Val(N)))
        extent_vector = SVector{N,T}(ntuple(d -> extent[d], Val(N)))
        limits = ntuple(d -> (center_vector[d]-extent_vector[d]/2, center_vector[d]+extent_vector[d]/2), Val(N))
    else
        length(spacing) == N || throw(DimensionMismatch("spacing must have $N components"))
        all(x -> isfinite(x) && x > 0, spacing) ||
            throw(ArgumentError("spacing must be positive and finite"))
        T = _float_type(ntuple(d -> typeof(spacing[d]), Val(N))..., ntuple(d -> typeof(center[d]), Val(N))...)
        center_vector = SVector{N,T}(ntuple(d -> center[d], Val(N)))
        spacing_vector = SVector{N,T}(ntuple(d -> spacing[d], Val(N)))
        limits = ntuple(d -> (center_vector[d]-spacing_vector[d]*dims[d]/2, center_vector[d]+spacing_vector[d]*dims[d]/2), Val(N))
    end
    return RegularGrid(limits; dims)
end
function RegularGrid(e::Tuple{Vararg{AbstractVector}})
    rect = RectangularGrid(e)
    inverse = SVector(map(axis -> (length(axis)-1)/(last(axis)-first(axis)), rect.edges))
    return RegularGrid{eltype(rect),ndims(rect)}(rect.edges, inverse)
end
RegularGrid(g::RectangularGrid) = RegularGrid(edges(g))
RectangularGrid(g::RegularGrid) = RectangularGrid(edges(g))
RegularGrid(::Tuple{}; kwargs...) = throw(ArgumentError("at least one spatial axis is required"))

"""
    PolarGrid(redges, thetaedges; center=(0, 0))

Construct an annular-sector grid from Cartesian point positions. Radial edges are
nonnegative, angular edges lie in `[0, 2π]`, and both are strictly increasing.
Cell axes are `(radius, angle)`. Input vectors are copied.
"""
struct PolarGrid{T<:AbstractFloat} <: AbstractGrid{T,2}
    edges::Tuple{Vector{T},Vector{T}}
    center::SVector{2,T}
    function PolarGrid{T}(edges::Tuple{Vector{T},Vector{T}}, center::SVector{2,T}) where {T<:AbstractFloat}
        isconcretetype(T) || throw(ArgumentError("coordinate type must be concrete"))
        foreach(_validate_edges, edges)
        first(edges[1]) >= 0 || throw(ArgumentError("radial edges must be nonnegative"))
        0 <= first(edges[2]) < last(edges[2]) <= T(2π) ||
            throw(ArgumentError("angular edges must lie in [0, 2π]"))
        all(isfinite, center) || throw(ArgumentError("center must be finite"))
        new{T}(edges, center)
    end
end
function PolarGrid(r::AbstractVector{<:Real}, theta::AbstractVector{<:Real}; center=zero(SVector{2,Int}))
    length(center) == 2 || throw(DimensionMismatch("center must have two components"))
    T = _float_type(eltype(r), eltype(theta), typeof(center[1]), typeof(center[2]))
    return PolarGrid{T}((collect(T, r), collect(T, theta)), SVector{2,T}(center[1], center[2]))
end

"""Return stored axis edge vectors in the same order as `size(grid)`."""
edges(g::AbstractGrid) = g.edges

"""Return `(lower, upper)` bounds for each rectangular axis."""
bounds(g::AbstractRectangularGrid) = map(e -> (first(e), last(e)), edges(g))
"""Return the physical length of each rectangular axis."""
extent(g::AbstractRectangularGrid{T,N}) where {T,N} = SVector{N,T}(map(b -> b[2]-b[1], bounds(g)))
"""Return the Cartesian center of a grid."""
gridcenter(g::AbstractRectangularGrid{T,N}) where {T,N} = SVector{N,T}(map(b -> b[1]+(b[2]-b[1])/2, bounds(g)))
gridcenter(g::PolarGrid) = g.center

Base.ndims(::AbstractGrid{T,N}) where {T,N} = N
Base.eltype(::AbstractGrid{T}) where T = T
Base.size(g::AbstractGrid) = map(e -> length(e)-1, edges(g))
function Base.size(g::AbstractGrid, d::Integer)
    d >= 1 || throw(ArgumentError("axis must be positive"))
    return d > ndims(g) ? 1 : size(g)[d]
end
Base.length(g::AbstractGrid) = prod(size(g))
Base.axes(g::AbstractGrid) = map(Base.OneTo, size(g))
Base.CartesianIndices(g::AbstractGrid) = CartesianIndices(size(g))
function Base.show(io::IO, g::AbstractGrid)
    print(io, nameof(typeof(g)), "(", join(size(g), " × "), " cells, ", eltype(g), ")")
end

"""Return spatial output dimensions; `outside=:include` adds a bin on both sides."""
function cellshape(g::AbstractGrid; outside::Symbol=:ignore)
    _outside_policy(outside)
    return outside === :include ? map(n -> n+2, size(g)) : size(g)
end
