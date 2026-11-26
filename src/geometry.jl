"""Return a vector of cell-center coordinates for each logical grid axis."""
axiscenters(g::AbstractGrid) = map(e -> [min(e[i]+(e[i+1]-e[i])/2,prevfloat(e[i+1])) for i in 1:length(e)-1], edges(g))
"""Return a vector of cell widths for each logical grid axis (angles in radians)."""
cellwidths(g::AbstractGrid) = map(diff, edges(g))

"""Allocate cell lengths, areas, or volumes in the same axis order as `size(grid)`."""
function cellvolumes(g::AbstractRectangularGrid{T,N}) where {T,N}
    widths = cellwidths(g)
    out = Array{T}(undef, size(g))
    for i in CartesianIndices(out)
        out[i] = prod(d -> widths[d][i[d]], 1:N)
    end
    return out
end
function cellvolumes(g::PolarGrid)
    r, theta = edges(g)
    return [(r[i+1]-r[i]) * (r[i]+(r[i+1]-r[i])/2) * (theta[j+1]-theta[j])
            for i in 1:size(g,1), j in 1:size(g,2)]
end
"""Allocate cell areas for a two-dimensional grid."""
cellareas(g::AbstractGrid{T,2}) where T = cellvolumes(g)

"""Allocate Cartesian cell centers as a vector of `SVector{ndims(grid),eltype(grid)}` points."""
function cellcenters(g::AbstractRectangularGrid{T,N}) where {T,N}
    centers = axiscenters(g)
    out = Vector{SVector{N,T}}(undef, length(g))
    for (p, i) in enumerate(CartesianIndices(g))
        out[p] = SVector{N,T}(ntuple(d -> centers[d][i[d]], Val(N)))
    end
    return out
end
function cellcenters(g::PolarGrid{T}) where T
    r, theta = axiscenters(g)
    out = Vector{SVector{2,T}}(undef, length(g))
    for (p,i) in enumerate(CartesianIndices(g))
        s, c = sincos(theta[i[2]])
        out[p] = g.center + r[i[1]]*SVector{2,T}(c,s)
    end
    return out
end

"""Allocate the Cartesian coordinates of cell centers as a tuple of arrays shaped `size(grid)`."""
function coordinategrid(g::AbstractGrid{T,N}) where {T,N}
    points = cellcenters(g)
    return ntuple(d -> reshape([p[d] for p in points], size(g)), Val(N))
end

"""Allocate unique rectangular cell corners as a vector of Cartesian `SVector` points."""
function cellcorners(g::AbstractRectangularGrid{T,N}) where {T,N}
    e = edges(g)
    shape = map(length, e)
    out = Vector{SVector{N,T}}(undef, prod(shape))
    for (p,i) in enumerate(CartesianIndices(shape))
        out[p] = SVector{N,T}(ntuple(d -> e[d][i[d]], Val(N)))
    end
    return out
end

"""
    circlemask(grid, radius; center=(0, 0), mode=:intersect)

Classify 2D rectangular cells as entirely `:inside`, entirely `:outside`, or
touching/crossing (`:intersect`) a circumference. Circles contained within a cell
are detected, and tangency counts as intersection. Results have shape `size(grid)`.
"""
function circlemask(g::AbstractRectangularGrid{T,2}, radius::Real;
                     center=zero(SVector{2,Int}), mode::Symbol=:intersect) where T
    isfinite(radius) && radius >= 0 || throw(ArgumentError("radius must be nonnegative and finite"))
    length(center) == 2 && all(isfinite, center) || throw(ArgumentError("center must be a finite pair"))
    mode in (:inside,:outside,:intersect) || throw(ArgumentError("invalid circle mask mode"))
    x, y = edges(g)
    cx, cy = center
    out = falses(size(g))
    @inbounds for j in 1:size(g,2), i in 1:size(g,1)
        nearest = hypot(max(x[i]-cx, zero(cx), cx-x[i+1]), max(y[j]-cy, zero(cy), cy-y[j+1]))
        farthest = hypot(max(abs(x[i]-cx),abs(x[i+1]-cx)), max(abs(y[j]-cy),abs(y[j+1]-cy)))
        inside, outside = farthest < radius, nearest > radius
        out[i,j] = mode === :inside ? inside : mode === :outside ? outside : !(inside || outside)
    end
    return out
end

"""
    rand([rng], grid)
    rand([rng], grid, n)
    rand([rng], grid, n, batches)
    rand!(points, grid)
    rand!(rng, points, grid)

Sample Cartesian `SVector` points uniformly in physical space. With no dimensions,
return one point; with `n`, return a vector; with `n, batches`, return a matrix
of points. `rand!` also accepts real coordinate arrays shaped `(D, P)` or
`(D, P, batches)`. Polar sampling is uniform in area. An explicit RNG is supported.
"""
Random.rand(rng::AbstractRNG, g::AbstractGrid) = _randpoint(rng, g, eltype(g))
Random.rand(g::AbstractGrid) = rand(Random.default_rng(), g)

function Random.rand(rng::AbstractRNG, g::AbstractGrid{T,N}, dims::Dims) where {T,N}
    length(dims) in (1,2) || throw(DimensionMismatch("provide one or two sample dimensions"))
    all(>=(0), dims) || throw(ArgumentError("sample counts must be nonnegative"))
    return rand!(rng, Array{SVector{N,T}}(undef, dims), g)
end
Random.rand(rng::AbstractRNG, g::AbstractGrid, n::Integer) = rand(rng, g, (Int(n),))
Random.rand(rng::AbstractRNG, g::AbstractGrid, n::Integer, batches::Integer) = rand(rng, g, (Int(n),Int(batches)))
Random.rand(g::AbstractGrid, n::Integer) = rand(Random.default_rng(), g, n)
Random.rand(g::AbstractGrid, n::Integer, batches::Integer) = rand(Random.default_rng(), g, n, batches)
Random.rand(g::AbstractGrid, dims::Dims) = rand(Random.default_rng(), g, dims)
Random.rand!(points::AbstractArray{<:AbstractFloat}, g::AbstractGrid) = rand!(Random.default_rng(), points, g)
Random.rand!(points::AbstractArray{<:SVector}, g::AbstractGrid) = rand!(Random.default_rng(), points, g)

@inline function _randpoint(rng, g::AbstractRectangularGrid{G,N}, ::Type{T}) where {G,N,T}
    limits = bounds(g)
    return SVector{N,T}(ntuple(Val(N)) do d
        lo, hi = limits[d]
        min(lo+(hi-lo)*rand(rng,T), prevfloat(T(hi)))
    end)
end

@inline function _randpoint(rng, g::PolarGrid, ::Type{T}) where T
    a, b = first(g.edges[1]), last(g.edges[1])
    ta, tb = first(g.edges[2]), last(g.edges[2])
    u = rand(rng,T)
    r = b*sqrt((a/b)^2 + (1-(a/b)^2)*u)
    theta = ta+(tb-ta)*rand(rng,T)
    s, c = sincos(theta)
    return SVector{2,T}(g.center + r*SVector(c,s))
end

function Random.rand!(rng::AbstractRNG, points::AbstractArray{SVector{N,T}}, g::AbstractGrid) where {N,T<:AbstractFloat}
    _validate_points(g, points)
    @inbounds for p in 1:length(points)
        points[p] = _randpoint(rng, g, T)
    end
    return points
end

function Random.rand!(rng::AbstractRNG, points::AbstractArray{T}, g::AbstractGrid{G,N}) where {T<:AbstractFloat,G,N}
    _validate_points(g, points)
    @inbounds for p in 1:prod(Base.tail(size(points)))
        point = _randpoint(rng, g, T)
        for d in 1:N
            points[d+(p-1)*N] = point[d]
        end
    end
    return points
end

"""Return 2D grid polylines as vectors of Cartesian `SVector` points."""
function gridlines(g::AbstractRectangularGrid{T,2}; arcpoints::Integer=200) where T
    x, y = edges(g)
    lines = Vector{SVector{2,T}}[]
    for a in x
        push!(lines, [SVector{2,T}(a,first(y)), SVector{2,T}(a,last(y))])
    end
    for b in y
        push!(lines, [SVector{2,T}(first(x),b), SVector{2,T}(last(x),b)])
    end
    return lines
end
function gridlines(g::PolarGrid{T}; arcpoints::Integer=200) where T
    arcpoints >= 2 || throw(ArgumentError("arcpoints must be at least two"))
    r, theta = edges(g)
    lines = Vector{SVector{2,T}}[]
    for t in theta
        s, c = sincos(t)
        direction = SVector{2,T}(c,s)
        push!(lines, [g.center+first(r)*direction, g.center+last(r)*direction])
    end
    angles = range(first(theta), last(theta); length=arcpoints)
    for radius in r
        line = Vector{SVector{2,T}}(undef, arcpoints)
        for (i,t) in enumerate(angles)
            s, c = sincos(t)
            line[i] = g.center + radius*SVector{2,T}(c,s)
        end
        push!(lines, line)
    end
    return lines
end
