@inline function _edge_index(edges, x)
    isnan(x) && throw(ArgumentError("point coordinates cannot contain NaN"))
    return searchsortedlast(edges, x)
end

@inline function _regular_index(edges, invspacing, n, x)
    isnan(x) && throw(ArgumentError("point coordinates cannot contain NaN"))
    x < first(edges) && return 0
    x >= last(edges) && return n+1
    i = floor(Int, (x-first(edges))*invspacing)+1
    if 1 <= i <= n && x < edges[i]
        i -= 1
    elseif 0 <= i < n && x >= edges[i+1]
        i += 1
    elseif i == n+1 && x < last(edges)
        i = n
    end
    # Translation close to floating point resolution can defeat an arithmetic
    # estimate; the stored boundaries remain authoritative.
    return 1 <= i <= n && edges[i] <= x < edges[i+1] ? i : searchsortedlast(edges, x)
end

"""
    cellindex(grid, point) -> CartesianIndex

Find the 1-based cell containing a Cartesian point (`SVector`, tuple, or vector). Cells are
left-closed and right-open. Index `0` or `size(grid, d)+1` on an axis denotes an
exterior point. NaNs are rejected. Rectangular infinities map to exterior bins.
"""
function cellindex(g::RectangularGrid{T,N}, point) where {T,N}
    length(point) == N || throw(DimensionMismatch("point must have $N components"))
    return CartesianIndex(ntuple(d -> _edge_index(g.edges[d], point[d]), Val(N)))
end
@inline function cellindex(g::RegularGrid{T,N}, point) where {T,N}
    length(point) == N || throw(DimensionMismatch("point must have $N components"))
    e = edges(g)
    return CartesianIndex(ntuple(d -> _regular_index(e[d], g.invspacing[d], length(e[d])-1, point[d]), Val(N)))
end
function cellindex(g::PolarGrid{T}, point) where T
    length(point) == 2 || throw(DimensionMismatch("point must have two components"))
    x, y = SVector{2}(point) - g.center
    (isnan(x) || isnan(y)) && throw(ArgumentError("point coordinates cannot contain NaN"))
    r, theta = hypot(x, y), mod(atan(y, x), T(2π))
    return CartesianIndex(_edge_index(g.edges[1], r), _edge_index(g.edges[2], theta))
end

function _validate_points(g, points::AbstractArray{<:Real})
    Base.require_one_based_indexing(points)
    ndims(points) in (2, 3) || throw(DimensionMismatch("points must have shape (D, P) or (D, P, batches)"))
    size(points, 1) == ndims(g) || throw(DimensionMismatch("first point axis must have $(ndims(g)) components"))
    return Base.tail(size(points))
end

function _validate_points(g, points::AbstractArray{<:StaticVector{N,<:Real}}) where N
    Base.require_one_based_indexing(points)
    N == ndims(g) || throw(DimensionMismatch("points must have $(ndims(g)) components"))
    ndims(points) in (1,2) || throw(DimensionMismatch("static-vector points must be a vector or a (points, batches) matrix"))
    return size(points)
end

_validate_points(g, points::AbstractArray) =
    throw(ArgumentError("points must be real coordinate arrays or arrays of static vectors"))

"""
    cellindices(grid, points)

Find cells for a vector or `(points, batches)` matrix of static-vector positions.
Coordinate arrays shaped `(D, P)` or `(D, P, batches)` are also supported.
The result is an array of `CartesianIndex{D}` with the sample axes. No singleton
batch axis is inserted or removed. An interior index `I` directly indexes a scalar
reduction result as `result[I]`.
"""
function cellindices(g::AbstractGrid{T,N}, points::AbstractArray) where {T,N}
    shape = _validate_points(g, points)
    result = Array{CartesianIndex{N}}(undef, shape)
    return cellindices!(result, g, points)
end

"""Write cell indices into reusable storage with the input's sample axes."""
function cellindices!(destination::AbstractArray{CartesianIndex{N}}, g::AbstractGrid{T,N},
                      points::AbstractArray) where {T,N}
    shape = _validate_points(g, points)
    Base.require_one_based_indexing(destination)
    size(destination) == shape || throw(DimensionMismatch("index destination must have shape $shape"))
    _check_alias(destination, points)
    _cellindices_kernel!(destination, g, points)
    return destination
end

function _cellindices_kernel!(destination, g::AbstractGrid{T,N}, points::AbstractArray{<:Real}) where {T,N}
    @inbounds for p in 1:length(destination)
        point = SVector{N}(ntuple(d -> points[d+(p-1)*N], Val(N)))
        destination[p] = cellindex(g, point)
    end
    return destination
end

function _cellindices_kernel!(destination, g::AbstractGrid, points::AbstractArray{<:StaticVector})
    @inbounds for p in 1:length(destination)
        destination[p] = cellindex(g, points[p])
    end
    return destination
end

"""Test whether a cell index lies in the grid's interior."""
@inline isinside(g::AbstractGrid{T,N}, i::CartesianIndex{N}) where {T,N} =
    all(d -> 1 <= i[d] <= size(g, d), 1:N)

"""Return a Boolean array flagging exterior indices, preserving sample axes."""
outsidemask(g::AbstractGrid, indices::AbstractArray{<:CartesianIndex}) = map(i -> !isinside(g, i), indices)
