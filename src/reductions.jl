@inline _point_limit(::Nothing, batch, n) = n
@inline _point_limit(lengths::Integer, batch, n) = Int(lengths)
@inline _point_limit(lengths::AbstractVector, batch, n) = Int(@inbounds lengths[batch])

function _validate_samples(g, indices::AbstractArray{CartesianIndex{N},B}, lengths) where {N,B}
    N == ndims(g) || throw(DimensionMismatch("cell index dimension must match grid"))
    B in (1, 2) || throw(DimensionMismatch("indices must be a vector or a (points, batches) matrix"))
    Base.require_one_based_indexing(indices)
    points, batches = size(indices, 1), size(indices, 2)
    if lengths isa Integer
        batches == 1 || throw(DimensionMismatch("scalar lengths requires one batch"))
        0 <= lengths <= points || throw(ArgumentError("lengths must lie in 0:$points"))
    elseif lengths isa AbstractVector{<:Integer}
        Base.require_one_based_indexing(lengths)
        length(lengths) == batches || throw(DimensionMismatch("lengths needs one count per batch"))
        all(n -> 0 <= n <= points, lengths) || throw(ArgumentError("lengths must lie in 0:$points"))
    elseif lengths !== nothing
        throw(ArgumentError("lengths must be an integer or an integer vector"))
    end
    return points, batches
end

@inline _batch_shape(indices::AbstractVector) = ()
@inline _batch_shape(indices::AbstractMatrix) = (size(indices, 2),)

function _value_shape(values::AbstractArray{T,V}, indices::AbstractArray{I,B}) where {T,V,I,B}
    Base.require_one_based_indexing(values)
    V >= B || throw(DimensionMismatch("values need trailing sample axes"))
    ntuple(d -> size(values, V-B+d), Val(B)) == size(indices) ||
        throw(DimensionMismatch("trailing value axes must match size(indices)"))
    return ntuple(d -> size(values, d), Val(V-B))
end

function _validate_destination(out, shape)
    Base.require_one_based_indexing(out)
    size(out) == shape || throw(DimensionMismatch("destination must have shape $shape"))
    return nothing
end

@inline function _cell_offset(dims::NTuple{N,Int}, i::CartesianIndex{N}, ::Val{POLICY}) where {N,POLICY}
    offset, stride = 0, 1
    for d in 1:N
        x, n = i[d], dims[d]
        if POLICY === :include
            x = x < 1 ? 1 : x > n ? n+2 : x+1
            n += 2
        elseif !(1 <= x <= n)
            POLICY === :ignore && return -1
            throw(BoundsError(dims, i))
        end
        offset += (x-1)*stride
        stride *= n
    end
    return offset
end

function _count_kernel!(out, dims, indices, lengths, policy, cells)
    points, batches = size(indices, 1), size(indices, 2)
    @inbounds for batch in 1:batches, p in 1:_point_limit(lengths, batch, points)
        cell = _cell_offset(dims, indices[p+(batch-1)*points], policy)
        cell >= 0 && (out[cell+1+(batch-1)*cells] += 1)
    end
    return out
end

"""
    cellcounts(grid, indices; outside=:ignore, lengths=nothing)

Count points by cell. A vector of indices returns `size(grid)`; a `(P, B)` matrix
returns `(size(grid)..., B)`. `outside=:ignore` skips exterior points, `:error`
rejects them, and `:include` adds lower and upper exterior bins on every axis.
In the latter layout, cell coordinate `i` maps to array index `i+1`.
`lengths[b]` limits each batch to its first valid samples.
"""
function cellcounts(g::AbstractGrid, indices::AbstractArray{<:CartesianIndex};
                    outside::Symbol=:ignore, lengths=nothing)
    _validate_samples(g, indices, lengths)
    out = Array{Int}(undef, cellshape(g; outside)..., _batch_shape(indices)...)
    return cellcounts!(out, g, indices; outside, lengths)
end

"""Reset and fill an integer count buffer with the layout of [`cellcounts`](@ref)."""
function cellcounts!(out::AbstractArray{<:Integer}, g::AbstractGrid,
                     indices::AbstractArray{<:CartesianIndex}; outside::Symbol=:ignore, lengths=nothing)
    _validate_samples(g, indices, lengths)
    spatial = cellshape(g; outside)
    _validate_destination(out, (spatial..., _batch_shape(indices)...))
    _check_alias(out, indices, lengths)
    fill!(out, 0)
    _count_kernel!(out, size(g), indices, lengths, _outside_policy(outside), prod(spatial))
    return out
end

function _sum_kernel!(out, counts, dims, values, indices, lengths, policy, cells, components)
    points, batches = size(indices, 1), size(indices, 2)
    @inbounds for batch in 1:batches, p in 1:_point_limit(lengths, batch, points)
        sample = p+(batch-1)*points
        cell = _cell_offset(dims, indices[sample], policy)
        if cell >= 0
            destination = cell+(batch-1)*cells
            counts !== nothing && (counts[destination+1] += 1)
            # Tensor components are contiguous in Julia storage. Locate the cell
            # once per point and accumulate its components without scratch arrays.
            @simd for component in 1:components
                out[component+components*destination] += values[component+components*(sample-1)]
            end
        end
    end
    return out
end

"""
    cellsum(grid, values, indices; outside=:ignore, lengths=nothing, init=zero(eltype(values)))

Sum scalar or tensor values by cell. Values have trailing sample axes matching
`indices`: `(value_shape..., P)` or `(value_shape..., P, batches)`. Results have
shape `(value_shape..., size(grid)...)` with an explicit trailing batch axis only
for batched input. Components stay contiguous in memory. Boolean sums use Int.
Custom additive types may supply `init`; addition must return the new value.
"""
function cellsum(g::AbstractGrid, values::AbstractArray, indices::AbstractArray{<:CartesianIndex};
                 outside::Symbol=:ignore, lengths=nothing, init=zero(eltype(values)))
    _validate_samples(g, indices, lengths)
    shape = (_value_shape(values, indices)..., cellshape(g; outside)..., _batch_shape(indices)...)
    T = eltype(values) === Bool ? Int : eltype(values)
    return cellsum!(Array{T}(undef, shape), g, values, indices; outside, lengths, init)
end

"""Reset and fill a sum buffer; destination comes first, as in other Julia `!` APIs."""
function cellsum!(out::AbstractArray, g::AbstractGrid, values::AbstractArray,
                  indices::AbstractArray{<:CartesianIndex}; outside::Symbol=:ignore,
                  lengths=nothing, init=zero(eltype(out)))
    _validate_samples(g, indices, lengths)
    value_shape = _value_shape(values, indices)
    spatial = cellshape(g; outside)
    _validate_destination(out, (value_shape..., spatial..., _batch_shape(indices)...))
    _check_alias(out, values, indices, lengths)
    fill!(out, init)
    _sum_kernel!(out, nothing, size(g), values, indices, lengths, _outside_policy(outside),
                 prod(spatial), prod(value_shape))
    return out
end

"""
    cellmean(grid, values, indices; counts=nothing, outside=:ignore, lengths=nothing)

Compute per-cell means with the layout of [`cellsum`](@ref). Empty cells remain
zero. Integer inputs produce floating point results. `counts` may supply matching
precomputed cell counts, which are left unchanged.
"""
function cellmean(g::AbstractGrid, values::AbstractArray, indices::AbstractArray{<:CartesianIndex};
                  outside::Symbol=:ignore, lengths=nothing, counts=nothing)
    _validate_samples(g, indices, lengths)
    shape = (_value_shape(values, indices)..., cellshape(g; outside)..., _batch_shape(indices)...)
    T = Base.promote_op(/, eltype(values), Int)
    isconcretetype(T) || throw(ArgumentError("mean type is not concrete; use cellmean! with a typed buffer"))
    return cellmean!(Array{T}(undef, shape), g, values, indices; outside, lengths, counts)
end

"""
    cellmean!(out, grid, values, indices; counts=nothing, count_buffer=nothing, kwargs...)

Reset and fill a mean buffer. `count_buffer` reuses integer scratch storage and
fuses counting into summation. Alternatively, `counts` supplies existing counts.
Without either, a count buffer is allocated. All buffers use the selected
interior or padded layout and must not alias inputs or each other.
"""
function cellmean!(out::AbstractArray, g::AbstractGrid, values::AbstractArray,
                   indices::AbstractArray{<:CartesianIndex}; outside::Symbol=:ignore,
                   lengths=nothing, counts=nothing, count_buffer=nothing)
    _validate_samples(g, indices, lengths)
    value_shape = _value_shape(values, indices)
    spatial = cellshape(g; outside)
    count_shape = (spatial..., _batch_shape(indices)...)
    _validate_destination(out, (value_shape..., count_shape...))
    counts !== nothing && count_buffer !== nothing &&
        throw(ArgumentError("supply either counts or count_buffer"))
    _check_alias(out, values, indices, lengths, counts, count_buffer)
    scratch = counts === nothing ? count_buffer : counts
    scratch === nothing || scratch isa AbstractArray{<:Integer} ||
        throw(ArgumentError("count storage must be an integer array"))
    scratch = scratch === nothing ? Array{Int}(undef, count_shape) : scratch
    _validate_destination(scratch, count_shape)
    if counts === nothing
        _check_alias(scratch, values, indices, lengths)
        fill!(scratch, 0)
    else
        all(>=(0), scratch) || throw(ArgumentError("counts must be nonnegative"))
    end
    fill!(out, zero(eltype(out)))
    components = prod(value_shape)
    _sum_kernel!(out, counts === nothing ? scratch : nothing, size(g), values, indices,
                 lengths, _outside_policy(outside), prod(spatial), components)
    @inbounds for cell in 1:length(scratch)
        n = scratch[cell]
        if n > 0
            @simd for component in 1:components
                out[component+components*(cell-1)] /= n
            end
        end
    end
    return out
end
