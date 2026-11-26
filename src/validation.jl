function _float_type(types...)
    T = float(promote_type(types...))
    isconcretetype(T) && T <: AbstractFloat ||
        throw(ArgumentError("coordinates must have a concrete real numeric type"))
    return T
end

function _validate_edges(e)
    length(e) >= 2 || throw(ArgumentError("each axis needs at least two edges"))
    all(isfinite, e) || throw(ArgumentError("edges must be finite"))
    all(i -> e[i] < e[i+1], 1:length(e)-1) ||
        throw(ArgumentError("edges must be strictly increasing"))
    isfinite(last(e)-first(e)) || throw(ArgumentError("grid extents must be finite"))
    return nothing
end

function _outside_policy(outside::Symbol)
    outside in (:ignore, :include, :error) ||
        throw(ArgumentError("outside must be :ignore, :include, or :error"))
    return Val(outside)
end

function _check_alias(destination, inputs...)
    any(a -> a isa AbstractArray && Base.mightalias(destination, a), inputs) &&
        throw(ArgumentError("destination and scratch buffers must not alias inputs or each other"))
    return nothing
end
