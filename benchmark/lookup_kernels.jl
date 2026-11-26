# Compare the former hoisted RegularGrid loops with the shared cellindex loops.
# Run `julia --project=. benchmark/lookup_kernels.jl [point_count]`.
using Grids, StaticArrays, Random, Printf

# Retained here solely as a benchmark reference, not as package implementations.
function hoisted_reference!(out, g::RegularGrid{T,N}, points::AbstractArray{<:StaticVector}) where {T,N}
    e, inverse, dims = edges(g), g.invspacing, size(g)
    @inbounds for p in 1:length(out)
        point = points[p]
        out[p] = CartesianIndex(ntuple(d -> Grids._regular_index(e[d],inverse[d],dims[d],point[d]),Val(N)))
    end
    return out
end
function hoisted_reference!(out, g::RegularGrid{T,N}, points::AbstractArray{<:Real}) where {T,N}
    e, inverse, dims = edges(g), g.invspacing, size(g)
    @inbounds for p in 1:length(out)
        out[p] = CartesianIndex(ntuple(d -> Grids._regular_index(e[d],inverse[d],dims[d],points[d+(p-1)*N]),Val(N)))
    end
    return out
end

function elapsed(f, out, g, points)
    start = time_ns()
    f(out,g,points)
    return (time_ns()-start)/1e6
end

function compare(g, points; repeats=15)
    expected = cellindices(g,points)
    out = similar(expected)
    old, shared = hoisted_reference!, Grids._cellindices_kernel!
    old(out,g,points)
    @assert out == expected
    shared(out,g,points)
    @assert out == expected
    old_times, shared_times = Float64[], Float64[]
    for i in 1:repeats
        # Alternate order to limit systematic cache/frequency differences.
        if isodd(i)
            push!(old_times,elapsed(old,out,g,points))
            push!(shared_times,elapsed(shared,out,g,points))
        else
            push!(shared_times,elapsed(shared,out,g,points))
            push!(old_times,elapsed(old,out,g,points))
        end
    end
    @assert out == expected
    return minimum(old_times), minimum(shared_times),
           @allocated(old(out,g,points)), @allocated(shared(out,g,points))
end

function main(count)
    println("Julia ",VERSION,"; ",count," points; best of 15 warmed, interleaved runs")
    println("type     D layout       hoisted reference    shared cellindex    bytes (reference/shared)")
    for T in (Float32,Float64), n in 1:4
        g = RegularGrid(ntuple(d -> 120÷d,n);extent=ntuple(d -> T(d+1),n))
        points = rand(MersenneTwister(n),g,count)
        for (layout,data) in (("static",points),("columns",stack(points)))
            before,after,old_bytes,new_bytes = compare(g,data)
            @printf("%-7s %2d %-8s %14.3f ms %17.3f ms     %d / %d\n",string(T),n,layout,before,after,old_bytes,new_bytes)
        end
    end
end
main(isempty(ARGS) ? 100_000 : parse(Int,first(ARGS)))
