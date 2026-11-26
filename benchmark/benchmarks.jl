# Run `julia --project=. benchmark/benchmarks.jl [N]`.
using Grids, Random, Printf, StaticArrays

function measure(label,f;samples=7)
    f()
    elapsed = minimum(1:samples) do _
        start = time_ns()
        f()
        (time_ns()-start)/1e6
    end
    bytes = @allocated f()
    @printf("%-28s %10.3f ms %12d bytes\n",label,elapsed,bytes)
    return elapsed
end

function main(n)
    grid = RegularGrid((100,75);extent=(4,3))
    points,values = Vector{SVector{2,Float64}}(undef,n),Matrix{Float64}(undef,3,n)
    for i in 1:n
        points[i] = SVector(mod(i*0.61803398875,1)*4-2,mod(i*0.41421356237,1)*3-1.5)
        values[1,i],values[2,i],values[3,i] = sin(i),cos(i),mod(i,17)/17
    end
    indices = cellindices(grid,points)
    counts = cellcounts(grid,indices;outside=:include)
    sums = cellsum(grid,values,indices;outside=:include)
    means = similar(sums)
    println("Julia ",VERSION,"; ",n," points; 100 × 75 cells; 3 components; exterior bins included")
    println("Best of 7 warmed runs; setup and compilation excluded")
    measure("cellindices",() -> cellindices(grid,points))
    measure("cellcounts",() -> cellcounts(grid,indices;outside=:include))
    measure("cellsum",() -> cellsum(grid,values,indices;outside=:include))
    measure("cellmean",() -> cellmean(grid,values,indices;outside=:include))
    measure("cellindices!",() -> cellindices!(indices,grid,points))
    measure("cellcounts!",() -> cellcounts!(counts,grid,indices;outside=:include))
    measure("cellsum!",() -> cellsum!(sums,grid,values,indices;outside=:include))
    measure("cellmean! (scratch)",() -> cellmean!(means,grid,values,indices;outside=:include,count_buffer=counts))
    @assert sum(counts) == n
    @assert all(isfinite,sums)
end
main(isempty(ARGS) ? 100_000 : parse(Int,first(ARGS)))
