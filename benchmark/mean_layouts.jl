# Run `julia --startup-file=no --project=. benchmark/mean_layouts.jl [point_count ...]`.
# Compare one-batch velocity fields with identical values and cell indices.
using Grids, StaticArrays, Random, Printf, Statistics

function elapsed_mean(f)
    start = time_ns()
    f()
    return (time_ns()-start)/1e6
end

function compare_means(matrix_call, static_call; repeats=21)
    matrix_call()
    static_call()
    GC.gc()
    matrix_times, static_times = Float64[], Float64[]
    for i in 1:repeats
        if isodd(i)
            push!(matrix_times, elapsed_mean(matrix_call))
            push!(static_times, elapsed_mean(static_call))
        else
            push!(static_times, elapsed_mean(static_call))
            push!(matrix_times, elapsed_mean(matrix_call))
        end
    end
    return minimum(matrix_times), minimum(static_times),
           median(matrix_times), median(static_times),
           @allocated(matrix_call()), @allocated(static_call())
end

function velocity_case(::Val{D}, n) where D
    rng = MersenneTwister(42)
    dims = D == 2 ? (100,75) : (25,20,15)
    grid = RegularGrid(dims; extent=ntuple(_ -> 2.0,D))
    indices = cellindices(grid, rand(rng,grid,n))
    matrix_values = randn(rng,D,n)
    static_values = [SVector{D}(matrix_values[:,p]) for p in 1:n]
    matrix_out = cellmean(grid,matrix_values,indices)
    static_out = cellmean(grid,static_values,indices)
    @assert reshape(matrix_out,D,:) ≈ stack(vec(static_out))
    matrix_counts, static_counts = zeros(Int,dims), zeros(Int,dims)

    calls = (
        ("cellmean", () -> cellmean(grid,matrix_values,indices),
                     () -> cellmean(grid,static_values,indices)),
        ("cellmean!", () -> cellmean!(matrix_out,grid,matrix_values,indices;count_buffer=matrix_counts),
                      () -> cellmean!(static_out,grid,static_values,indices;count_buffer=static_counts)),
    )
    for (name,matrix_call,static_call) in calls
        mbest,sbest,mmedian,smedian,mbytes,sbytes = compare_means(matrix_call,static_call)
        @printf("%dD %8d %-9s %8.3f %8.3f %8.3f %8.3f  %8d / %-8d\n",
                D,n,name,mbest,sbest,mmedian,smedian,mbytes,sbytes)
    end
    @assert reshape(matrix_out,D,:) ≈ stack(vec(static_out))
    @assert matrix_counts == static_counts
    @assert sum(matrix_counts) == n
end

println("Julia ",VERSION,"; Float64 velocity fields; one batch; 7,500 cells; outside=:ignore")
println("Identical data and indices; conversion, lookup, and compilation excluded")
println("21 warmed, interleaved runs; times in ms; M = Matrix, S = Vector{SVector}")
println("D    samples operation    M best   S best M median S median  bytes (M / S)")
for n in (isempty(ARGS) ? [100_000,1_000_000] : parse.(Int,ARGS))
    velocity_case(Val(2),n)
    velocity_case(Val(3),n)
end
