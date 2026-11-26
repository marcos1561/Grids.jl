using Grids, Random, StaticArrays

grid = RegularGrid((40,20);extent=SVector(4.0,2.0))
points = rand(MersenneTwister(42),grid,10_000) # Vector{SVector{2,Float64}}
indices = cellindices(grid,points)
values = [sum(abs2,p) for p in points]
counts = cellcounts(grid,indices)
means = cellmean(grid,values,indices)
@assert sum(counts) == length(points)
@assert size(means) == size(grid) == (40,20)
println(grid, ": accumulated ",sum(counts)," points")

# Tensor components come first and stay contiguous in memory.
vectors = vcat(transpose(values),transpose(2values),transpose(3values))
buffer = zeros(3,size(grid)...)
cellmean!(buffer,grid,vectors,indices;counts)
@assert buffer[1,:,:] ≈ means

# Batches are explicit trailing axes, including unequal sample counts.
positions = rand(MersenneTwister(43),grid,100,2)
tensors = rand(MersenneTwister(44),2,2,100,2)
batch_indices = cellindices(grid,positions)
batch_means = cellmean(grid,tensors,batch_indices;lengths=[100,65])
@assert size(batch_means) == (2,2,40,20,2)
