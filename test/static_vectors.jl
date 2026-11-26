@testset "Static geometry and structural tuples" begin
    for T in (Float16,Float32,Float64), n in 1:4
        dims = ntuple(_ -> 2,n)
        ext = SVector{n,T}(ntuple(_ -> 4,n))
        center = SVector{n,T}(ntuple(_ -> 1,n))
        g = @inferred RegularGrid(dims;extent=ext,center)
        @test g.invspacing isa SVector{n,T}
        @test g.spacing isa SVector{n,T}
        @test extent(g) === ext
        @test gridcenter(g) === center
        @test g.spacing == ext ./ SVector(dims)
        @test size(g) isa NTuple{n,Int}
        @test edges(g) isa NTuple{n,Vector{T}}
        @test bounds(g) isa NTuple{n,Tuple{T,T}}
        @test eltype(cellcenters(g)) === SVector{n,T}
        @test eltype(cellcorners(g)) === SVector{n,T}
        rect = RectangularGrid(g)
        @test gridcenter(rect) === center && extent(rect) === ext
    end
    center = MVector(1f0,2f0)
    polar = @inferred PolarGrid(Float32[0,1,2],Float32[0,π,2π];center)
    @test polar.center isa SVector{2,Float32}
    @test gridcenter(polar) === SVector(1f0,2f0)
    center[1] = 99
    @test polar.center[1] == 1
    @test isconcretetype(fieldtype(typeof(polar),:center))
    @test_throws DimensionMismatch RegularGrid((2,2);extent=SVector(1.0))
    @test_throws DimensionMismatch PolarGrid([0,1],[0,π];center=SVector(1.0))
end

@testset "Static-point collections and coordinate-array interoperability" begin
    for g in (RegularGrid((4,3);extent=(4,3)),
              RectangularGrid(([-2,-1,0,2],[-1,0,1,3])),
              PolarGrid([0,1,2],[0,π,2π];center=SVector(0.5,-1.0)),
              RegularGrid((2,3,4);extent=(2,3,4)))
        n = ndims(g)
        rng = MersenneTwister(7)
        point = @inferred rand(rng,g)
        @test point isa SVector{n,eltype(g)}
        @test isinside(g,cellindex(g,point))
        points = @inferred rand(MersenneTwister(7),g,20)
        @test first(points) === point
        coordinates = stack(points)
        @test cellindices(g,points) == cellindices(g,coordinates)
        @test cellindices(g,MVector.(points)) == cellindices(g,points)
        @test rand!(MersenneTwister(7),similar(points),g) == points
        @test rand!(MersenneTwister(7),zeros(n,20),g) == coordinates
        @test rand(g) isa SVector{n,eltype(g)}
        @test rand(g,2) isa Vector{SVector{n,eltype(g)}}
        @test size(rand(g,(3,2))) == (3,2)
        batches = rand(MersenneTwister(7),g,20,2)
        batch_coordinates = cat(coordinates,stack(batches[:,2]);dims=3)
        @test batches[:,1] == points
        indices = cellindices(g,batches)
        @test size(indices) == (20,2)
        @test indices == cellindices(g,batch_coordinates)
        @test cellindices!(similar(indices),g,batches) == indices
        @test cellindices(g,@view batches[1:2:end,:]) == indices[1:2:end,:]
        @test cellindices(g,@view points[1:2:end]) == indices[1:2:end,1]
        empty = rand(g,0,2)
        @test size(cellindices(g,empty)) == (0,2)
        @test all(iszero,cellcounts(g,cellindices(g,empty)))
        @test all(line -> eltype(line) === SVector{2,eltype(g)}, n == 2 ? gridlines(g) : ())
    end
    g = RegularGrid((2,2);extent=(2,2))
    @test_throws DimensionMismatch cellindices(g,[SVector(0.0,0.0,0.0)])
    @test_throws DimensionMismatch cellindices!(Vector{CartesianIndex{2}}(undef,2),g,[SVector(0.0,0.0)])
    @test_throws DimensionMismatch cellindices(g,fill(SVector(0.0,0.0),2,2,2))
    @test_throws DimensionMismatch rand!(Vector{SVector{3,Float64}}(undef,2),g)
    @test_throws ArgumentError rand(g,2,-1)
end

@testset "Vector-valued reductions retain SVector values" begin
    g = RegularGrid((3,2);extent=(3,2))
    points = rand(MersenneTwister(3),g,20)
    indices = cellindices(g,points)
    vectors = points .- Ref(gridcenter(g))
    sums = cellsum(g,vectors,indices)
    means = cellmean(g,vectors,indices)
    @test eltype(sums) === eltype(vectors)
    @test eltype(means) === eltype(vectors)
    component_sums = cellsum(g,stack(vectors),indices)
    component_means = cellmean(g,stack(vectors),indices)
    for i in CartesianIndices(g)
        @test sums[i] ≈ SVector{2}(component_sums[:,i])
        @test means[i] ≈ SVector{2}(component_means[:,i])
    end
    @test cellmean!(similar(means),g,vectors,indices;count_buffer=zeros(Int,size(g))) ≈ means
    batches = rand(MersenneTwister(3),g,20,2)
    batch_indices = cellindices(g,batches)
    @test size(cellmean(g,batches,batch_indices)) == (size(g)...,2)
    @test cellmean(g,batches,batch_indices)[:,:,1] ≈ cellmean(g,points,indices)
end
