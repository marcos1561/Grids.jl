@testset "Direct grid constructors and ordinary fields" begin
    g = @inferred RegularGrid((4,3); extent=(4,3), center=(1,2))
    @test size(g) == (4,3)
    @test size(g,3) == 1
    @test ndims(g) == 2 && eltype(g) === Float64
    @test length(g) == 12
    @test bounds(g) == ((-1,3),(0.5,3.5))
    @test extent(g) == SVector(4,3)
    @test gridcenter(g) == SVector(1,2)
    @test edges(g) == (-1.0:1:3, 0.5:1:3.5)
    @test axes(g) == (Base.OneTo(4),Base.OneTo(3))
    @test CartesianIndices(g) == CartesianIndices((4,3))
    @test cellshape(g; outside=:include) == (6,5)
    @test propertynames(g) == fieldnames(typeof(g))
    @test occursin("RegularGrid", sprint(show,g))
    @test edges(RegularGrid(edges(g))) == edges(g)
    @test edges(RectangularGrid(g)) == edges(g)
    @test edges(RegularGrid(RectangularGrid(g))) == edges(g)
    @test_throws ArgumentError size(g,0)

    for T in (Float16,Float32,Float64), n in 1:4
        dims = ntuple(_ -> 2,n)
        grid = @inferred RegularGrid(dims; extent=ntuple(_ -> T(4),n))
        @test eltype(grid) === T
        @test size(grid) == dims
        @test sum(cellvolumes(grid)) ≈ T(4)^n
    end

    e = (Float32[0,1,3],Float32[-1,0,2,5])
    r = @inferred RectangularGrid(e)
    @test r.edges == e && r.edges[1] !== e[1] && r.edges[2] !== e[2]
    @test cellwidths(r) == ([1,2],[1,2,3])
    @test axiscenters(r) == ([0.5,2],[-0.5,1,3.5])
    @test cellareas(r) == [1 2 3; 2 4 6]
    @test coordinategrid(r) == ([0.5 0.5 0.5; 2 2 2],[-0.5 1 3.5; -0.5 1 3.5])
    e[1][1] = -100
    @test first(edges(r)[1]) == 0
    @test propertynames(r) == (:edges,)

    redges, theta = Float32[0,1,2],Float32[0,π,2π]
    polar = @inferred PolarGrid(redges,theta; center=Float32[2,3])
    @test eltype(polar) === Float32
    @test edges(polar)[1] !== redges && edges(polar)[2] !== theta
    @test gridcenter(polar) == SVector(2,3)
    @test size(polar) == (2,2)
    @test sum(cellareas(polar)) ≈ 4π

    @test_throws ArgumentError RectangularGrid(())
    @test_throws ArgumentError RegularGrid(();extent=())
    for bad in ([0.0],[1.0,0.0],[0.0,0.0],[0.0,NaN],[0.0,Inf],[-floatmax(Float64),0,floatmax(Float64)])
        @test_throws ArgumentError RectangularGrid((bad,))
        @test_throws ArgumentError RectangularGrid{Float64,1}((bad,))
    end
    @test_throws ArgumentError RectangularGrid{Float64,0}(())
    @test_throws ArgumentError RegularGrid((0,2); extent=(1,1))
    @test_throws ArgumentError RegularGrid((2,2); extent=(-1,1))
    @test_throws ArgumentError RegularGrid((2,2); extent=(1,Inf))
    @test_throws ArgumentError RegularGrid((2,2); extent=(1,1),center=(NaN,0))
    @test_throws DimensionMismatch RegularGrid((2,2); extent=(1,1),center=(0,))
    @test_throws ArgumentError RegularGrid(([0,1,3],[0,1]))
    @test_throws ArgumentError RegularGrid(([1e16,1e16+2,1e16+8],))
    @test_throws ArgumentError RegularGrid{Float64,1}(([0.0,0.5,1.0],),SVector(1.0))
    @test_throws ArgumentError PolarGrid([-1,1],[0,π])
    @test_throws ArgumentError PolarGrid([0,1],[-π,π])
    @test_throws ArgumentError PolarGrid([0,1],[0,3π])
    @test_throws ArgumentError PolarGrid{Float64}(([0.0,1.0],[0.0,π]),SVector(NaN,0.0))
    @test_throws ArgumentError cellshape(g; outside=:unknown)
end
