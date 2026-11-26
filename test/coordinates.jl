@testset "Point columns and Cartesian cell indices" begin
    g = RegularGrid((2,2);extent=(2,2))
    p = [-2.0 -1 -0.1 0 0.9 1 Inf; -2 -1 0 0 0.9 1 -Inf]
    expected = CartesianIndex.([(0,0),(1,1),(1,2),(2,2),(2,2),(3,3),(3,0)])
    @test cellindices(g,p) == expected
    @test cellindices(RectangularGrid(g),p) == expected
    @test cellindices!(similar(expected),g,p) == expected
    @test @inferred(cellindex(g,(0.0,0.0))) == CartesianIndex(2,2)
    @test cellindex(g,[0.0,0.0]) == CartesianIndex(2,2)
    @test size(cellindices(g,zeros(2,0))) == (0,)
    @test size(cellindices(g,zeros(2,0,3))) == (0,3)
    @test size(cellindices(g,zeros(2,3,0))) == (3,0)
    batches = cat(p,p;dims=3)
    @test cellindices(g,batches) == hcat(expected,expected)
    @test outsidemask(g,expected) == [true,false,false,false,false,true,true]
    @test filter(i -> isinside(g,i),expected) == expected[2:5]
    @test size(cellindices(g,reshape(p,2,7,1))) == (7,1)
    @test cellindices(g,@view p[:,1:2:end]) == expected[1:2:end]
    buffer = fill(CartesianIndex(0,0),14)
    @test cellindices!(@view(buffer[1:2:end]),g,p) == expected

    for T in (Float32,Float64), center in ((0,0),(3.25,-7.75))
        regular = RegularGrid((37,29);extent=(T(3.7),T(2.9)),center=T.(center))
        rect = RectangularGrid(regular)
        for d in 1:2
            e = edges(regular)[d]
            q = repeat(reshape(collect(T,gridcenter(regular)),2,1),1,3length(e))
            q[d,:] = vcat(prevfloat.(e),collect(e),nextfloat.(e))
            @test cellindices(regular,q) == cellindices(rect,q)
        end
        q = rand(MersenneTwister(4),regular,10_000)
        @test cellindices(regular,q) == cellindices(rect,q)
    end
    translated = RegularGrid((3,3);extent=(6,6),center=(1e16,-1e16))
    @test cellindices(translated,cellcenters(translated)) == cellindices(RectangularGrid(translated),cellcenters(translated))
    @test cellindices(translated,cellcenters(translated)) == vec(collect(CartesianIndices(translated)))
    tiny = nextfloat(0.0)
    subnormal = RectangularGrid(([-2tiny,-tiny],))
    @test all(i -> isinside(subnormal,i),cellindices(subnormal,cellcenters(subnormal)))

    polar = PolarGrid([0,1,2],[0,π/2,π,3π/2,2π];center=(1,2))
    q = [1.0 1.5 1 0.5 1 3; 2 2 2.5 2 1.5 2]
    @test cellindices(polar,q) == CartesianIndex.([(1,1),(1,1),(1,2),(1,3),(1,4),(3,1)])
    sector = PolarGrid([1,2],[π/2,π])
    @test cellindices(sector,[0.0 1.5 0 -1.5; 0 0 1.5 0]) == CartesianIndex.([(0,0),(1,0),(1,1),(1,2)])
    @test_throws ArgumentError cellindex(g,(NaN,0.0))
    @test_throws ArgumentError cellindex(polar,(0.0,NaN))
    @test_throws DimensionMismatch cellindex(g,(0.0,))
    @test_throws DimensionMismatch cellindices(g,zeros(2))
    @test_throws DimensionMismatch cellindices(g,zeros(3,2))
    @test_throws DimensionMismatch cellindices!(fill(CartesianIndex(0,0),1),g,p)
end
