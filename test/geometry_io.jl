@testset "Geometry, masks, and Random integration" begin
    g = RectangularGrid(([-2,-1,1,2],[-2,-1,1,2]))
    inside = circlemask(g,1.5;mode=:inside)
    outside = circlemask(g,1.5;mode=:outside)
    intersect = circlemask(g,1.5)
    @test inside[2,2] && count(inside) == 1
    @test all((inside .+ outside .+ intersect) .== 1)
    @test circlemask(g,0.25)[2,2]
    @test !circlemask(g,0.25;mode=:outside)[2,2]
    @test circlemask(g,1)[1,2] # tangency
    @test all(circlemask(g,0.1;center=(10,10),mode=:outside))
    @test_throws ArgumentError circlemask(g,-1)
    @test_throws ArgumentError circlemask(g,1;mode=:unknown)

    regular = RegularGrid((3,2);extent=(3,2))
    polar = PolarGrid([1,2,3],[0.2,1,3,5.8];center=(0.5,-1))
    for grid in (regular,g,polar,RectangularGrid(([0,1],)),RegularGrid((2,3,4);extent=(2,3,4)))
        p = rand(MersenneTwister(2),grid,1000)
        @test size(p) == (1000,) && eltype(p) === SVector{ndims(grid),eltype(grid)}
        @test !any(outsidemask(grid,cellindices(grid,p)))
        @test p == rand(MersenneTwister(2),grid,1000)
        @test rand!(MersenneTwister(2),similar(p),grid) == p
        @test size(rand(grid,0)) == (0,)
        @test size(rand!(zeros(ndims(grid),5,2),grid)) == (ndims(grid),5,2)
        @test size(cellcenters(grid)) == (length(grid),)
        @test all(i -> isinside(grid,i),cellindices(grid,cellcenters(grid)))
        @test all(a -> size(a) == size(grid),coordinategrid(grid))
    end
    p = rand(MersenneTwister(3),polar,50_000)
    r2 = [sum(abs2,point-polar.center) for point in p]
    @test abs(sum(r2)/length(r2)-5) < 0.04
    @test size(cellcorners(regular)) == (12,)
    @test length(gridlines(regular)) == 7
    @test length(gridlines(polar;arcpoints=20)) == 7
    @test size(last(gridlines(polar;arcpoints=20))) == (20,)
    @test_throws ArgumentError rand(regular,-1)
    @test_throws DimensionMismatch rand!(zeros(3,2),regular)
    @test_throws ArgumentError gridlines(polar;arcpoints=1)
end

@testset "Explicit grid file IO" begin
    mktempdir() do dir
        grids = (
            RegularGrid((7,3);extent=(3,2),center=(1,-2)),
            RectangularGrid(([-2,-1,0,3],[0,2,3])),
            RegularGrid((2,3,4);extent=(2,3,4)),
            RectangularGrid(([0,1],[1,2],[-1,0,1])),
            PolarGrid([0,1,2],[0,π/2,2π];center=(-1,2)),
            RegularGrid((7,3);extent=(3f0,2f0)),
            PolarGrid(Float32[0,1,2],Float32[0,2π]),
        )
        for (n,g) in enumerate(grids)
            path = joinpath(dir,"$n.toml")
            @test savegrid(path,g) === nothing
            restored = loadgrid(path)
            @test typeof(restored) === typeof(g)
            @test edges(restored) == edges(g)
            @test gridcenter(restored) === gridcenter(g)
            p = rand(MersenneTwister(1),g,20)
            @test cellindices(restored,p) == cellindices(g,p)
            io = IOBuffer()
            savegrid(io,g)
            seekstart(io)
            @test edges(loadgrid(io)) == edges(g)
        end
        path = joinpath(dir,"test.yaml")
        write(path,"existing content")
        @test_throws ArgumentError savegrid(path,first(grids))
        @test read(path,String) == "existing content"
        @test_throws ArgumentError loadgrid(IOBuffer("type = \"unknown\""))
        @test_throws ArgumentError loadgrid(IOBuffer("format_version = 99"))
        @test_throws ArgumentError savegrid(joinpath(dir,"test.txt"),first(grids))
    end
    old = IOBuffer("""
        format_version = 1
        type = "regular_rectangular"
        length = 4.0
        height = 3.0
        num_cols = 4
        num_rows = 3
        center = [0.25, -0.5]
        """)
    @test edges(loadgrid(old)) == edges(RegularGrid((4,3);extent=(4,3),center=(0.25,-0.5)))
    @test_throws ArgumentError plotgrid!(nothing,first((RegularGrid((2,2);extent=(1,1)),)))
end
