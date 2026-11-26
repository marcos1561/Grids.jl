using Grids, Test, YAML, CairoMakie, StaticArrays
import Makie

@testset "YAML extension" begin
    mktempdir() do dir
        for (n,g) in enumerate((
            RegularGrid((4,2);extent=(2,1)),
            RectangularGrid(([0,1,3],[-1,0,2])),
            PolarGrid([0,1,2],[0,π,2π]),
            RegularGrid((4,2);extent=(2f0,1f0)),
        ))
            path = joinpath(dir,"$n.yaml")
            savegrid(path,g)
            restored = loadgrid(path)
            @test typeof(restored) === typeof(g)
            @test edges(restored) == edges(g)
            @test YAML.load_file(path)["format_version"] == 2
        end
    end
end

@testset "Makie extension and native heatmap orientation" begin
    g = RegularGrid((4,2);extent=(2,1))
    polar = PolarGrid([0,1,2],[0,π/2,π,2π])
    figure = Figure(size=(800,350))
    axis = Axis(figure[1,1];aspect=DataAspect())
    axis2 = Axis(figure[1,2];aspect=DataAspect())
    @test plotgrid!(axis,g) isa Makie.Lines
    @test plotcenters!(axis,g) isa Makie.Scatter
    @test plotcorners!(axis,g) isa Makie.Scatter
    @test plotindices!(axis,g,[-0.75 0.5; -0.25 0.25]) isa Makie.Scatter
    @test plotgrid!(axis2,polar;color=:blue) isa Makie.Lines
    @test plotcenters!(axis2,polar) isa Makie.Scatter
    @test plotindices!(axis2,polar,[SVector(0.5,0.0),SVector(-0.5,0.5)]) isa Makie.Scatter
    x,y = axiscenters(g)
    @test heatmap!(axis,x,y,cellareas(g)) isa Makie.Heatmap
    mktempdir() do dir
        path = joinpath(dir,"grid.png")
        save(path,figure)
        @test filesize(path) > 1000
    end
end
